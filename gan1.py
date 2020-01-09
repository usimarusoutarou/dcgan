import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers ,reporter, cuda
from chainer.training import extensions
from chainer.datasets import TupleDataset
import numpy as np
import os
import math
from numpy import random
from PIL import Image

batch_size = 10			# バッチサイズ10
uses_device = 0			# GPU#0を使用

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda
else:
	cp = np

xp = cuda.cupy
def calc_mean_std(feature, eps = 1e-5):
    batch, channels, _, _ = feature.shape
    feature_a = feature.data
    feature_var = xp.var(feature_a.reshape(batch, channels, -1),axis = 2) + eps
    feature_var = chainer.as_variable(feature_var)
    feature_std = F.sqrt(feature_var).reshape(batch, channels, 1,1)
    feature_mean = F.mean(feature.reshape(batch, channels, -1), axis = 2)
    feature_mean = feature_mean.reshape(batch, channels, 1,1)

    return feature_std, feature_mean

def adain(content_feature, style_feature):
    shape = content_feature.shape
    style_std, style_mean = calc_mean_std(style_feature)
    style_mean = F.broadcast_to(style_mean, shape = shape)
    style_std = F.broadcast_to(style_std, shape = shape)
    
    content_std, content_mean = calc_mean_std(content_feature)
    content_mean = F.broadcast_to(content_mean, shape = shape)
    content_std = F.broadcast_to(content_std, shape = shape)
    normalized_feat = (content_feature - content_mean) / content_std

    return normalized_feat * style_std + style_mean

class CBR(chainer.Chain):

	def __init__(self, in_ch, out_ch):
		w = chainer.initializers.GlorotUniform()
		super(CBR, self).__init__()
		with self.init_scope():
			self.c0 = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
			self.bn0 = L.BatchNormalization(out_ch)

	def __call__(self, x):
		h = F.relu(self.bn0(self.c0(x)))

		return h

class ResBlock(chainer.Chain):

	def __init__(self, in_ch, out_ch):
		w = chainer.initializers.GlorotUniform()
		super(ResBlock, self).__init__()
		with self.init_scope():
			self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
			self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

			self.bn0 = L.BatchNormalization(out_ch)
			self.bn1 = L.BatchNormalization(out_ch)

	def __call__(self, x):
		h = F.relu(self.bn0(self.c0(x)))
		h = self.bn1(self.c1(h))

		return h + x

class AdainResBlock(chainer.Chain):

	def __init__(self, in_ch, out_ch):
		w = chainer.initializers.GlorotUniform()
		super(AdainResBlock, self).__init__()
		with self.init_scope():
			self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
			self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

	def __call__(self, x, z):
		h = F.relu(adain(self.c0(x), z))
		h = F.relu(adain(self.c1(h), z))

		return h + x

class Upsamp(chainer.Chain):

	def __init__(self, in_ch, out_ch):
		w = chainer.initializers.GlorotUniform()
		super(Upsamp, self).__init__()
		with self.init_scope():
			self.d0 = L.Deconvolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
			self.d1 = L.Deconvolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
			self.c0 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
			self.bnd0 = L.BatchNormalization(out_ch)
			self.bnd1 = L.BatchNormalization(out_ch)
			self.bnc0 = L.BatchNormalization(out_ch)

	def __call__(self, x):
		h = F.relu(self.bnd0(self.d0(x)))
		h = F.relu(self.bnd1(self.d1(h)))
		h = F.relu(self.bnc0(self.c0(h)))

		return h

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self,base = 32):
		# 重みデータの初期値を指定する
		w = chainer.initializers.GlorotUniform()
		# 全ての層を定義する
		super(DCGAN_Generator_NN, self).__init__()

		with self.init_scope():

			#ヒント画像
			self.x2c0 = L.Convolution2D(3, base, 3, 1, 1, initialW=w)
			self.x2bnc0 = L.BatchNormalization(base)
			self.x2cbr0 = CBR(base, base*2)
			self.x2cbr1 = CBR(base*2, base*4)
			self.x2cbr2 = CBR(base*4, base*8)
			self.x2cbr3 = CBR(base*8, base*16)

			# Input layer
			self.c0 = L.Convolution2D(3, base, 3, 1, 1, initialW=w)
			self.bnc0 = L.BatchNormalization(base)

			# UNet
			self.cbr0 = CBR(base, base*2)
			self.cbr1 = CBR(base*2, base*4)
			self.cbr2 = CBR(base*4, base*8)
			self.cbr3 = CBR(base*8, base*16)
			self.cbr4 = CBR(base*32, base*16)
			self.res0 = ResBlock(base*16, base*16)
			self.res1 = ResBlock(base*16, base*16)
			self.ad0 = AdainResBlock(base*16, base*16)
			self.ad1 = AdainResBlock(base*16, base*16)
			self.ad2 = AdainResBlock(base*16, base*16)
			self.ad3 = AdainResBlock(base*16, base*16)
			self.up0 = Upsamp(base*16, base*16)
			self.up1 = Upsamp(base*16, base*8)
			self.up2 = Upsamp(base*16, base*4)
			self.up3 = Upsamp(base*8, base*2)
			self.up4 = Upsamp(base*4, base)

			# Output layer
			self.c1 = L.Convolution2D(base*2, 3, 3, 1, 1, initialW=w)

	def __call__(self, x1,x2):
		x2e0 = F.relu(self.x2bnc0(self.x2c0(x2)))
		x2e1 = self.x2cbr0(x2e0)
		x2e2 = self.x2cbr1(x2e1)
		x2e3 = self.x2cbr2(x2e2)
		x2e4 = self.x2cbr3(x2e3)

		#U-Net
		e0 = F.relu(self.bnc0(self.c0(x1)))
		e1 = self.cbr0(e0)
		e2 = self.cbr1(e1)
		e3 = self.cbr2(e2)
		e4 = self.cbr3(e3)
		e5 = self.cbr4(F.concat([e4, x2e4]))
		r0 = self.res0(e5)
		r1 = self.res1(r0)
		a0 = self.ad0(r1,x2e4)
		a1 = self.ad1(a0,x2e4)
		a2 = self.ad2(a1,x2e4)
		a3 = self.ad3(a2,x2e4)
		d0 = self.up0(a3)
		d1 = self.up1(d0)
		d2 = self.up2(F.concat([d1, e3]))
		d3 = self.up3(F.concat([d2, e2]))
		d4 = self.up4(F.concat([d3, e1]))
		d5 = F.sigmoid(self.c1(F.concat([d4,e0])))
		
		return d5	# 結果を返すのみ

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.GlorotUniform()
		super(DCGAN_Discriminator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.c0_0 = L.Convolution2D(3, 8, 3, 1, 1, initialW=w)
			self.c0_1 = L.Convolution2D(8, 16, 4, 2, 1, initialW=w)
			self.c1_0 = L.Convolution2D(16, 16, 3, 1, 1, initialW=w)
			self.c1_1 = L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
			self.c2_0 = L.Convolution2D(32, 32, 3, 1, 1, initialW=w)
			self.c2_1 = L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
			self.c3_0 = L.Convolution2D(64, 64, 3, 1, 1, initialW=w)
			self.l4 = L.Linear(128 * 128, 1, initialW=w)
			self.bn0_1 = L.BatchNormalization(16, use_gamma=False)
			self.bn1_0 = L.BatchNormalization(16, use_gamma=False)
			self.bn1_1 = L.BatchNormalization(32, use_gamma=False)
			self.bn2_0 = L.BatchNormalization(32, use_gamma=False)
			self.bn2_1 = L.BatchNormalization(64, use_gamma=False)
			self.bn3_0 = L.BatchNormalization(64, use_gamma=False)

	def __call__(self, x):
		h = F.leaky_relu(self.c0_0(x))
		h = F.dropout(F.leaky_relu(self.bn0_1(self.c0_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_0(self.c1_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_1(self.c1_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_0(self.c2_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_1(self.c2_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn3_0(self.c3_0(h))),ratio=0.2)
		return self.l4(h)	# 結果を返すのみ

# カスタムUpdaterのクラス
class DCGANUpdater(training.StandardUpdater):

	def __init__(self, train_iter, optimizer, device):
		super(DCGANUpdater, self).__init__(
			train_iter,
			optimizer,
			device=device
		)
	
	# 画像認識側の損失関数
	def loss_dis(self, dis, y_fake, y_real):
		batchsize = len(y_fake)
		L1 = F.sum(F.softplus(-y_real)) / batchsize
		L2 = F.sum(F.softplus(y_fake)) / batchsize
		loss = L1 + L2
		reporter.report({'dis_loss':loss})
		return loss

	# 画像生成側の損失関数
	def loss_gen(self, gen, y_fake):
		batchsize = len(y_fake)
		loss = F.sum(F.softplus(-y_fake)) / batchsize
		reporter.report({'gen_loss':loss})
		return loss

	def update_core(self):
		# Iteratorからバッチ分のデータを取得
		batch = self.get_iterator('main').next()
		src = self.converter(batch, self.device)
		
		# Optimizerを取得
		optimizer_gen = self.get_optimizer('opt_gen')
		optimizer_dis = self.get_optimizer('opt_dis')
		# ニューラルネットワークのモデルを取得
		gen = optimizer_gen.target
		dis = optimizer_dis.target

		# 画像を生成して認識と教師データから認識
		x_fake = gen(src[0],src[1])		# 線画からの生成結果
		y_fake = dis(x_fake)	# 線画から生成したものの識別結果
		y_real = dis(src[1])		# 着色画像の識別結果

		# ニューラルネットワークを学習
		optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)
		optimizer_gen.update(self.loss_gen, gen, y_fake)
		optimizer_gen.update(self.loss_gen, gen, y_fake)
		
# ニューラルネットワークを作成
model_gen = DCGAN_Generator_NN()
model_dis = DCGAN_Discriminator_NN()

if uses_device >= 0:
	# GPUを使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	# GPU用データ形式に変換
	model_gen.to_gpu()
	model_dis.to_gpu()

#chainer.serializers.load_hdf5( 'gan-gen.hdf5', model_gen )
#chainer.serializers.load_hdf5( 'gan-dis.hdf5', model_dis )

listdataset1 = []
listdataset2 = []

fs = os.listdir('/home/nagalab/soutarou/dcgan/images')
fs.sort()

for fn in fs:
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('/home/nagalab/soutarou/dcgan/images/' + fn).convert('RGB').resize((128, 128))

	if 'png' in fn:
		# 画素データを0〜1の領域にする
		hpix1 = np.array(img, dtype=np.float32) / 255.0
		hpix1 = hpix1.transpose(2,0,1)
		listdataset1.append(hpix1)
	else:
		# 画素データを0〜1の領域にする
		hpix2 = np.array(img, dtype=np.float32) / 255.0
		hpix2= hpix2.transpose(2,0,1)
		listdataset2.append(hpix2)

# 配列に追加
tupledataset1 = tuple(listdataset1)
tupledataset2 = tuple(listdataset2)
dataset = TupleDataset(tupledataset1, tupledataset2)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(dataset, batch_size, shuffle=True)

# 誤差逆伝播法アルゴリズムを選択する
optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_gen.setup(model_gen)
optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(model_dis)

# デバイスを選択してTrainerを作成する
updater = DCGANUpdater(train_iter, \
		{'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, \
		device=uses_device)
trainer = training.Trainer(updater, (50000, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport(trigger=(500, 'epoch'), log_name='log'))
trainer.extend(extensions.PlotReport(['dis_loss', 'gen_loss'], x_key='epoch', file_name='loss.png'))

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(100, 'epoch'))
def save_model(trainer):
	# NNのデータを保存
	global n_save
	n_save = n_save+1
	chainer.serializers.save_hdf5( 'gan-gen-'+str(n_save)+'.hdf5', model_gen )
	chainer.serializers.save_hdf5( 'gan-dis-'+str(n_save)+'.hdf5', model_dis )
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'gan-gen.hdf5', model_gen )
chainer.serializers.save_hdf5( 'gan-dis.hdf5', model_dis )

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers ,reporter
from chainer.training import extensions
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

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Generator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.c0 = L.Convolution2D(3, 64, 3, 8, 1, initialW=w)
			self.dc0 = L.Deconvolution2D(64, 32, 4, 2, 1, initialW=w)
			self.dc1 = L.Deconvolution2D(32, 16, 4, 2, 1, initialW=w)
			self.dc2 = L.Deconvolution2D(16, 8, 4, 2, 1, initialW=w)
			self.dc3 = L.Deconvolution2D(8, 3, 3, 1, 1, initialW=w)
			self.bn0 = L.BatchNormalization(64)
			self.bn1 = L.BatchNormalization(32)
			self.bn2 = L.BatchNormalization(16)
			self.bn3 = L.BatchNormalization(8)

	def __call__(self, z):
		h = F.relu(self.bn0(self.c0(z)))
		h = F.relu(self.bn1(self.dc0(h)))
		h = F.relu(self.bn2(self.dc1(h)))
		h = F.relu(self.bn3(self.dc2(h)))
		x = F.sigmoid(self.dc3(h))
		return x	# 結果を返すのみ

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
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

		# 乱数データを用意
		rnd = random.uniform(-1, 1, (src.shape[0], 100))
		rnd = cp.array(rnd, dtype=cp.float32)
		
		# 画像を生成して認識と教師データから認識
		x_fake = gen(src[1])		# 乱数からの生成結果
		y_fake = dis(x_fake)	# 乱数から生成したものの認識結果
		y_real = dis(src[0])		# 教師データからの認識結果

		# ニューラルネットワークを学習
		optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)
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

dataset = []

fs = os.listdir('/home/nagalab/soutarou/dcgan/images')
fs.sort()

for fn in fs:
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('/home/nagalab/soutarou/dcgan/images/' + fn).convert('RGB').resize((128, 128))

	if 'jpg' in fn:
		# 画素データを0〜1の領域にする
		hpix1 = np.array(img, dtype=np.float32) / 255.0
		hpix1 = hpix1.transpose(2,0,1)
	else:
		# 画素データを0〜1の領域にする
		hpix2 = np.array(img, dtype=np.float32) / 255.0
		hpix2= hpix2.transpose(2,0,1)
		# 配列に追加
		dataset.append([hpix1,hpix2])

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(dataset, batch_size, shuffle=False)

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
@chainer.training.make_extension(trigger=(1000, 'epoch'))
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
chainer.serializers.save_hdf5( 'chapt04.hdf5', model_gen )

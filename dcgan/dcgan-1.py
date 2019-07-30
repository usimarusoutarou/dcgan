import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import os
import math
from numpy import random
from PIL import Image

batch_size = 10			# バッチサイズ10
uses_device = 0			# GPU#0を使用
image_size = 128		# 生成画像のサイズ
neuron_size = 64		# 中間層のサイズ

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda
else:
	cp = np

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self):
		super(DCGAN_Generator_NN, self).__init__(
			c0=L.Convolution2D(4, 32, 3, 1, 1),
            c1=L.Convolution2D(32, 64, 4, 2, 1),
            c2=L.Convolution2D(64, 64, 3, 1, 1),
            c3=L.Convolution2D(64, 128, 4, 2, 1),
            c4=L.Convolution2D(128, 128, 3, 1, 1),
            c5=L.Convolution2D(128, 256, 4, 2, 1),
            c6=L.Convolution2D(256, 256, 3, 1, 1),
            c7=L.Convolution2D(256, 512, 4, 2, 1),
            c8=L.Convolution2D(512, 512, 3, 1, 1),

            dc8=L.Deconvolution2D(1024, 512, 4, 2, 1),
            dc7=L.Convolution2D(512, 256, 3, 1, 1),
            dc6=L.Deconvolution2D(512, 256, 4, 2, 1),
            dc5=L.Convolution2D(256, 128, 3, 1, 1),
            dc4=L.Deconvolution2D(256, 128, 4, 2, 1),
            dc3=L.Convolution2D(128, 64, 3, 1, 1),
            dc2=L.Deconvolution2D(128, 64, 4, 2, 1),
            dc1=L.Convolution2D(64, 32, 3, 1, 1),
            dc0=L.Convolution2D(64, 3, 3, 1, 1),

            bnc0=L.BatchNormalization(32),
            bnc1=L.BatchNormalization(64),
            bnc2=L.BatchNormalization(64),
            bnc3=L.BatchNormalization(128),
            bnc4=L.BatchNormalization(128),
            bnc5=L.BatchNormalization(256),
            bnc6=L.BatchNormalization(256),
            bnc7=L.BatchNormalization(512),
            bnc8=L.BatchNormalization(512),

            bnd8=L.BatchNormalization(512),
            bnd7=L.BatchNormalization(256),
            bnd6=L.BatchNormalization(256),
            bnd5=L.BatchNormalization(128),
            bnd4=L.BatchNormalization(128),
            bnd3=L.BatchNormalization(64),
            bnd2=L.BatchNormalization(64),
            bnd1=L.BatchNormalization(32)
            # l = L.Linear(3*3*256, 2)'
		)
	def calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(F.concat([e0, d1]))

        return d0

	def __call__(self, x, t):
        h = self.calc(x)
        loss = F.mean_absolute_error(h, t)
        chainer.report({'loss': loss}, self)
        return loss

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

	def __init__(self):
		super(DCGAN_Discriminator_NN, self).__init__(
			c1=L.Convolution2D(3, 32, 4, 2, 1),
            c2=L.Convolution2D(32, 32, 3, 1, 1),
            c3=L.Convolution2D(32, 64, 4, 2, 1),
            c4=L.Convolution2D(64, 64, 3, 1, 1),
            c5=L.Convolution2D(64, 128, 4, 2, 1),
            c6=L.Convolution2D(128, 128, 3, 1, 1),
            c7=L.Convolution2D(128, 256, 4, 2, 1),
            l8l=L.Linear(None, 2,
                         initialW=chainer.initializers.HeNormal(
                             math.sqrt(0.02 * math.sqrt(8 * 8 * 256) / 2))),

            bnc1=L.BatchNormalization(32),
            bnc2=L.BatchNormalization(32),
            bnc3=L.BatchNormalization(64),
            bnc4=L.BatchNormalization(64),
            bnc5=L.BatchNormalization(128),
            bnc6=L.BatchNormalization(128),
            bnc7=L.BatchNormalization(256),
		)

	def calc(self, x):
        h = F.relu(self.bnc1(self.c1(x)))
        h = F.relu(self.bnc2(self.c2(h)))
        h = F.relu(self.bnc3(self.c3(h)))
        h = F.relu(self.bnc4(self.c4(h)))
        h = F.relu(self.bnc5(self.c5(h)))
        h = F.relu(self.bnc6(self.c6(h)))
        h = F.relu(self.bnc7(self.c7(h)))
        return self.l8l(h)

	def __call__(self, x, t):
        h = self.calc(x)
        loss = F.softmax_cross_entropy(h, t)
        #chainer.report({'loss': loss }, self)
        return loss


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
		return loss

	# 画像生成側の損失関数
	def loss_gen(self, gen, y_fake):
		batchsize = len(y_fake)
		loss = F.sum(F.softplus(-y_fake)) / batchsize
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
		x_fake = gen(rnd)		# 乱数からの生成結果
		y_fake = dis(x_fake)	# 乱数から生成したものの認識結果
		y_real = dis(src)		# 教師データからの認識結果

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

chainer.serializers.save_hdf5( 'dcgan-gen.hdf5', model_gen )
chainer.serializers.save_hdf5( 'dcgan-dis.hdf5', model_dis )

images = []

fs = os.listdir('/home/nagalab/soutarou/images')
for fn in fs:
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('/home/nagalab/soutarou/images/' + fn).convert('RGB').resize((128, 128))
	# 画素データを0〜1の領域にする
	hpix = np.array(img, dtype=np.float32) / 255.0
	hpix = hpix.transpose(2,0,1)
	# 配列に追加
	images.append(hpix)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)

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

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(1000, 'epoch'))
def save_model(trainer):
	# NNのデータを保存
	global n_save
	n_save = n_save+1
	chainer.serializers.save_hdf5( 'dcgan-gen-'+str(n_save)+'.hdf5', model_gen )
	chainer.serializers.save_hdf5( 'dcgan-dis-'+str(n_save)+'.hdf5', model_dis )
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'dcgan-gen.hdf5', model_gen )
chainer.serializers.save_hdf5( 'dcgan-dis.hdf5', model_dis )

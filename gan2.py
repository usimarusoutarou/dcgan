# -*- coding: utf-8 -*-
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
import codecs

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
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Generator_NN, self).__init__()
		with self.init_scope():
			self.c0=L.Convolution2D(3, 8, 3, 1, 1, initialW=w)
			self.c1=L.Convolution2D(8, 16, 4, 2, 1, initialW=w)
			self.c2=L.Convolution2D(16, 16, 3, 1, 1, initialW=w)
			self.c3=L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
			self.c4=L.Convolution2D(32, 32, 3, 1, 1, initialW=w)
			self.c5=L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
			self.c6=L.Convolution2D(64, 64, 3, 1, 1, initialW=w)
			self.c7=L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
			self.c8=L.Convolution2D(128, 128, 3, 1, 1, initialW=w)

			self.dc8=L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
			self.dc7=L.Convolution2D(128, 64, 3, 1, 1, initialW=w)
			self.dc6=L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w)
			self.dc5=L.Convolution2D(64, 32, 3, 1, 1, initialW=w)
			self.dc4=L.Deconvolution2D(64, 32, 4, 2, 1, initialW=w)
			self.dc3=L.Convolution2D(32, 16, 3, 1, 1, initialW=w)
			self.dc2=L.Deconvolution2D(32, 16, 4, 2, 1, initialW=w)
			self.dc1=L.Convolution2D(16, 8, 3, 1, 1, initialW=w)
			self.dc0=L.Convolution2D(16, 3, 3, 1, 1, initialW=w)

			self.bnc0=L.BatchNormalization(8)
			self.bnc1=L.BatchNormalization(16)
			self.bnc2=L.BatchNormalization(16)
			self.bnc3=L.BatchNormalization(32)
			self.bnc4=L.BatchNormalization(32)
			self.bnc5=L.BatchNormalization(64)
			self.bnc6=L.BatchNormalization(64)
			self.bnc7=L.BatchNormalization(128)
			self.bnc8=L.BatchNormalization(128)

			self.bnd8=L.BatchNormalization(128)
			self.bnd7=L.BatchNormalization(64)
			self.bnd6=L.BatchNormalization(64)
			self.bnd5=L.BatchNormalization(32)
			self.bnd4=L.BatchNormalization(32)
			self.bnd3=L.BatchNormalization(16)
			self.bnd2=L.BatchNormalization(16)
			self.bnd1=L.BatchNormalization(8)

	def __call__(self, x):
		e0 = F.leaky_relu(self.bnc0(self.c0(x)))
		e1 = F.leaky_relu(self.bnc1(self.c1(e0)))
		e2 = F.leaky_relu(self.bnc2(self.c2(e1)))
		del e1
		e3 = F.leaky_relu(self.bnc3(self.c3(e2)))
		e4 = F.leaky_relu(self.bnc4(self.c4(e3)))
		del e3
		e5 = F.leaky_relu(self.bnc5(self.c5(e4)))
		e6 = F.leaky_relu(self.bnc6(self.c6(e5)))
		del e5
		e7 = F.leaky_relu(self.bnc7(self.c7(e6)))
		e8 = F.leaky_relu(self.bnc8(self.c8(e7)))

		d8 = F.leaky_relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
		del e7, e8
		d7 = F.leaky_relu(self.bnd7(self.dc7(d8)))
		del d8
		d6 = F.leaky_relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
		del d7, e6
		d5 = F.leaky_relu(self.bnd5(self.dc5(d6)))
		del d6
		d4 = F.leaky_relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
		del d5, e4
		d3 = F.leaky_relu(self.bnd3(self.dc3(d4)))
		del d4
		d2 = F.leaky_relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
		del d3, e2
		d1 = F.leaky_relu(self.bnd1(self.dc1(d2)))
		del d2
		d0 = F.sigmoid(self.dc0(F.concat([e0, d1])))
		
		return d0	# 結果を返すのみ

# ニューラルネットワークを作成
model = DCGAN_Generator_NN()

if uses_device >= 0:
	# GPUを使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	# GPU用データ形式に変換
	model.to_gpu()

# 学習結果を読み込む
chainer.serializers.load_hdf5( 'gan-gen-20.hdf5', model )

# 画像を生成する
num_generate = 1	# 生成する画像の数

images = []

# 画像を読み込んで128×128ピクセルにリサイズ
img = Image.open('/home/nagalab/soutarou/dcgan/images/' + 'image14-525.png').convert('RGB').resize((128, 128))
# 画素データを0〜1の領域にする
hpix = np.array(img, dtype=np.float32) / 255.0
hpix = hpix.transpose(2,0,1)
# 配列に追加
images.append(hpix)
	
images = cp.array(images, dtype=cp.float32)

result = model(images)

data = np.zeros((128, 128, 3), dtype=np.uint8)
dst = result.data[0] * 255.0
if uses_device >= 0:
	dst = chainer.cuda.to_cpu(dst)
data[:,:,0] = dst[0]
data[:,:,1] = dst[1]
data[:,:,2] = dst[2]
himg = Image.fromarray(data, 'RGB')
himg.save('gen-'+str(0)+'.png')
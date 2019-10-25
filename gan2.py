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
		# 全ての層を定義する
		with self.init_scope():
			self.c0 = L.Convolution2D(3, 8, 4, 2, 1, initialW=w)
			self.c1 = L.Convolution2D(8, 16, 3, 1, 1, initialW=w)
			self.c2 = L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
			self.c3 = L.Convolution2D(32, 64, 3, 1, 1, initialW=w)
			self.dc0 = L.Deconvolution2D(64, 32, 4, 2, 1, initialW=w)
			self.dc1 = L.Deconvolution2D(32, 16, 3, 1, 1, initialW=w)
			self.dc2 = L.Deconvolution2D(16, 8, 4, 2, 1, initialW=w)
			self.dc3 = L.Deconvolution2D(8, 3, 3, 1, 1, initialW=w)
			self.bn0 = L.BatchNormalization(8)
			self.bn1 = L.BatchNormalization(16)
			self.bn2 = L.BatchNormalization(32)
			self.bn3 = L.BatchNormalization(64)
			self.bn4 = L.BatchNormalization(32)
			self.bn5 = L.BatchNormalization(16)
			self.bn6 = L.BatchNormalization(8)


	def __call__(self, z):
		h = F.relu(self.bn0(self.c0(z)))
		h = F.relu(self.bn1(self.c1(h)))
		h = F.relu(self.bn2(self.c2(h)))
		h = F.relu(self.bn3(self.c3(h)))
		h = F.relu(self.bn4(self.dc0(h)))
		h = F.relu(self.bn5(self.dc1(h)))
		h = F.relu(self.bn6(self.dc2(h)))
		x = F.sigmoid(self.dc3(h))
		return x	# 結果を返すのみ
# ニューラルネットワークを作成
model = DCGAN_Generator_NN()

if uses_device >= 0:
	# GPUを使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	# GPU用データ形式に変換
	model.to_gpu()

# 学習結果を読み込む
chainer.serializers.load_hdf5( 'gan-gen-10.hdf5', model )

# 画像を生成する
num_generate = 1	# 生成する画像の数

images = []

# 画像を読み込んで128×128ピクセルにリサイズ
img = Image.open('/home/nagalab/soutarou/dcgan/images/' + 'image2-101.png').convert('RGB').resize((128, 128))
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
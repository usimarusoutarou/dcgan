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
			self.l0 = L.Linear(100, neuron_size * image_size * image_size // 8 // 8,
							   initialW=w)
			self.dc1 = L.Deconvolution2D(neuron_size, neuron_size // 2, 4, 2, 1, initialW=w)
			self.dc2 = L.Deconvolution2D(neuron_size // 2, neuron_size // 4, 4, 2, 1, initialW=w)
			self.dc3 = L.Deconvolution2D(neuron_size // 4, neuron_size // 8, 4, 2, 1, initialW=w)
			self.dc4 = L.Deconvolution2D(neuron_size // 8, 3, 3, 1, 1, initialW=w)
			self.bn0 = L.BatchNormalization(neuron_size * image_size * image_size // 8 // 8)
			self.bn1 = L.BatchNormalization(neuron_size // 2)
			self.bn2 = L.BatchNormalization(neuron_size // 4)
			self.bn3 = L.BatchNormalization(neuron_size // 8)

	def __call__(self, z):
		shape = (len(z), neuron_size, image_size // 8, image_size // 8)
		h = F.reshape(F.relu(self.bn0(self.l0(z))), shape)
		h = F.relu(self.bn1(self.dc1(h)))
		h = F.relu(self.bn2(self.dc2(h)))
		h = F.relu(self.bn3(self.dc3(h)))
		x = F.sigmoid(self.dc4(h))
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
chainer.serializers.load_hdf5( 'dcgan-gen-28.hdf5', model )

# 画像を生成する
num_generate = 100	# 生成する画像の数
# 元となるベクトルを作成
rnd = random.uniform(-1, 1, (num_generate, 100, 1, 1))
rnd = cp.array(rnd, dtype=cp.float32)

# バッチ処理を使って一度に生成する
with chainer.using_config('train', False):
	result = model(rnd)

# 生成した画像と元となったベクトルを保存する
f = codecs.open('vectors.txt', 'w', 'utf8')
for i in range(num_generate):
	# 画像を保存する
	data = np.zeros((128, 128, 3), dtype=np.uint8)
	dst = result.data[i] * 255.0
	if uses_device >= 0:
		dst = chainer.cuda.to_cpu(dst)
	data[:,:,0] = dst[0]
	data[:,:,1] = dst[1]
	data[:,:,2] = dst[2]
	himg = Image.fromarray(data, 'RGB')
	himg.save('gen-'+str(i)+'.png')
	# 画像の元となったベクトルを保存する
	f.write(','.join([str(j) for j in rnd[i][:,0][:,0]]))
	f.write('\n')
f.close()

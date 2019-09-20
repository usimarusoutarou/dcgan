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
		super(DCGAN_Generator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.c0=L.Convolution2D(3, 32, 3, 1, 1)
			self.c1=L.Convolution2D(32, 64, 4, 2, 1)
			self.c2=L.Convolution2D(64, 64, 3, 1, 1)
			self.c3=L.Convolution2D(64, 128, 4, 2, 1)
			self.c4=L.Convolution2D(128, 128, 3, 1, 1)
			self.c5=L.Convolution2D(128, 256, 4, 2, 1)
			self.c6=L.Convolution2D(256, 256, 3, 1, 1)
			self.c7=L.Convolution2D(256, 512, 4, 2, 1)
			self.c8=L.Convolution2D(512, 512, 3, 1, 1)

			self.dc8=L.Deconvolution2D(1024, 512, 4, 2, 1)
			self.dc7=L.Convolution2D(512, 256, 3, 1, 1)
			self.dc6=L.Deconvolution2D(512, 256, 4, 2, 1)
			self.dc5=L.Convolution2D(256, 128, 3, 1, 1)
			self.dc4=L.Deconvolution2D(256, 128, 4, 2, 1)
			self.dc3=L.Convolution2D(128, 64, 3, 1, 1)
			self.dc2=L.Deconvolution2D(128, 64, 4, 2, 1)
			self.dc1=L.Convolution2D(64, 32, 3, 1, 1)
			self.dc0=L.Convolution2D(64, 3, 3, 1, 1)

			self.bnc0=L.BatchNormalization(32)
			self.bnc1=L.BatchNormalization(64)
			self.bnc2=L.BatchNormalization(64)
			self.bnc3=L.BatchNormalization(128)
			self.bnc4=L.BatchNormalization(128)
			self.bnc5=L.BatchNormalization(256)
			self.bnc6=L.BatchNormalization(256)
			self.bnc7=L.BatchNormalization(512)
			self.bnc8=L.BatchNormalization(512)

			self.bnd8=L.BatchNormalization(512)
			self.bnd7=L.BatchNormalization(256)
			self.bnd6=L.BatchNormalization(256)
			self.bnd5=L.BatchNormalization(128)
			self.bnd4=L.BatchNormalization(128)
			self.bnd3=L.BatchNormalization(64)
			self.bnd2=L.BatchNormalization(64)
			self.bnd1=L.BatchNormalization(32)

	def __call__(self, x):
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
chainer.serializers.load_hdf5( 'dcgan-gen-28.hdf5', model )

# 画像を生成する
num_generate = 1	# 生成する画像の数
# 元となるベクトルを作成
rnd = random.uniform(-1, 1, (num_generate, 100, 1, 1))
rnd = cp.array(rnd, dtype=cp.float32)

images = []
"""
fs = os.listdir('/home/nagalab/soutarou/images')
for fn in fs:
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('/home/nagalab/soutarou/images/' + fn).convert('RGB').resize((128, 128))
	# 画素データを0〜1の領域にする
	hpix = np.array(img, dtype=np.float32) / 255.0
	hpix = hpix.transpose(2,0,1)
	# 配列に追加
	images.append(hpix)
	
images = cp.array(images, dtype=cp.float32)
"""
# 画像を読み込んで128×128ピクセルにリサイズ
img = Image.open('/home/nagalab/soutarou/images/' + '68068733_p0_master1200.jpg').convert('RGB').resize((128, 128))
# 画素データを0〜1の領域にする
hpix = np.array(img, dtype=np.float32) / 255.0
hpix = hpix.transpose(2,0,1)
# 配列に追加
images.append(hpix)
	
images = cp.array(images, dtype=cp.float32)

# バッチ処理を使って一度に生成する
with chainer.using_config('train', False):
	result = model(images)

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

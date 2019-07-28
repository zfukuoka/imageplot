#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 必要なライブラリのimport
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import csv
import datetime

def rgb_normalization():
  # 仮実装

  # テストデータで0～255のRGB256諧調を前提とする
  test_np = np.arange(27).reshape(3,3,3)

  # 0～1の間で正規化
  test_np2 = test_np/255
  print(test_np.shape)
  print(test_np)
  print(test_np2)

  # ガンマ補正を元に戻し、リニア化
  LINEAR_THR = 0.04045
  print(np.piecewise(test_np2, [test_np2 <= LINEAR_THR, test_np2 > LINEAR_THR], [0, 1]))
  print(np.piecewise(test_np2, [test_np2 <= LINEAR_THR, test_np2 > LINEAR_THR], [lambda test_np2: test_np2/12.92, lambda test_np2: ((test_np2+0.055)/1.055)**2.4]))


def viewer():
  # 画像読み込み：仮実装のため、固定ファイル読み込み
  print('speed(opjp): ', datetime.datetime.now())
  im = Image.open("sample.jpg")
  im_list = np.asarray(im)

  # 画像の間引き：とりあえず、XGA以上の解像度を想定し、横幅1024未満にする間引き
  MAX_IMG_WIDTH = 1024
  (img_x, img_y, img_dim) = im_list.shape
  if img_x  > img_y:
    skip = int(img_x / MAX_IMG_WIDTH) + 1
  else:
    skip = int(img_y / MAX_IMG_WIDTH) + 1
  im_list2 = im_list[::skip,::skip,::]

  # 画像とグラフの同時表示
  fig = plt.figure(figsize=(16,8))
  fig.subplots_adjust(wspace=10)

  ax_img  = fig.add_subplot(111)
  ax_plot = fig.add_subplot(122)
  ax_img.imshow(im_list2)

  plt.show()

viewer()
#rgb_normalization()

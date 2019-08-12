#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 必要なライブラリのimport
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import csv
import datetime

def normalizeRgb(originPixel):
  # JPEGを前提として、RGBの各々の解像度8bit(255)と定義
  RGB_RESOLUTION = 255

  # 0～1の間で正規化
  tempPixel = originPixel / RGB_RESOLUTION

  # ガンマ補正を元に戻し、リニア化
  #   正規化値  <= 0.04045 : 正規化値 / 12.92
  #   正規化値  >  0.04045 : ((正規化値 + 0.055) / 1.055) に 2.4階乗
  LINEAR_THR = 0.04045
  return np.piecewise(tempPixel, [tempPixel <= LINEAR_THR, tempPixel > LINEAR_THR], [lambda tempPixel: tempPixel/12.92, lambda tempPixel: ((tempPixel+0.055)/1.055)**2.4])

# convert from normalized RGB to CIE XYZ
def convertToCieXYZ(normalizedRgb):
  # RGB of sRGB color space to CIE XYZ convert matrix
  MATRIX = np.array([[0.412424, 0.357579, 0.180464],[0.212656, 0.715158, 0.072186],[0.019332, 0.119193, 0.950444]], dtype='float32')

  # 画像を１次元に並べ、ループで画素毎にRGBからXYZを算出
  ret_arr = np.empty((0,3), dtype='float32')
  temp = normalizedRgb.reshape(-1, 3)
  (number, _) = temp.shape

  for i in range(number):
    ret_arr = np.append(ret_arr, np.dot(MATRIX, temp[i]).reshape(1,3), axis=0)

  return ret_arr

# convert from CIE XYZ to CIE xyz 
def convertToCiexyz(cieXYZ):
  # ループで画素毎にXYZからxyzを算出
  #   x = X / (X + Y + Z)
  #   y = Y / (X + Y + Z)
  #   z = Z / (X + Y + Z)
  ret_arr = np.empty((0,3), dtype='float32')
  (number, _) = cieXYZ.shape

  for i in range(number):
    xyz = np.array([cieXYZ[i][0]/np.sum(cieXYZ[i]), cieXYZ[i][1]/np.sum(cieXYZ[i]), cieXYZ[i][2]/np.sum(cieXYZ[i])])
    ret_arr = np.append(ret_arr, xyz.reshape(1,3), axis=0)

  return ret_arr

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

#viewer()

# テストデータで0～255のRGB256諧調を前提とする
#test_np = np.arange(27).reshape(3,3,3)
#test_np = np.linspace(228, 255, 27).reshape(3,3,3)
# sRGB Red Green Blue White point
test_np = np.array(
    [ [[255, 0, 0], [0, 255, 0]],
      [[0, 0, 255], [255, 255, 255]] ])

test_np2 = normalizeRgb(test_np)
test_np3 = convertToCieXYZ(test_np2)
test_np4 = convertToCiexyz(test_np3)

print("test_np2")
print(test_np2)
print("test_np3")
print(test_np3)
print("test_np4")
print(test_np4)

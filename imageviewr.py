#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 必要なライブラリのimport
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import datetime

DEFAULT_DTYPE = np.float32

def normalizeRgb(originPixel):
  # JPEGを前提として、RGBの各々の解像度8bit(255)と定義
  RGB_RESOLUTION = 255

  # 0～1の間で正規化
  tempPixel = (originPixel / RGB_RESOLUTION).astype(DEFAULT_DTYPE)

  # ガンマ補正を元に戻し、リニア化
  #   正規化値  <= 0.04045 : 正規化値 / 12.92
  #   正規化値  >  0.04045 : ((正規化値 + 0.055) / 1.055) に 2.4階乗
  LINEAR_THR = 0.04045
  ret_arr =  np.piecewise(
    tempPixel,
    [
      tempPixel <= LINEAR_THR,
      tempPixel > LINEAR_THR
    ],
    [
      lambda tempPixel: tempPixel/12.92,
      lambda tempPixel: ((tempPixel+0.055)/1.055)**2.4
    ])
  return ret_arr.astype(DEFAULT_DTYPE)


# convert from normalized RGB to CIE XYZ
def convertToCieXYZ(normalizedRgb):
  # RGB of sRGB color space to CIE XYZ convert matrix
  MATRIX = np.array([
    [0.412424, 0.357579, 0.180464],
    [0.212656, 0.715158, 0.072186],
    [0.019332, 0.119193, 0.950444]], dtype=DEFAULT_DTYPE)

  # 1次元行列の配列のまま2次元行列を掛けるため、
  # 掛け算の順序逆転と2次元配列を転置してRGBからXYZを算出
  ret_arr = np.dot(normalizedRgb, MATRIX.T)

  # 前の実装を維持するため、2次元配列を1次元配列
  return ret_arr.reshape(-1, 3)


# convert from normalized RGB to YCbCr
def convertToYCbCr(normalizedRgb):
  # RGB of sRGB color space to YCbCr(ITU-R BT.601) convert matrix
  MATRIX = np.array([
    [ 0.299000,  0.587000,  0.114000],
    [-0.168736, -0.331264,  0.500000],
    [ 0.500000, -0.418688, -0.081312]], dtype=DEFAULT_DTYPE)

  # 1次元行列の配列のまま2次元行列を掛けるため、
  # 掛け算の順序逆転と2次元配列を転置してRGBからYCbCrを算出
  ret_arr = np.dot(normalizedRgb, MATRIX.T)

  # 前の実装を維持するため、2次元配列を1次元配列
  return ret_arr.reshape(-1, 3)


# convert from CIE XYZ to CIE xyz 
def convertToCiexyz(cieXYZ):
  # 画素毎にXYZからxyzを算出
  #   x = X / (X + Y + Z)
  #   y = Y / (X + Y + Z)
  #   z = Z / (X + Y + Z)
  return cieXYZ / np.tile(np.sum(cieXYZ, axis=1), (3,1)).T


# convert from CIE XYZ to CIE L*a*b*
def convertToCielab(cieXYZ):
  # 画素ごとにXYZからL*a*b*を算出
  #   L* = 116 * f(Y/Yn) -16
  #   a* = 500 * (f(X/Xn) - f(Y/Yn))
  #   b* = 200 * (f(Y/Yn) - f(Z/Zn))
  #
  #   f(t) = t ^ (1/3) ... t > (6/29)^3
  #   f(t) = (1/3) * (29/6)^2 * t + (4/29) ... t <= (6/29)^3

  # 現時点で4段階の計算に分ける予定
  #   1. CIE XYZ => X/Xn, Y/Yn, Z/Zn
  #   2. X/Xn, Y/Yn, Z/Zn => f(X/Xn), f(Y/Yn), f(Z/Zn)
  #   3. f(X/Xn), f(Y/Yn), f(Z/Zn) => L* + 16, a*, b*
  #   4. L* + 16, a*, b* => L*, a*, b*

  # 基準点の白点と変換式の行列
  REFERENCEPOINT_D50 = np.array([
    [0.964212,  1.0,  0.825188],], dtype=DEFAULT_DTYPE)
  REFERENCEPOINT_D65 = np.array([
    [0.950489,  1.0,  1.088840],], dtype=DEFAULT_DTYPE)
  CONVERT_MATRIX = np.array([
    [0.0, 116.0, 0.0],
    [500.0, -500.0, 0.0],
    [0.0, 200.0, -200.0]], dtype=DEFAULT_DTYPE)
  L_OFFSET = np.array([
    [-16.0, 0.0, 0.0],], dtype=DEFAULT_DTYPE)
  
  # 基準のWhitePointの補正
  correctedWPXYZ = cieXYZ / REFERENCEPOINT_D50
  THR = 0.008856451679035631 # (6/29)^3
  preConverted =  np.piecewise(
    correctedWPXYZ,
    [
      correctedWPXYZ <= THR,
      correctedWPXYZ > THR
    ],
    [
      lambda correctedWPXYZ:
        7.787037037037035 * correctedWPXYZ + 0.13793103448275862,
      lambda correctedWPXYZ:
        correctedWPXYZ ** 0.3333333333333333
    ])
  return np.dot(preConverted, CONVERT_MATRIX.T) + L_OFFSET


def convertToCieLch(cieLab):
  # (c, h) = convertToCieLch(cieLab)
  c = np.sqrt(
    np.power(cieLab[0:,1], 2) + np.power(cieLab[0:,2], 2),
    dtype=DEFAULT_DTYPE)
  h = np.arctan2(cieLab[0:,2], cieLab[0:,1], dtype=DEFAULT_DTYPE)
  return (c, h)


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

  # 処理時間がかかるため、追加で固定値で間引き
  skip *= 4
  im_list2 = im_list[::skip,::skip,::]

  # RGB to CIE xyz
  normalizedRGB = normalizeRgb(im_list2)
  cieXYZ = convertToCieXYZ(normalizedRGB)
  ciexyz = convertToCiexyz(cieXYZ)
  ycbcr = convertToYCbCr(normalizedRGB)
  cieLab = convertToCielab(cieXYZ)
  (c, h) = convertToCieLch(cieLab)
  
  # pick up x and y for ploting
  x = ciexyz[0:, 0]
  y = ciexyz[0:, 1]

  # 画像とグラフの同時表示
  fig = plt.figure(figsize=(16,16))
  fig.subplots_adjust(wspace=0.2)

  # 画像表示
  ax_img = fig.add_subplot(321)
  ax_img.imshow(im_list2)
  ax_img.set_title("Image")

  # CIE xy のプロット
  # RGBWのCIE xy座標
  POLARS = np.array([
      [0.64, 0.33], [0.30, 0.60],
      [0.15, 0.06], [0.3127, 0.3290]
    ])
  ax_plot = fig.add_subplot(322)
  ax_plot.plot(
    POLARS[0:,0], POLARS[0:,1], "r+",
    label="R/G/B polar and white point in sRGB color space")
  ax_plot.plot(x, y, 'k.', alpha=0.3, label="color in image")

  # 描画の補助情報(描画範囲、メジャーとマイナーのメモリとグリッド線)
  ax_plot.set_xlim(0, 1.0)
  ax_plot.set_ylim(0, 1.0)
  ax_plot.xaxis.set_major_locator(MultipleLocator(0.1))
  ax_plot.yaxis.set_major_locator(MultipleLocator(0.1))
  ax_plot.grid(linestyle="--", zorder=-10)
  ax_plot.xaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot.yaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot.legend()
  ax_plot.set_aspect('equal')
  ax_plot.set_xlabel("x")
  ax_plot.set_ylabel("y")
  ax_plot.set_title("CIE xy")

  # RGBWCyanMagentaYellowのCbCr座標とプロット色
  POLARS_CBCR = [
    (-0.168736, 0.5, '#FF0000'), (-0.331264, -0.418688, '#00FF00'),
    (0.5, -0.081312, '#0000FF'), (0.0, 0.0, '#000000'),
    (0.168736, -0.5, '#00FFFF'), (0.331264, 0.418688, '#FF00FF'),
    (-0.5, 0.081312, '#C0C000')
  ]
  
  ax_plot2 = fig.add_subplot(323)
  for (cb, cr, plot_color) in POLARS_CBCR:
    ax_plot2.plot(
      cb, cr, marker="+", color=plot_color, alpha=1.0
    )
  ax_plot2.plot(
    ycbcr[0:, 1], ycbcr[0:, 2], 'k.', alpha=0.3,
    label="color in image")
  ax_plot2.set_xlim(-0.52, 0.52)
  ax_plot2.set_ylim(-0.52, 0.52)
  ax_plot2.xaxis.set_major_locator(MultipleLocator(0.1))
  ax_plot2.yaxis.set_major_locator(MultipleLocator(0.1))
  ax_plot2.grid(linestyle="--", zorder=-10)
  ax_plot2.axvline(c='gray', linewidth=1.0)
  ax_plot2.axhline(c='gray', linewidth=1.0)
  ax_plot2.xaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot2.yaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot2.legend()
  ax_plot2.set_aspect('equal')
  ax_plot2.set_xlabel("Cb")
  ax_plot2.set_ylabel("Cr")
  ax_plot2.set_title("CbCr")

  # RGBCyanMagentaYellowのCIE L*a*b*座標とプロット色
  POLARS_LAB = [
    (78.28357, 62.150043, '#FF0000'), (-87.905914, 73.916306, '#00FF00'),
    (77.819214, -126.371704, '#0000FF'), (-50.057648, -33.38832, '#00FFFF'),
    (96.19589, -79.46594, '#FF00FF'), (-23.782745, 84.73825, '#C0C000')]
  ax_plot3 = fig.add_subplot(324)
  for(a_ast, b_ast, plot_color) in POLARS_LAB:
    ax_plot3.plot(
      a_ast, b_ast, marker="+", color=plot_color, alpha=1.0
    )
  ax_plot3.plot(
    cieLab[0:, 1], cieLab[0:, 2], 'k.', alpha=0.3,
    label="color in image")
  ax_plot3.set_xlim(-128, 128)
  ax_plot3.set_ylim(-128, 128)
  ax_plot3.xaxis.set_major_locator(MultipleLocator(25.0))
  ax_plot3.yaxis.set_major_locator(MultipleLocator(25.0))
  ax_plot3.axvline(c='gray', linewidth=1.0)
  ax_plot3.axhline(c='gray', linewidth=1.0)
  for i in range(9):
    grid_circle = patches.Circle(
      xy=(0, 0), radius=(i*25.0), fill=False,
      edgecolor='gray', linestyle="--", zorder=-10)
    ax_plot3.add_patch(grid_circle)
  ax_plot3.xaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot3.yaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot3.legend()
  ax_plot3.set_aspect('equal')
  ax_plot3.set_xlabel("a*")
  ax_plot3.set_ylabel("b*")
  ax_plot3.set_title("CIE L*a*b*(D50)")

  # RGBCyanMagentaYellowのCIE L*C*H座標とプロット色
  POLARS_LCH = [
    (99.95472, 0.671016, '#FF0000'), (114.85239, 2.4424305, '#00FF00'),
    (148.41037, -1.0188428, '#0000FF'), (60.17099, -2.5533612, '#00FFFF'),
    (124.773735, -0.690445, '#FF00FF'), (88.01244, 1.8444182, '#C0C000')]
  ax_plot4 = fig.add_subplot(313)
  for(c_ast, h_ast, plot_color) in POLARS_LCH:
    ax_plot4.plot(
      h_ast, c_ast, marker="+", color=plot_color, alpha=1.0
    )
  ax_plot4.plot(
    h, c, 'k.', alpha=0.3,
    label="color in image")
  ax_plot4.set_xlim(-1.0 * np.pi, 1.0 * np.pi)
  ax_plot4.set_ylim(0, 175)
  ax_plot4.xaxis.set_major_locator(MultipleLocator(np.pi/6))
  ax_plot4.yaxis.set_major_locator(MultipleLocator(25.0))
  ax_plot4.grid(linestyle="--", zorder=-10)
  ax_plot4.xaxis.set_minor_locator(AutoMinorLocator(3))
  ax_plot4.yaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot4.legend()
  ax_plot4.set_xlabel("Hue(H) [radian]")
  ax_plot4.set_ylabel("Chroma(C*)")
  ax_plot4.set_title("CIE L*C*H(D50)")


  print('speed(edjp): ', datetime.datetime.now())

  plt.show()
  # plt.savefig("sample.png",format = 'png', dpi=120)


viewer()

# テストデータで0～255のRGB256諧調を前提とする
#test_np = np.arange(27).reshape(3,3,3)
#test_np = np.linspace(228, 255, 27).reshape(3,3,3)
# sRGB Red Green Blue White point
# test_np = np.array([
#     [[255, 0, 0], [0, 255, 0]],
#     [[0, 0, 255], [255, 255, 255]] ])
# test_np = np.array([
#   [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#   [[0,255, 255], [255, 0, 255], [255, 255, 0]]
# ])
# test_np2 = normalizeRgb(test_np)
# test_np3 = convertToCieXYZ(test_np2)
# test_np4 = convertToCiexyz(test_np3)
# test_np5 = convertToYCbCr(test_np2)
# test_np6 = convertToCielab(test_np3)
# (test_np7c, test_np7h) = convertToCieLch(test_np6)
#
# print("test_np2")
# print(test_np2)
# print("test_np3")
# print(test_np3)
# print("test_np4")
# print(test_np4)
# print("test_np5")
# print(test_np5)
# print("test_np6")
# print(test_np6)
# print("test_np7")
# print(test_np7c)
# print(test_np7h)

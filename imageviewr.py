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

  # RGB to CIE xyz, YCbCr, CIE L*a*b* and CIE L*C*h
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
  
  # YCbCrのプロット
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

  # sRGBの0,0,0からRGBCyanMagentaYellowへ徐々に同じ割合で増加した座標
  RED_LAB = [
    [ 0.0, 4.339012, 12.0952835, 23.040688, 29.08194, 33.762222,
      38.285065, 42.67778, 46.960007, 51.14662, 55.24913, 59.27684,
      63.237152, 67.13638, 70.97974, 74.77173],
    [ 0.0, 1.5270233, 4.256687, 8.710579, 14.891321, 20.972176,
      26.528345, 31.582932, 36.149002, 40.233482, 43.839302, 47.06043,
      50.204575, 53.300194, 56.351486, 59.361996]]
  GREEN_LAB = [
    [ 0.0, -6.9461594, -19.071365, -27.179245, -32.656616, -37.91217,
      -42.990967, -47.923584, -52.732193, -57.43344, -62.040222, -66.56296,
      -71.01007, -75.38858, -79.70438, -83.962494],
    [ 0.0, 4.6055145, 12.721571, 21.063679, 27.323513, 31.878716,
      36.149246, 40.29689, 44.340237, 48.29329, 52.16696, 55.969925,
      59.709312, 63.391, 67.019966, 70.600426]]
  BLUE_LAB = [
    [ 0.0, 2.3195648, 6.4659576, 13.231476, 22.874817, 31.837372,
      38.03235, 42.424637, 46.681473, 50.84326, 54.92148, 58.925247,
      62.86209, 66.73819, 70.55878, 74.32826],
    [ 0.0, -8.712122, -21.846209, -33.910652, -44.532593, -53.811886,
      -61.792637, -68.89394, -75.80669, -82.56508, -89.18772, -95.6895,
      -102.08258, -108.37702, -114.581314, -120.70267]]
  CYAN_LAB = [
    [ 0.0, -4.6265945, -12.092834, -15.477104, -18.59616, -21.588928,
      -24.481018, -27.289886, -30.028107, -32.70523, -35.328552, -37.90399,
      -40.4364, -42.929688, -45.3873, -47.812042],
    [ 0.0, -4.1066093, -8.131218, -10.323193, -12.403595, -14.399757,
      -16.328781, -18.202278, -20.028687, -21.814316, -23.564056, -25.28189,
      -26.970963, -28.634018, -30.273224, -31.890533]]
  MAGENTA_LAB = [
    [ 0.0, 6.658577, 18.56115, 29.71354, 35.736282, 41.487457,
      47.04521, 52.443024, 57.705124, 62.849686, 67.890915, 72.840164,
      77.706696, 82.49808, 87.220856, 91.880554],
    [ 0.0, -7.1850986, -17.406364, -24.558186, -29.52119, -34.272163,
      -38.86332, -43.32238, -47.669296, -51.91915, -56.08364, -60.17215,
      -64.19229, -68.1504, -72.05182, -75.90109]]
  YELLOW_LAB = [
    [ 0.0, -2.6071472, -5.7919083, -7.3532715, -8.835159, -10.257019,
      -11.631088, -12.965607, -14.266541, -15.538422, -16.78482, -18.008392,
      -19.211578, -20.396149, -21.563812, -22.71579],
    [ 0.0, 6.132538, 16.143116, 24.99825, 31.476112, 36.546013,
      41.441784, 46.196682, 50.832, 55.363815, 59.80462, 64.16437,
      68.45123, 72.67195, 76.83223, 80.9369]]
    
  # RGBの原色から補色間のプロット
  RED_TO_YELLOW_LAB = [
    [ 75.09357, 67.53348, 55.71701, 40.90918, 24.55774, 7.8143005,
      -8.587097, -23.782745],
    [ 62.401733, 63.163517, 64.75249, 67.31845, 70.79765, 75.01154,
      79.76728, 84.73825]]
  RED_TO_MAGENTA_LAB = [
    [ 78.58859, 79.35852, 80.705414, 82.67145, 85.2583, 88.43564,
      92.14966, 96.19589],
    [ 51.136993, 33.007538, 12.779709, -7.368164, -26.811333, -45.39196,
      -63.100983, -79.46594]]
  GREEN_TO_CYAN_LAB = [
    [ -87.25238, -85.60391, -82.7247, -78.532135, -73.0347, -66.313965,
      -58.504425, -50.057648],
    [ 70.11995, 61.52334, 48.841476, 33.63788, 17.146057, 0.1320343,
      -16.968353, -33.38832]]
  GREEN_TO_YELLOW_LAB = [
    [ -83.2211, -69.64282, -48.51953, -23.782745],
    [ 74.53055, 76.45332, 79.90204, 84.73825]]
  BLUE_TO_MAGENTA_LAB = [
    [ 78.58574, 81.60315, 87.70828, 96.19589],
    [ -122.44697, -111.830696, -96.58079, -79.46594]]
  BLUE_TO_CYAN_LAB = [
    [ 71.03102, 56.646378, 37.518463, 17.238464, -2.1560059, -19.919556,
      -35.97162, -50.057648],
    [ -122.70176, -114.60118, -103.03966, -89.59697, -75.37581, -61.001846,
      -46.80681, -33.38832]]

  ax_plot3 = fig.add_subplot(324)

  # RGBCyanMagentaYellow の極値のプロット
  for(a_ast, b_ast, plot_color) in POLARS_LAB:
    ax_plot3.plot(
      a_ast, b_ast, marker="+", color=plot_color, alpha=1.0
    )
  
  # 補助線のプロット
  ax_plot3.plot(
    RED_LAB[0], RED_LAB[1], color='#FF0000',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    GREEN_LAB[0], GREEN_LAB[1], color='#00FF00',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    BLUE_LAB[0], BLUE_LAB[1], color='#0000FF',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    CYAN_LAB[0], CYAN_LAB[1], color='#00FFFF',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    MAGENTA_LAB[0], MAGENTA_LAB[1], color='#FF00FF',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    YELLOW_LAB[0], YELLOW_LAB[1], color='#C0C000',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    RED_TO_YELLOW_LAB[0], RED_TO_YELLOW_LAB[1], color='#FFC000',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    RED_TO_MAGENTA_LAB[0], RED_TO_MAGENTA_LAB[1], color='#FF00C0',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    GREEN_TO_CYAN_LAB[0], GREEN_TO_CYAN_LAB[1], color='#00FFC0',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    GREEN_TO_YELLOW_LAB[0], GREEN_TO_YELLOW_LAB[1], color='#C0FF00',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    BLUE_TO_MAGENTA_LAB[0], BLUE_TO_MAGENTA_LAB[1], color='#C000FF',
    linestyle="--", alpha=1.0, zorder=-5.0)
  ax_plot3.plot(
    BLUE_TO_CYAN_LAB[0], BLUE_TO_CYAN_LAB[1], color='#00C0FF',
    linestyle="--", alpha=1.0, zorder=-5.0)

  # CIE L*a*b* のプロット
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

  # RGBCyanMagentaYellowのCIE L*C*h座標とプロット色
  POLARS_LCH = [
    (99.95472, 0.671016, '#FF0000'), (114.85239, 2.4424305, '#00FF00'),
    (148.41037, -1.0188428, '#0000FF'), (60.17099, -2.5533612, '#00FFFF'),
    (124.773735, -0.690445, '#FF00FF'), (88.01244, 1.8444182, '#C0C000')]

  # 角度を横軸、彩度を縦軸としたCIE L*C*hのプロット
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
  ax_plot4.set_xlabel("Hue(h) [radian]")
  ax_plot4.set_ylabel("Chroma(C*)")
  ax_plot4.set_title("CIE L*C*h(D50)")


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
# test_np = np.r_[np.array([0,0,0])]
# for i in range(15):
#   test_np = np.r_[test_np, np.array([16*(i+1),0,16*(i+1)])]
# test_np.resize(4,4,3)
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

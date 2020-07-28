#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 必要なライブラリのimport
from PIL import Image
import datetime
import json
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import sys

DEFAULT_DTYPE = np.float32

def normalizeRgb(originPixel):
  """Normalize RGB data and revert gamma correction.

  Args:
      originPixel (numpy.ndarray): Image data which has 3 dimensional data(height/width/RGB). The RGB data takes value from 0 to 255 and corrects gamma curb 2.2.

  Returns:
      numpy.ndarray: Normalized and reverted gamma correction image data(height/width/RGB).
  """  
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
  """Convert from normalized RGB image to CIE XYZ data using matrix for sRGB.

  Args:
      normalizedRgb (numpy.ndarray): Normalized and reverted gamma correction image data(height/width/RGB).

  Returns:
      numpy.ndarray: CIE XYZ data has 2 dimensional data(serialized array/XYZ).
  """  
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
  return np.c_[cieLab[0:,0], c, h]


def plotPolarPoint(axe, colorspace, target1, target2, color=None, marker="+"):
  """Plots polar points which is read from auxiliary_line.json.

  Args:
      axe (matplotlib.axes.Axes): Target axe for plotting.
      colorspace (str): Group key defined in json.
      target1 (str): Array data key defined under "colorspace". Plots this key-value on x-axis.
      target2 (str): Array data key defined under "colorspace". Plots this key-value on y-axis.
      color (type, optional): Color parameter defined by matplotlib. When specified "None", set color defined by json or "#000000". Defaults to None.
      marker (str, optional): Marker style parameter defined by matplotlib. Defaults to "+".
  """
  POLAR_KEY = 'polar'
  PLOTCOLOR_KEY = 'plot_color'
  with open('auxiliary_line.json') as auxline_file:
    polor_point = json.load(auxline_file)

    if not colorspace in polor_point:
      return
    if not POLAR_KEY in polor_point[colorspace]:
      return
    for key, value in polor_point[colorspace][POLAR_KEY].items():
      if target1 in value and target2 in value:
        if color is None:
          if PLOTCOLOR_KEY in value:
            plot_color = value[PLOTCOLOR_KEY]
          else:
            plot_color = '#000000'
        else:
          plot_color = color
        axe.plot(
          value[target1], value[target2],
          color=plot_color, marker=marker, linestyle="",
          alpha=1.0, zorder=-5.0
        )

        
def plotAuxiliaryLine(axe, colorspace, target1, target2, color=None, linestyle="--"):
  """Plots auxiliary line which is read from auxiliary_line.json.

  Args:
      axe (matplotlib.axes.Axes): Target axe for plotting.
      colorspace (str): Group key defined in json.
      target1 (str): Array data key defined under "colorspace". Plots this key-value on x-axis.
      target2 (str): Array data key defined under "colorspace". Plots this key-value on y-axis.
      color (type, optional): Color parameter defined by matplotlib. When specified "None", set color defined by json or "#000000". Defaults to None.
      linestyle (str, optional): Line style parameter defined by matplotlib. Defaults to "--".
  """
  AUX_KEY = 'auxiliary_line'
  PLOTCOLOR_KEY = 'plot_color'
  with open('auxiliary_line.json') as auxline_file:
    auxline = json.load(auxline_file)

    if not colorspace in auxline:
      return
    if not AUX_KEY in auxline[colorspace]:
      return
    for key, value in auxline[colorspace][AUX_KEY].items():
      if target1 in value and target2 in value:
        if color is None:
          if PLOTCOLOR_KEY in value:
            plot_color = value[PLOTCOLOR_KEY]
          else:
            plot_color = '#000000'
        else:
          plot_color = color
        axe.plot(
          value[target1], value[target2],
          color=plot_color, linestyle=linestyle, alpha=1.0, zorder=-5.0
        )


def debug_print():
  """Print for debug or output of json 
  """  
  # テストデータで0～255のRGB256諧調を前提とする
  #test_np = np.arange(27).reshape(3,3,3)
  #test_np = np.linspace(228, 255, 27).reshape(3,3,3)

  # sRGB Red Green Blue White point
  test_np = np.array([
      [[255, 0, 0], [0, 255, 0]],
      [[0, 0, 255], [255, 255, 255]] ])

  # sRGB Red/Blue/Green/Cyan/Magenta/Yellow point
  # test_np = np.array([
  #   [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
  #   [[0,255, 255], [255, 0, 255], [255, 255, 0]]
  # ])

  # sRGB Black to Yellow
  # test_np = np.r_[np.array([0,0,0])]
  # for i in range(15):
  #   test_np = np.r_[test_np, np.array([16*(i+1),0,16*(i+1)])]
  # test_np.resize(4,4,3)

  test_np2 = normalizeRgb(test_np)
  test_np3 = convertToCieXYZ(test_np2)
  test_np4 = convertToCiexyz(test_np3)
  test_np5 = convertToYCbCr(test_np2)
  test_np6 = convertToCielab(test_np3)
  test_np7 = convertToCieLch(test_np6)

  print("test_np2")
  print(test_np2)
  print("test_np3")
  print(test_np3)
  print("test_np4")
  print(test_np4)
  print("test_np5")
  print(test_np5)
  print("test_np6")
  print(test_np6)
  print("test_np7")
  print(test_np7)

  # print("For json")
  # print("\"rgb_r\":", test_np[:,:,0].flatten().tolist(), ), 
  # print("\"rgb_g\":", test_np[:,:,1].flatten().tolist())
  # print("\"rgb_b\":", test_np[:,:,2].flatten().tolist())
  # print("\"CIEXYZ_X\":", test_np3[:,0].tolist())
  # print("\"CIEXYZ_Y\":", test_np3[:,1].tolist())
  # print("\"CIEXYZ_Z\":", test_np3[:,2].tolist())
  # print("\"CIExyz_x\":", test_np4[:,0].tolist())
  # print("\"CIExyz_y\":", test_np4[:,1].tolist())
  # print("\"CIExyz_z\":", test_np4[:,2].tolist())
  # print("\"YCbCr_Y\":",  test_np5[:,0].tolist())
  # print("\"YCbCr_Cb\":", test_np5[:,1].tolist())
  # print("\"YCbCr_Cr\":", test_np5[:,2].tolist())
  # print("\"CIELab_D50_L\":", test_np6[:,0].tolist())
  # print("\"CIELab_D50_a\":", test_np6[:,1].tolist())
  # print("\"CIELab_D50_b\":", test_np6[:,2].tolist())
  # print("\"CIELCh_D50_L\":", test_np7[:,0].tolist())
  # print("\"CIELCh_D50_C\":", test_np7[:,1].tolist())
  # print("\"CIELCh_D50_h\":", test_np7[:,2].tolist())


def viewer(arg):
  print('speed(opjp): ', datetime.datetime.now())
  if(len(arg) > 1):
    # 画像読み込み：引数が指定されていたら先頭のファイルを読み込み
    im = Image.open(arg[1])
  else:
    # 画像読み込み：仮実装のため、固定ファイル読み込み
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
  cieLch = convertToCieLch(cieLab)
  
  # pick up x and y for ploting
  x = ciexyz[0:, 0]
  y = ciexyz[0:, 1]

  # 画像とグラフの同時表示
  fig = plt.figure(figsize=(18,24))
  fig.subplots_adjust(wspace=0.2)

  # 画像表示
  ax_img = fig.add_subplot(521)
  ax_img.imshow(im_list2)
  ax_img.set_title("Image")

  # CIE xy のプロット
  ax_plot = fig.add_subplot(522)

  # 極値(RGBCyanMagentaYellow)を CIE xy色度 のグラフにプロット
  plotPolarPoint(ax_plot, 'srgb', 'CIExyz_x', 'CIExyz_y')

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

  # YCbCrのプロット
  ax_plot2 = fig.add_subplot(523)

  # 極値(RGBCyanMagentaYellow)を Cb/Cr のグラフにプロット
  plotPolarPoint(ax_plot2, 'srgb', 'YCbCr_Cb', 'YCbCr_Cr')

  # 補助線のプロット
  plotAuxiliaryLine(ax_plot2, 'srgb', 'YCbCr_Cb', 'YCbCr_Cr')

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

  ax_plot3 = fig.add_subplot(524)

  # RGBCyanMagentaYellow の極値のプロット
  plotPolarPoint(ax_plot3, 'srgb', 'CIELab_D50_a', 'CIELab_D50_b')

  # 補助線のプロット
  plotAuxiliaryLine(ax_plot3, 'srgb', 'CIELab_D50_a', 'CIELab_D50_b')

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

  # 角度を横軸、彩度を縦軸としたCIE L*C*hのプロット
  ax_plot4 = fig.add_subplot(513)

  # 極値(RGBCyanMagentaYellow)をC* hのグラフにプロット
  plotPolarPoint(ax_plot4, 'srgb', 'CIELCh_D50_h', 'CIELCh_D50_C')
  
  # RGBの原色から補色間のプロット
  plotAuxiliaryLine(ax_plot4, 'srgb', 'CIELCh_D50_h', 'CIELCh_D50_C')

  # 画像のプロット
  ax_plot4.plot(
    cieLch[0:,2], cieLch[0:,1], 'k.', alpha=0.3,
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


  # 角度を横軸、輝度を縦軸としたCIE L*C*hのプロット
  ax_plot5 = fig.add_subplot(514)

  # 極値(RGBCyanMagentaYellow)をL* hのグラフにプロット
  plotPolarPoint(ax_plot5, 'srgb', 'CIELCh_D50_h', 'CIELCh_D50_L')

  ax_plot5.plot(
    cieLch[0:,2], cieLch[0:,0], 'k.', alpha=0.3,
    label="color in image")
  ax_plot5.set_xlim(-1.0 * np.pi, 1.0 * np.pi)
  ax_plot5.set_ylim(0,100)
  ax_plot5.xaxis.set_major_locator(MultipleLocator(np.pi/6))
  ax_plot5.yaxis.set_major_locator(MultipleLocator(10.0))
  ax_plot5.grid(linestyle="--", zorder=-10)
  ax_plot5.xaxis.set_minor_locator(AutoMinorLocator(3))
  ax_plot5.yaxis.set_minor_locator(AutoMinorLocator(5))
  ax_plot5.legend()
  ax_plot5.set_xlabel("Hue(h) [radian]")
  ax_plot5.set_ylabel("Luminance(L)")
  ax_plot5.set_title("CIE L*C*h(D50)")

  ax_plot6 = fig.add_subplot(529)
  ax_plot6.xaxis.set_major_locator(MultipleLocator(np.pi/3))
  ax_plot6.yaxis.set_major_locator(MultipleLocator(25.0))
  ax_plot7 = fig.add_subplot(5,2,10)
  ax_plot7.xaxis.set_major_locator(MultipleLocator(np.pi/3))
  ax_plot7.yaxis.set_major_locator(MultipleLocator(10.0))
  BINS_CH = [12, 7] # piの12等分(30度単位)と彩度25単位の7等分
  BINS_LH = [12, 10]  # piの12等分(30度単位)と輝度10単位の10等分
  num_ch = ax_plot6.hist2d(
    cieLch[0:,2], cieLch[0:,1], bins=BINS_CH,
    range=[[-np.pi, np.pi], [0, 175]], cmap=cm.jet
  )
  plt.colorbar(num_ch[3], ax=ax_plot6)
  num_lh = ax_plot7.hist2d(
    cieLch[0:,2], cieLch[0:,0], bins=BINS_LH,
    range=[[-np.pi, np.pi], [0, 100]], cmap=cm.jet
  )
  ax_plot6.set_xlabel("Hue(h) [radian]")
  ax_plot6.set_ylabel("Chroma(C*)")
  ax_plot6.set_title("Heatmap of CIE L*C*h(D50)")
  ax_plot7.set_xlabel("Hue(h) [radian]")
  ax_plot7.set_ylabel("Luminance(L)")
  ax_plot7.set_title("Heatmap of CIE L*C*h(D50)")
  plt.colorbar(num_lh[3], ax=ax_plot7)

  print('speed(edjp): ', datetime.datetime.now())

  plt.show()
  # plt.savefig("sample.png",format = 'png', dpi=120)


viewer(sys.argv)

# debug_print()


from PIL import Image
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

DEFAULT_DTYPE = np.float32

def normalizeLinearRgb(originPixel):
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


# convert from CIE XYZ to CIE xyz 
def convertToCiexyz(cieXYZ):
  # 画素毎にXYZからxyzを算出
  #   x = X / (X + Y + Z)
  #   y = Y / (X + Y + Z)
  #   z = Z / (X + Y + Z)
  return cieXYZ / np.tile(np.sum(cieXYZ, axis=1), (3,1)).T

def viewer():
    im = Image.open("test.jpg")
    im_list = np.asarray(im)

    # 固定で8つ分のスキップ
    skip = 8

    # 画像を1ドットずつずらしながら間引き画像を生成
    reducedImages = list()
    for i in range(skip):
        for j in range(skip):
            reducedImages.append(im_list[i::skip,j::skip,::])
    # print(len(reducedImages))

    # 間引き画像毎に、CIE xyzの値を算出
    ciexyzs = list()
    for i in range(skip*skip):
        # print(reducedImages[i].shape)
        ciexyzs.append(
            convertToCiexyz(
                convertToCieXYZ(
                    normalizeLinearRgb(
                        reducedImages[i]))))
    
    # 間引きした画像の全てのCIE xy色度図を 8x8 に並べてみる
    fig = plt.figure(figsize=(32,32))
    for i in range(skip):
        for j in range(skip):
            ax_plot = fig.add_subplot(
                skip, skip, i*skip+j+1,
                xlim=(0,0.7), ylim=(0, 0.7), aspect='equal')
            ax_plot.xaxis.set_major_locator(MultipleLocator(0.1))
            ax_plot.yaxis.set_major_locator(MultipleLocator(0.1))
            ax_plot.grid(linestyle="--", zorder=-10)
            ax_plot.plot(
                ciexyzs[i*skip+j][0:,0],
                ciexyzs[i*skip+j][0:,1],
                'k.', alpha=0.3)
    
    # plt.show()
    plt.savefig("test.png", format='png', dpi=120)
            
viewer()
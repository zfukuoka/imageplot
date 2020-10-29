import numpy as np

DEFAULT_DTYPE = np.float32

def convertToCieXYZForSrgb(normalizedRgb):
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


def convertToCieXYZForDcip3d65(normalizedRgb):
  """Convert from normalized RGB image to CIE XYZ data using matrix for DCI-P3 D65.

  Args:
      normalizedRgb (numpy.ndarray): Normalized and reverted gamma correction image data(height/width/RGB).

  Returns:
      numpy.ndarray: CIE XYZ data has 2 dimensional data(serialized array/XYZ).
  """  
  # RGB of sRGB color space to CIE XYZ convert matrix
  MATRIX = np.array([
    [0.48657, 0.26567, 0.19822],
    [0.22897, 0.69174, 0.07929],
    [0, 0.04511, 1.04394]], dtype=DEFAULT_DTYPE)

  # 1次元行列の配列のまま2次元行列を掛けるため、
  # 掛け算の順序逆転と2次元配列を転置してRGBからXYZを算出
  ret_arr = np.dot(normalizedRgb, MATRIX.T)

  # 前の実装を維持するため、2次元配列を1次元配列
  return ret_arr.reshape(-1, 3)


def convertToCiexyz(cieXYZ):
  """Convert from CIE XYZ data to CIE xyz data.

  Args:
      cieXYZ (numpy.ndarray): CIE XYZ data.

  Returns:
      numpy.ndarray: CIE xyz data has 2 dimensional data(serialized array/xyz).
  """
  # 画素毎にXYZからxyzを算出
  #   x = X / (X + Y + Z)
  #   y = Y / (X + Y + Z)
  #   z = Z / (X + Y + Z)
  return cieXYZ / np.tile(np.sum(cieXYZ, axis=1), (3,1)).T

sample = np.array([
    [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=DEFAULT_DTYPE)

print("sRGB:CIEXYZ,CIExyz")
print(convertToCieXYZForSrgb(sample))
print(convertToCiexyz(convertToCieXYZForSrgb(sample)))

print("DCI-P3 D65:CIEXYZ,CIExyz")
print(convertToCieXYZForDcip3d65(sample))
print(convertToCiexyz(convertToCieXYZForDcip3d65(sample)))

target_dcip3 = np.array([
    [0.3127, 0.3290, (1-0.3127-0.3290)],
    [0.680, 0.320, (1-0.680-0.320)],
    [0.265, 0.690, (1-0.265-0.690)],
    [0.150, 0.060, (1-0.150-0.060)]], dtype=DEFAULT_DTYPE)
print("DCI-P3 D65 CIExyz")
print(target_dcip3)
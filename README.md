# imageplot

## 概要 overview
ローカルに置いてある samplel.jpg を読み込み、読み込んだ画像の色をCIE xyz空間にプロットします。
This program plots color on CIE xy color space using sample.jpg which located on local file system.

## Requirements
- Python 3.5 or above
- numpy
- matplotlib

## 制限事項 restictions
- 全ピクセルを処理すると、時間がかかるため間引きしています
- 実装例を示すことと自身のプログラミングの習得を目的としているため、異常終了などの処理は全く行っていません
- 対象となる画像はjpegのみとしており、色空間がsRGB、ガンマ2.2を前提とした実装になっています
  - 近年のiPhoneは、jpegに異なる色空間(Display P3)を用いているので、正しく動作しません


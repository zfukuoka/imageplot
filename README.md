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


## 参考文献 reference

- [色彩工学入門-定量的な色の理解と活用](https://www.amazon.co.jp/%E8%89%B2%E5%BD%A9%E5%B7%A5%E5%AD%A6%E5%85%A5%E9%96%80-%E5%AE%9A%E9%87%8F%E7%9A%84%E3%81%AA%E8%89%B2%E3%81%AE%E7%90%86%E8%A7%A3%E3%81%A8%E6%B4%BB%E7%94%A8-%E7%AF%A0%E7%94%B0-%E5%8D%9A%E4%B9%8B/dp/4627846819/ref=sr_1_1?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&keywords=%E8%89%B2%E5%BD%A9%E5%B7%A5%E5%AD%A6&qid=1582021897&sr=8-1)

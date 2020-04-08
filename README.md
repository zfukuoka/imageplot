# imageplot

## 概要 overview
ローカルに置いてある samplel.jpg を読み込み、読み込んだ画像の色をCIE xyz空間、ITU-R BT.601 CbCr空間、CIE L\*a\*b\*(D50)、CIE L\*C\*h(D50)にプロットします。

This program plots color on CIE xy, ITU-R BT.601 CbCr, CIE L\*a\*b\*(D50) and CIE L\*C\*h color space using sample.jpg which located on local file system.

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

- [YUV - Wikipedia](https://ja.wikipedia.org/wiki/YUV)

- [Lab色空間 - Wikipedia](https://ja.wikipedia.org/wiki/Lab%E8%89%B2%E7%A9%BA%E9%96%93)

- [L*a*b*色空間（CIE 1976）-楽しく学べる知恵袋 | コニカミノルタ](https://www.konicaminolta.jp/instruments/knowledge/color/section5/08.html)

## ベンチマーク benchmark

下表はある時点のソースを用いて、Raspberry Pi 2 で実行にかかった時間（時間計測を行っている出力の期間）を記す。
なお、Out of order に対応した 今時のそこそこのCPU環境( Core i7、 Rapberry Pi 4、Jetson nanoなど)ではコード完成初期バージョンですら、2～3秒程度で終わってしまうので、割愛。

| date of codes | elapsed time | memo |
| :---: | ---: | :--- |
| 2020/04/08 | 1.8 sec | L\*a\*b\* と L\*C\*hのプロット機能追加 |
| 2020/03/12 | 1.0 sec | CbCrのプロット機能追加 |
| 2020/03/06 | 0.6 sec | もう1か所あった配列での繰り返し計算をコードによるループから numpy で完結するよう変更 |
| 2020/03/05 | 7.3 sec | ある個所の配列での繰り返し計算をコードによるループから numpy で完結するよう変更 |
| 2020/02/22 | 9.5 sec | 粒度の高いデフォルトのfloat64から明示的にfloat32に変更
| 2019/09/27 | 13.9 Sec | コード完成初期バージョン |
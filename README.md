# Master-s_Researches
## ソースコード
ソースコードはすべてsrc階層の中に含まれている。
```
src/
  ├ calibration.cpp        //キャリブレーション用のプログラム
  ├ camera.xml             //キャリブレーションによって得られるカメラの外部関数
  ├ hog_detector.cpp       //HOG特徴を使った画像変換のテスト表示プログラム
  ├ match_Tsukuba_LBP.cpp  //前処理にLBP特徴と画像縮小を使用した自己位置推定のプログラム　卒研で使用した
  ├ positioning_MT.cpp     //マルチスレッドを使用した自己位置推定のプログラム　修士研究で使用している本番環境
  ├ stereoGUI.cpp          //OpenCV内関数のステレオBMのパラメータを設定するGUI
　├ stereo_camera.cpp      //マルチスレッドでステレオ画像をを表示するためのテストプログラム
　├ sub_image.cpp          //差分画像を表示するためのテストプログラム
  ├test.cpp                //いろいろなアルゴリズムを試してみるテスト環境
```

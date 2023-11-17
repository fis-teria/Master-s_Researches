///*
#include <exiv2/exiv2.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
  using namespace std;

  // 画像のファイル名
    cv::VideoCapture cap(0); // デバイスのオープン
    cv::Mat frame;
    cap >> frame;
    cv::imwrite("images/exiv/focus.jpg",frame);
    std::string img_name = "images/exiv/focus.jpg";

  try
  {
    // 画像を開く
    Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(img_name);
    // Exifデータを読み込む
    image->readMetadata();

    Exiv2::ExifData &exif = image->exifData();

    if (exif.empty())
    {
      cerr << "no exif" << endl;
      return -1;
    }

    // Exifデータの表示
    int num = 0;
    for (auto i = exif.begin(); i != exif.end(); ++i)
    {
      if (num == 7)
        cout << num << " " << i->key() << " " << i->value() << endl;
      num++;
    }

    return 0;
  }
  catch (Exiv2::Error &e)
  {
    cout << e.what() << endl;
    return -1;
  }
}
//*/
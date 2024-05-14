#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>


int image_to_video(std::string result, std::string image_name, std::string ext, int frame_num, int frame_num2, double frame_rate)
{

    // {image_name}_00x.ext にするための桁数の取得
    int digit = std::to_string(frame_num).length();

    std::stringstream base;
    base << image_name << std::setw(digit) << std::setfill('0') << 0 << ext;
    std::cout << base.str() << std::endl;

    cv::Mat Img = cv::imread(base.str(), 1);

    std::cout << Img.cols << std::endl;

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(result, fourcc, frame_rate, cv::Size(Img.cols, Img.rows), true);
    std::cout << 0 << std::endl;
    //cv::imshow("Img", Img);
    //cv::waitKey(0);

    for (int i = 0; i < frame_num2; i++)
    {

        // {image_name}_00x.extの文字列作成
        std::stringstream ss;
        ss << image_name << std::setw(digit) << std::setfill('0') << i << ext;
        std::cout << ss.str() << std::endl;

        // フレームを取得する
        Img = cv::imread(ss.str().c_str(), 1);

        if (Img.empty())
        {
            return -1;
        }

        // フレームの書き出し
        writer << Img;
    }
    return 0;
}

int main()
{
    std::string result = "sample_data/excolor.mp4";
    std::string image_name = "ex_data/color/";
    int frame_num2 = 75;

    int frame_num = 100000;
    std::string ext = ".jpg";
    double frame_rate = 60.0;

    int ret = image_to_video(result, image_name, ext, frame_num, frame_num2, frame_rate);

    if (ret)
        std::cout << "フレームが指定の数ありませんでした" << std::endl;
    else
        std::cout << "動画の書き出しが完了しました。" << std::endl;

    return 0;
}
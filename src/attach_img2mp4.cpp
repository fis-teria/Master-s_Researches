#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int image_to_video(std::string result, std::vector<std::string> image_name, std::string ext, int frame_num, int frame_num2, double frame_rate)
{

    // {image_name}_00x.ext にするための桁数の取得
    int digit = std::to_string(frame_num).length();

    std::vector<std::stringstream> base;
    base.resize(4);
    base[0] << image_name[0] << std::setw(digit) << std::setfill('0') << 0 << ext;
    base[1] << image_name[1] << std::setw(digit) << std::setfill('0') << 0 << ext;
    base[2] << image_name[2] << std::setw(digit) << std::setfill('0') << 0 << ext;
    base[3] << image_name[3] << std::setw(digit) << std::setfill('0') << 0 << ext;
    std::cout << base[0].str() << " : " << base[1].str() << std::endl;

    cv::Mat LRhimg, LRDhimg;
    cv::Mat getImg;

    cv::Mat Img[4];
    Img[0] = cv::imread(base[0].str(), 1);
    Img[1] = cv::imread(base[1].str(), 1);
    Img[2] = cv::imread(base[2].str(), 1);
    Img[3] = cv::imread(base[3].str(), 1);
    // cv::imshow("Img", Img);
    // cv::waitKey(0);

    std::cout << Img[0].cols << std::endl;

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(result, fourcc, frame_rate, cv::Size(2 * Img[0].cols, 2 * Img[0].rows), true);
    std::cout << writer.getBackendName() << std::endl;

    for (int i = 0; i < frame_num2; i++)
    {

        // {image_name}_00x.extの文字列作成
        std::vector<std::stringstream> ss;
        ss.resize(4);
        ss[0] << image_name[0] << std::setw(digit) << std::setfill('0') << i << ext;
        ss[1] << image_name[1] << std::setw(digit) << std::setfill('0') << i << ext;
        ss[2] << image_name[2] << std::setw(digit) << std::setfill('0') << i << ext;
        ss[3] << image_name[3] << std::setw(digit) << std::setfill('0') << i << ext;
        std::cout << ss[0].str() << " : " << ss[1].str() << std::endl;
        std::cout << ss[2].str() << " : " << ss[3].str() << std::endl;

        // フレームを取得する
        Img[0] = cv::imread(ss[0].str().c_str(), 1);
        Img[1] = cv::imread(ss[1].str().c_str(), 1);
        Img[2] = cv::imread(ss[2].str().c_str(), 1);
        Img[3] = cv::imread(ss[3].str().c_str(), 1);
        if (Img[0].empty())
        {
            std::cout << ss[0].str() << " is not found" << std::endl;
            return -1;
        }
        if (Img[1].empty())
        {
            std::cout << ss[1].str() << " is not found" << std::endl;
            return -1;
        }
        if (Img[2].empty())
        {
            std::cout << ss[2].str() << " is not found" << std::endl;
            return -1;
        }
        if (Img[3].empty())
        {
            std::cout << ss[3].str() << " is not found" << std::endl;
            return -1;
        }
        cv::hconcat(Img[0], Img[1], LRhimg);
        cv::hconcat(Img[2], Img[3], LRDhimg);
        
        cv::vconcat(LRhimg, LRDhimg, getImg);
        //cv::imshow("a", getImg);
        //cv::waitKey(0);
        
        // フレームの書き出し
        writer << getImg;
    }
    return 0;
}

int main()
{
    std::string result = "resultmv/result_20231231.mp4";
    std::vector<std::string> image_name;
    image_name.resize(4);
    image_name[0] = "images/20231231/left/";
    image_name[1] = "images/20231231/right/";
    image_name[2] = "images/20231231/left_depth/";
    image_name[3] = "images/20231231/right_depth/";
    int frame_num2 = 5192;

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
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include <time.h>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <functional>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

class readXml
{
public:
    cv::Mat camera_matrix, distcoeffs;
    std::string FILE_NAME;

public:
    readXml()
    {
    }

public:
    readXml(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        std::cout << "constractor" << std::endl;
    }

public:
    readXml(const readXml &xml)
    {
        std::cout << "copy constractor" << std::endl;
    }

public:
    void read(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        std::cout << "member function" << std::endl;
    }
};

int main(){
    readXml xml00 = readXml("camera/out_camera_data00.xml");
    readXml xml02 = readXml("camera/out_camera_data02.xml");
    /*
    for (int x = 0; x < xml00.camera_matrix.cols; x++)
    {
        for (int y = 0; y < xml00.camera_matrix.rows; y++)
        {
            std::cout << xml00.camera_matrix.at<double>(y, x) << " ";
        }
        std::cout << std::endl;
    }
    */
    cv::VideoCapture cap(0); // デバイスのオープン
                             // cap.open(0);//こっちでも良い．
                             // capの画像の解像度を変える部分 word CaptureChange
    /*
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    */
    if (!cap.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return 0;
    }

    cv::VideoCapture cap2(2); // デバイスのオープン
                              // cap.open(0);//こっちでも良い．
                              // capの画像の解像度を変える部分 word CaptureChange
    /*
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    */
    if (!cap2.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return 0;
    }

    cv::Mat frame; // 取得したフレーム
    cv::Mat distort;
    cv::Mat matx, maty;

    cv::Mat frame2; // 取得したフレーム
    cv::Mat distort2;

    cv::Mat matx2, maty2;
    cv::initUndistortRectifyMap(xml02.camera_matrix, xml02.distcoeffs, cv::Mat(), xml02.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx, maty);
    cv::initUndistortRectifyMap(xml00.camera_matrix, xml00.distcoeffs, cv::Mat(), xml00.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx2, maty2);

    cv::Mat f1g, f2g;
    while (1) // 無限ループ
    {
        cap >> frame;
        cap2 >> frame2;
        cv::remap(frame, distort, matx, maty, cv::INTER_LINEAR);
        cv::remap(frame2, distort2, matx2, maty2, cv::INTER_LINEAR);

        cv::resize(distort, distort, cv::Size(), 0.7, 0.7);
        cv::resize(distort2, distort2, cv::Size(), 0.7, 0.7);


        cv::imshow("a", distort);
        cv::imshow("b", distort2);

        const int key = cv::waitKey(10);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }
    }
    cv::destroyAllWindows();

    return 0;
}
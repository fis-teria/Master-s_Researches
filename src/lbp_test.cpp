#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

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
#include <random>
#include <filesystem>
#include <chrono>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

std::vector<int> UNIFORMED_LUT(256, 0);



/*
    test data
    images/20231231/left/004567.jpg
    images/20240220/left/000036.jpg
    images/20240327/left/000339.jpg
    images/test_img/left.JPG
*/
const std::string LEFT_IMG = "images/20240220/left/000000.jpg";

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_LBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc;
    // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    // cv::imshow("first", src);
    // cv::imshow("third", lbp);
    for (int x = 1; x < padsrc.cols - 1; x++)
    {
        for (int y = 1; y < padsrc.rows - 1; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= padsrc.at<unsigned char>(y, x))
                        lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                }
            }
        }
    }
    dst = lbp.clone();
    // cv::imshow("second", lbp);
}

void cvt_ILBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc, gray;
    //cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    // cv::imshow("first", src);
    // cv::imshow("third", lbp);

    int ave = 0;
    for (int x = 1; x < padsrc.cols - 1; x++)
    {
        for (int y = 1; y < padsrc.rows - 1; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    ave += (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i);
                }
            }
            ave /= 9;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= ave)
                        lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                }
            }
            ave = 0;
        }
    }
    dst = lbp.clone();
    // cv::imshow("second", lbp);
}

void cvt_ILLBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc, gray;
    //cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    // cv::imshow("first", src);
    // cv::imshow("third", lbp);

    int ave = 0;
    for (int x = 1; x < padsrc.cols - 1; x++)
    {
        for (int y = 1; y < padsrc.rows - 1; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    ave += (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i);
                }
            }
            ave /= 8;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= ave)
                    {
                        lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                        //lbp.at<unsigned char>(y - 1, x - 1) += 255;
                    }
                }
            }
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
                    //lbp.at<unsigned char>(y - 1, x - 1) = UNIFORMED_LUT[lbp.at<unsigned char>(y - 1, x - 1)];
                    // std::cout << (int)lbp.at<unsigned char>(y - 1, x - 1) << std::endl;
                }
            }
        }
    }
    dst = lbp.clone();
    // cv::imshow("second", lbp);
}

void make_LUT(std::vector<int> &lut)
{
    std::cout << "make_LUT start" << std::endl;
    std::ifstream ifs("tools/Uniformed_LBP_Table.txt");
    std::string str;
    int count = 0;
    if (ifs.fail())
    {
        std::cout << "not lut file" << std::endl;
        return;
    }

    while (getline(ifs, str))
    {
        lut[count] = atoi(str.c_str());
        if (lut[count] != 0)
        {
            double dist = 255 / 35;
            lut[count] = dist * lut[count];
        }
        count++;
    }
    std::cout << "make_LUT end" << std::endl;
}

int main(){
    make_LUT(UNIFORMED_LUT);
    cv::Mat img = cv::imread(LEFT_IMG, 0);
    cv::Mat lbp, ilbp, illbp;
    cvt_LBP(img, lbp);
    cvt_ILBP(img, ilbp);
    cvt_ILLBP(img, illbp);

    cv::imshow("a", lbp);
    cv::imshow("b", ilbp);
    cv::imshow("c", illbp);

    const int key = cv::waitKey(0);
    if (key == 'q' /*113*/) // qボタンが押されたとき
    {
        cv::imwrite("sample_data/lbp.jpg", lbp);
        cv::imwrite("sample_data/ilbp.jpg", ilbp);
        cv::imwrite("sample_data/illbp10.jpg", illbp);
        /*
        cv::imwrite("sample_data/dst.jpg", dst);
        cv::imwrite("sample_data/lbp.jpg", lbp);
        cv::imwrite("sample_data/depth.jpg", depth_image);
        cv::imwrite("sample_data/color.jpg", color_image);
        */
        ; // whileループから抜ける．
    }
    return 0;
}
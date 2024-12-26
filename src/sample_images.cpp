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
#include <random>
#include <bitset>
#include <filesystem>
#include <zmq.h>
#include <zmq.hpp>

#include "common.hpp"
#include "rs2_utils.hpp"
#include "structure.hpp"

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};
void cvt_ILBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc, gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
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
    //cv::imshow("second", lbp);
}

void cvt_ELBP(const cv::Mat &src, cv::Mat &dst, std::vector<int> &UNIFORMED_LUT, int w)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc, gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    copyMakeBorder(gray, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

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
            ave /= w;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= ave)
                        lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                }
            }
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
                    lbp.at<unsigned char>(y - 1, x - 1) = UNIFORMED_LUT[lbp.at<unsigned char>(y - 1, x - 1)];
                    // std::cout << (int)lbp.at<unsigned char>(y - 1, x - 1) << std::endl;
                }
            }
        }
    }
    dst = lbp.clone();
    //cv::imshow("second", lbp);
}

void make_LUT(std::vector<int> &lut)
{
    std::cout << "make_LUT start" << std::endl;
    std::ifstream ifs("../data/tools/Uniformed_LBP_Table.txt");
    std::string str;
    int count = 0;
    lut.resize(255);
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

int main(int argc, char **argv)
{
    std::vector<int> lut(256, 0);
    cv::Mat img = cv::imread("../data/images/corridor/running4/color/000161.jpg");
    cv::Mat night = cv::imread("../data/images/corridor/running1/color/000133.jpg");
    cv::Mat depth = cv::imread("../data/images/corridor/running1/depth/000133.jpg");
    if(img.empty() == 1) return -1;
    if(night.empty() == 1) return -1;
    if(depth.empty() == 1) return -1;
    cv::Mat ilbp, elbp8, elbp9, elbp10, nightelbp, nightgray, nightsobel, n_sobel, n_elbp;


    make_LUT(lut);
    std::cout << "a" <<std::endl;
    cvt_ILBP(img, ilbp);
    std::cout << "a" <<std::endl;
    cvt_ELBP(img, elbp8, lut, 8);
    std::cout << "a" <<std::endl;
    cvt_ELBP(img, elbp9, lut, 9);
    std::cout << "a" <<std::endl;
    cvt_ELBP(img, elbp10, lut, 10);
    std::cout << "a" <<std::endl;
    cvt_ELBP(night, nightelbp, lut, 8);
    std::cout << "a" <<std::endl;
    cv::Sobel(nightgray, nightsobel, CV_8U, 1, 1);
    common::cvt_depth_edge_image(nightsobel, depth, n_sobel);
    common::cvt_depth_edge_image(nightelbp, depth, n_elbp);
    cv::cvtColor(ilbp, ilbp, cv::COLOR_GRAY2BGR);

    cv::imwrite("../data/images/sample/ilbp.jpg", ilbp);
    cv::imwrite("../data/images/sample/elbp8.jpg", elbp8);
    cv::imwrite("../data/images/sample/elbp9.jpg", elbp9);
    cv::imwrite("../data/images/sample/elbp10.jpg", elbp10);
    cv::imwrite("../data/images/sample/n_sobel.jpg", n_sobel);
    cv::imwrite("../data/images/sample/n_elbp.jpg", n_elbp);
    

    return 0;
}
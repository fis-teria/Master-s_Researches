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

#include "common.hpp"

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

std::string make_spath(std::string dir, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + "/00000" + std::to_string(var) + tag;
    }
    else if (var < 100)
    {
        back = dir + "/0000" + std::to_string(var) + tag;
    }
    else if (var < 1000)
    {
        back = dir + "/000" + std::to_string(var) + tag;
    }
    else if (var < 10000)
    {
        back = dir + "/00" + std::to_string(var) + tag;
    }
    std::cout << back << std::endl;
    return back;
}

int main(int argc, char *argv[])
{
    int check = atoi(argv[1]);

    int root_c = 0;
    int depth_c = 0;
    int edge_c = 0;
    int edepth_c = 0;

    switch (check)
    {
    case 0:
        depth_c = 1;
        edge_c = 1;
        edepth_c = 1;
        break;
    case 1:
        root_c = 1;
        edge_c = 2;
        break;

    default:
        return 0;
    }
    std::string root = argv[2];
    std::string depth = argv[3];
    std::string edge = argv[4];
    std::string edepth = argv[5];

    std::vector<int> lut;

    std::string tag = ".jpg";

    cv::Mat ro, dep, ed, edep;
    int count = 0;

    while (true)
    {
        if(root_c == 1)
            ro = cv::imread(common::make_path(root, count, tag));
        if (depth_c == 1)
            dep = cv::imread(make_spath(depth, count, ".jpg"), 1);
        if(edge_c == 1)
            cv::Mat ed = cv::imread(make_spath(edge, count, ".jpg"), 0);


        if(edge_c == 2){
            common::make_LUT(lut);
            common::cvt_ELBP(ro, ed, lut);
            cv::imshow("ed", ed);
            std::cout << common::make_path(edge, count, tag) << std::endl;
            cv::imwrite(common::make_path(edge, count, tag), ed);
        }
        if (edepth_c == 1)
        {
            cv::Mat depth_edge = cv::Mat(dep.rows, dep.cols, CV_8UC3);
            for (int x = 0; x < dep.cols; x++)
            {
                for (int y = 0; y < dep.rows; y++)
                {
                    if (ed.at<unsigned char>(y, x) > 20)
                    {
                        depth_edge.at<cv::Vec3b>(y, x) = dep.at<cv::Vec3b>(y, x);
                    }
                    else
                    {
                        depth_edge.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    }
                }
            }
            cv::imshow("depth_edge", depth_edge);
            cv::imwrite(make_spath(edepth, count, ".jpg"), depth_edge);
        }

        const int key = cv::waitKey(100);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }

        count++;
    }

    return 0;
}

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
    std::string depth = "data/depth";
    std::string edge = "data/edge";
    std::string cdepth = "data/cdepth";

    int count = 0;

    while (true)
    {
        cv::Mat dep = cv::imread(make_spath(depth, count, ".jpg"), 1);
        cv::Mat ed = cv::imread(make_spath(edge, count, ".jpg"), 0);

        cv::imshow("depth", dep);
        cv::imshow("edge", ed);

        cv::Mat depth_edge = cv::Mat(dep.rows, dep.cols, CV_8UC3);
        for(int x = 0; x < dep.cols;x++){
            for(int y = 0; y < dep.rows;y++){
                if(ed.at<unsigned char>(y, x) > 20){
                    depth_edge.at<cv::Vec3b>(y,x) = dep.at<cv::Vec3b>(y,x);
                }else{
                    depth_edge.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        cv::imshow("depth_edge", depth_edge);
        cv::imwrite(make_spath(cdepth, count, ".jpg"), depth_edge);
        const int key = cv::waitKey(100);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }

        count++;

    }


    return 0;
}

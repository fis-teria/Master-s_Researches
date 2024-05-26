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

cv::Mat cvt_EdgeDepth(const cv::Mat &edge, const cv::Mat &depth, cv::Mat &dst)
{
    cv::Mat re = cv::Mat(edge.rows, edge.cols, CV_8UC3);
    for (int x = 0; x < edge.cols; x++)
    {
        for (int y = 0; y < edge.rows; y++)
        {
            if (edge.at<unsigned char>(y, x) <= 10)
            {
                re.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
            else
            {
                re.at<cv::Vec3b>(y, x) = depth.at<cv::Vec3b>(y, x);
            }
        }
    }
    dst = re.clone();
    if(dst.empty()){
        std::cout << "Error dst has not image" << std::endl;
    }
}

void obstacle_VHconcat()
{
    cv::Mat mask, color, edge, depth, edepth, result, obstacle;
    for (int i = 0; i < 10000; i++)
    {
        mask = cv::imread(make_spath("ex_data/mask", i, ".jpg"), 1);
        color = cv::imread(make_spath("ex_data/color", i, ".jpg"), 1);
        edge = cv::imread(make_spath("ex_data/edge", i, ".jpg"), 0);
        depth = cv::imread(make_spath("ex_data/depth", i, ".jpg"), 1);
        if (mask.empty() || color.empty())
        {
            break;
        }

        cv::resize(mask, mask, cv::Size(848, 480));
        std::cout << mask.size() << " " << color.size() << std::endl;
        obstacle = cv::Mat(mask.rows, mask.cols, CV_8UC3);
        result = cv::Mat(2 * mask.rows, 2 * mask.cols, CV_8UC3);

        cvt_EdgeDepth(edge, depth, edepth);

        for (int x = 0; x < mask.cols; x++)
        {
            for (int y = 0; y < mask.rows; y++)
            {
                if (mask.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0))
                {
                    obstacle.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    depth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    edepth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
                else
                {
                    obstacle.at<cv::Vec3b>(y, x) = color.at<cv::Vec3b>(y, x);
                }
            }
        }

        cv::Mat Himg1[2];
        cv::Mat Himg2[2];
        cv::Mat Vimg[2];

        Himg1[0] = mask.clone();
        Himg1[1] = obstacle.clone();
        Himg2[0] = depth.clone();
        Himg2[1] = edepth.clone();

        cv::hconcat(Himg1, 2, Vimg[0]);
        cv::hconcat(Himg2, 2, Vimg[1]);
        cv::vconcat(Vimg, 2, result);

        cv::imshow("a", result);
        //cv::imwrite(make_spath("ex_data/obstacle/", i, ".jpg"), result);
        const int key = cv::waitKey(100);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }
    }
}

void obstacle()
{
    cv::Mat mask, color, edge, depth, edepth_af;
    cv::Mat def_edge = cv::imread("ex_data/edge/000007.jpg", 0);
    cv::Mat def_depth = cv::imread("ex_data/depth/000007.jpg", 1);
    cv::Mat img;
    if (def_edge.empty() || def_depth.empty())
    {
        return;
    }
    img = cv::Mat(def_edge.rows, def_edge.cols, CV_8UC3);
    for (int x = 0; x < def_edge.cols; x++)
    {
        for (int y = 0; y < def_edge.rows; y++)
        {
            if (def_edge.at<unsigned char>(y, x) <= 10)
            {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
            else
            {
                img.at<cv::Vec3b>(y, x) = def_depth.at<cv::Vec3b>(y, x);
            }
        }
    }
    cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("aaa", img);
    std::ofstream ofs("ex_data/result.csv");

    for (int i = 0; i < 10000; i++)
    {
        mask = cv::imread(make_spath("ex_data/mask", i, ".jpg"), 1);
        color = cv::imread(make_spath("ex_data/color", i, ".jpg"), 1);
        edge = cv::imread(make_spath("ex_data/edge", i, ".jpg"), 0);
        depth = cv::imread(make_spath("ex_data/depth", i, ".jpg"), 1);
        if (mask.empty() || color.empty())
        {
            break;
        }

        cv::resize(mask, mask, cv::Size(848, 480));
        std::cout << mask.size() << " " << color.size() << std::endl;

        cvt_EdgeDepth(edge, depth, edepth_af);
        ///*
        for (int x = 0; x < mask.cols; x++)
        {
            for (int y = 0; y < mask.rows; y++)
            {
                if (mask.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0))
                {
                    depth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    edepth_af.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
                else
                {
                }
            }
        }
        //*/
        cv::Mat sl_tim;
        double min_sl, max_sl;
        cv::Point p_min_sl, p_max_sl;
        cvtColor(edepth_af, edepth_af, cv::COLOR_BGR2GRAY);
        cv::matchTemplate(img, edepth_af(cv::Range(edepth_af.rows / 10, (9 * edepth_af.rows) / 10), cv::Range(edepth_af.cols / 10, (9 * edepth_af.cols) / 10)), sl_tim, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(sl_tim, &min_sl, &max_sl, &p_min_sl, &p_max_sl);

        
        ofs << i << ", " << max_sl << std::endl;


        cv::imshow("a", edepth_af);
        // cv::imwrite(make_spath("ex_data/obstacle/", i, ".jpg"), result);
        const int key = cv::waitKey(100);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }
    }
    ofs.close();
}
int main()
{
    obstacle();
    //obstacle_VHconcat();
    return 0;
}
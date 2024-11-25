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

void localization(rs2_utils &rs2_utils)
{
    //std::thread th1(system, "python3 ../pycoral/code/main.py");
    cv::Mat elbp_img, dst_img;
    std::vector<ResultImage> result;
    std::vector<int> lut(256, 0);
    common::make_LUT(lut);
    while (1)
    {
        time_t begin = clock();
        std::cout << "--------------Start Localization--------------" << std::endl;
        int zmq_loop = 0;
        rs2_utils.get_frames();
        // cv::imshow("a", rs2_utils.color_image);
        // cv::imshow("a", rs2_utils.depth_image);
        // cv::waitKey(100);

        //cvt ELBP Image
        common::cvt_ELBP(rs2_utils.color_image, elbp_img, lut);
        common::cvt_depth_edge_image(elbp_img, rs2_utils.depth_image, dst_img);

        std::cout << "image serve\t";

        common::zmq_serve(rs2_utils.color_image, "color");
        std::cout << "OK" << std::endl;
        std::cout << "waiting time\t";
        zmq_loop = common::zmq_n_recive();
        std::cout << "Finish\nzmp loop " << zmq_loop << std::endl;
        for (int n = 0; n < zmq_loop; n++)
        {
            std::cout << "challenge recive img" << std::endl;
            result.push_back(common::zmq_img_recive());
            common::zmq_check_serve();

            std::cout << "complete recive img" << std::endl;
        }

        ///*
        for (int i = 0; i < zmq_loop; i++)
        {
            //std::string win_name = "win_" + std::to_string(i);
            //cv::imshow(win_name, result[i].image);
            result[i].get_data();
        }
        //cv::waitKey(10);
        //*/
        std::cout << "---------------End Localization---------------" << std::endl;
        time_t end = clock();
        common::print_elapsed_time(begin, end);
    }
    //th1.join();
    return;
}

void localization_debug(){
    //std::thread th1(system, "python3 ../pycoral/code/main.py");
    std::vector<ResultImage> result;
    while (1)
    {
        time_t begin = clock();
        std::cout << "--------------Start Localization--------------" << std::endl;
        int zmq_loop = 0;
        cv::Mat color_image = cv::imread("../data/images/amalab/lab_root_2/000300.jpg", 1);
        std::cout << color_image.size() <<std::endl;
        // cv::waitKey(100);

        common::zmq_serve(color_image, "color");
        std::cout << "served img" << std::endl;
        zmq_loop = common::zmq_n_recive();
        std::cout << "zmp loop " << zmq_loop << std::endl;
        for (int n = 0; n < zmq_loop; n++)
        {
            std::cout << "challenge recive img" << std::endl;
            result.push_back(common::zmq_img_recive());
            common::zmq_check_serve();

            std::cout << "complete recive img" << std::endl;
        }

        ///*
        for (int i = 0; i < zmq_loop; i++)
        {
            std::string win_name = "win_" + std::to_string(i);
            cv::imshow(win_name, result[i].image);
        }
        cv::waitKey(10);
        //*/
        std::cout << "--------------End Localization--------------" << std::endl;
        time_t end = clock();
        common::print_elapsed_time(begin, end);
    }
    //th1.join();
    return;
}

int main(int argc, char **argv)
{
    if(argc <=1){
        std::cout << "---------Missing argument---------\n";
        std::cout << "0: Used RealSense localization\nother: debug mode" << std::endl;
        return -1;
    }

    int debug = atoi(argv[1]);
    if (debug == 0)
    {
        rs2_utils rs2_utils;
        // rs2_utils.info();
        rs2_utils.pipe_start();
        rs2_utils.test_get_frames();
        localization(rs2_utils);
    }
    else
    {
        localization_debug();
    }
    return 0;
}
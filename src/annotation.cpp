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

int main(int argc, char **argv)
{
    int start, end;
    int anno = 0;
    int before_anno = 0;
    int count = 0;
    int lcount = 0;
    int input = 0;
    std::string loop_ans;

    std::string run_dir = argv[1];
    std::string run_path = run_dir + "/color";
    std::string test_dir = argv[2];
    std::string test_path = test_dir + "/color";
    int frame_num = atoi(argv[3]);

    std::string output = run_dir + "/annotation.txt";
    std::ifstream ifs(output);
    std::string line;
    while (getline(ifs, line))
    {
        anno = atoi(line.c_str());
    }
    ifs.close();
    std::cout << "The previous value of annno is " << anno << std::endl;

    std::cout << "output file " << output << std::endl;
    std::ofstream ofs(output, std::ios::app);

    std::string save_file = "../data/logs/save/annotation_save.txt";
    int save_num = common::read_save(save_file);

    cv::Mat run_img, test_img_0, test_img_1, test_img_2, test_img_3, test_img_4, test_img_5, test_img_6, test_img_7, test_img_8, test_img_9, test_img_10;


    for (int i = save_num; i < 100000; i++)
    {
        start = anno - 5;
        end = anno + 5;
        if (anno - 5 < 0)
        {
            start = 0;
            end = start + 10;
        }
        if (anno + 5 > frame_num)
        {
            end = frame_num;
            start = end - 10;
        }
        run_img = cv::imread(common::make_path(run_path, i, ".jpg"));
        if (run_img.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(run_img, std::__cxx11::to_string(i), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_0 = cv::imread(common::make_path(test_path, start, ".jpg"));
        if (test_img_0.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_0, std::__cxx11::to_string(start), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_1 = cv::imread(common::make_path(test_path, start + 1, ".jpg"));
        if (test_img_1.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_1, std::__cxx11::to_string(start + 1), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_2 = cv::imread(common::make_path(test_path, start + 2, ".jpg"));
        if (test_img_2.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_2, std::__cxx11::to_string(start + 2), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_3 = cv::imread(common::make_path(test_path, start + 3, ".jpg"));
        if (test_img_3.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_3, std::__cxx11::to_string(start + 3), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_4 = cv::imread(common::make_path(test_path, start + 4, ".jpg"));
        if (test_img_4.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_4, std::__cxx11::to_string(start + 4), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_5 = cv::imread(common::make_path(test_path, start + 5, ".jpg"));
        if (test_img_5.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_5, std::__cxx11::to_string(start + 5), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_6 = cv::imread(common::make_path(test_path, start + 6, ".jpg"));
        if (test_img_6.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_6, std::__cxx11::to_string(start + 6), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_7 = cv::imread(common::make_path(test_path, start + 7, ".jpg"));
        if (test_img_7.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_7, std::__cxx11::to_string(start + 7), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_8 = cv::imread(common::make_path(test_path, start + 8, ".jpg"));
        if (test_img_8.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_8, std::__cxx11::to_string(start + 8), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_9 = cv::imread(common::make_path(test_path, start + 9, ".jpg"));
        if (test_img_9.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_9, std::__cxx11::to_string(start + 9), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);
        test_img_10 = cv::imread(common::make_path(test_path, end, ".jpg"));
        if (test_img_10.empty() != 0)
        {
            std::cout << "running file path not found" << std::endl;
            return -1;
        }
        cv::putText(test_img_10, std::__cxx11::to_string(start + 10), cv::Point(30, 75), cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 3);

        cv::Mat upper, lower;
        cv::hconcat(run_img, test_img_0, upper);
        cv::hconcat(upper, test_img_1, upper);
        cv::hconcat(upper, test_img_2, upper);
        cv::hconcat(upper, test_img_3, upper);
        cv::hconcat(upper, test_img_4, upper);
        cv::hconcat(test_img_5, test_img_6, lower);
        cv::hconcat(lower, test_img_7, lower);
        cv::hconcat(lower, test_img_8, lower);
        cv::hconcat(lower, test_img_9, lower);
        cv::hconcat(lower, test_img_10, lower);
        cv::vconcat(upper, lower, upper);

        cv::resize(upper, upper, cv::Size(), 0.4, 0.4);
        cv::imshow("upper", upper);
        cv::waitKey(100);
        std::cout << "Please enter frame number here : ";
        while (1)
        {
            if (lcount > 0)
            {
                input = anno + 1;
                lcount--;
                break;
            }
            std::cin >> input;
            if (input >= start && input <= end)
            {
                break;
            }
            std::cout << "Please enter a number between" << start << " < X < " << end << std::endl;
        }
        std::cout << "The number you entered is " << input << std::endl;
        before_anno = anno;
        anno = input;
        if (lcount == 0)
        {
            if (anno - before_anno == 1)
            {
                count++;
            }
            else
            {
                count = 0;
            }
            if (count == 10)
            {
                lcount = 10;
            }
        }else if(lcount == 1){
            std::cout << "Do you want to continue the loop? ( y or n)" << std::endl;
            std::cin >> loop_ans;
            if(loop_ans == "y"){
                lcount = 10;
            }
        }
        ofs << anno << std::endl;
        common::save(save_file, i + 1);
    }
    ofs.close();
    return 0;
}
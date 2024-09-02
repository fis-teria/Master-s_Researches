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

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_ILBP(const cv::Mat &src, cv::Mat &dst)
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
            ave /= 8;
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

class rs2_utils
{
private:
    int WIDTH = 848;
    int HEIGHT = 480;
    int FPS = 30;
    int DEPTH_WIDTH = 848;
    int DEPTH_HEIGHT = 480;
    int DEPTH_FPS = 30;

    rs2::config config;
    rs2::pipeline pipe;
    rs2::colorizer color_map;
    

public:
    cv::Mat color_image;
    cv::Mat depth_image;
    rs2_vector gyro;
    rs2_vector accel;

    rs2_utils()
    {
        initialize();
    }

    void initialize(){
        std::cout << "config images";
        config.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
        config.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);
        config.enable_stream(RS2_STREAM_GYRO);
        config.enable_stream(RS2_STREAM_ACCEL);
        std::cout << "\tOK" << std::endl;
    }

    void pipe_start()
    {
        pipe.start(this->config);
    }

    void test_get_frames()
    {
        std::cout << "start up check";
        for (int i = 0; i < 3; i++)
        {
            rs2::frameset test_frames = pipe.wait_for_frames();
            cv::waitKey(10);
        }
        std::cout << "\tOK" << std::endl;
    }

    void get_frames(){
        rs2::align align(RS2_STREAM_COLOR);
        rs2::frameset frames = this->pipe.wait_for_frames();
        auto aligned_frames = align.process(frames);

        rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
        rs2::video_frame depth_frame = aligned_frames.get_depth_frame().apply_filter(color_map);
        
        color_image = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        depth_image = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        if (rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
        {
            accel = accel_frame.get_motion_data();
        }

        if (rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
        {
            gyro = gyro_frame.get_motion_data();
        }
    }
};
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

// Include short list of convenience functions for rendering

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char *argv[])
try
{

    make_LUT(UNIFORMED_LUT);
    int WIDTH = 848;
    int HEIGHT = 480;
    int FPS = 30;
    int DEPTH_WIDTH = 848;
    int DEPTH_HEIGHT = 480;
    int DEPTH_FPS = 30;
    int c = 0;
    std::string color_dir = "data/color";
    std::string depth_dir = "data/depth";
    std::string edge_dir = "data/edge";

    rs2::config config;
    std::cout << "config images";
    config.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    config.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);
    config.enable_stream(RS2_STREAM_GYRO);
    config.enable_stream(RS2_STREAM_ACCEL);
    std::cout << "\tOK" << std::endl;


    rs2::pipeline pipe;
    std::cout << "pipe start";
    pipe.start(config);
    std::cout << "\t OK" << std::endl;

    rs2::colorizer color_map;
    rs2::align align(RS2_STREAM_COLOR);

    std::cout << "start up check";
    for (int i = 0; i < 3; i++)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        cv::waitKey(10);
    }
    std::cout << "\tOK" << std::endl;

    while (true)
    {
        if (c == 0)
        {
            std::cout << "get frame" << std::endl;
        }
        rs2::frameset frames = pipe.wait_for_frames();
        auto aligned_frames = align.process(frames);
        rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
        rs2::video_frame depth_frame = aligned_frames.get_depth_frame().apply_filter(color_map);
        if (c == 0)
        {
            std::cout << "\tOK" << std::endl;
            std::cout << "frame change images";
        }

        cv::Mat color_image(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_image(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        if (c == 0)
        {
            std::cout << "\tOK" << std::endl;
            std::cout << "images copyTo Mat";
        }

        cv::Mat images(cv::Size(2 * WIDTH, HEIGHT), CV_8UC3);
        cv::Mat color_positon(images, cv::Rect(0, 0, WIDTH, HEIGHT));
        color_image.copyTo(color_positon);
        cv::Mat depth_positon(images, cv::Rect(WIDTH, 0, WIDTH, HEIGHT));
        depth_image.copyTo(depth_positon);

        if (c == 0)
        {
            std::cout << "\tOK" << std::endl;
        }

        cv::Mat lbp;
        cv::Mat dst = cv::Mat(depth_image.rows, depth_image.cols, CV_8UC3);
        cvt_ILBP(color_image, lbp);
        cv::imshow("lbp", lbp);

        for (int x = 0; x < depth_image.cols; x++)
        {
            for (int y = 0; y < depth_image.rows; y++)
            {
                if (lbp.at<unsigned char>(y, x) != 0)
                {
                    dst.at<cv::Vec3b>(y, x) = depth_image.at<cv::Vec3b>(y, x);
                }
                else
                {
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        // Find and retrieve IMU data
        if (rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
        {
            rs2_vector accel_sample = accel_frame.get_motion_data();
            std::cout << "Accel:" << accel_sample.x << ", " << accel_sample.y << ", " << accel_sample.z;
            //...
        }

        if (rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
        {
            rs2_vector gyro_sample = gyro_frame.get_motion_data();
            std::cout << "\tGyro:" << gyro_sample.x << ", " << gyro_sample.y << ", " << gyro_sample.z << std::endl;
            //...
        }

        cv::imshow("dst", dst);
        cv::imshow("depth", depth_image);
        cv::imshow("images", images);

        const int key = cv::waitKey(100);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            /*
            cv::imwrite("sample_data/dst.jpg", dst);
            cv::imwrite("sample_data/lbp.jpg", lbp);
            cv::imwrite("sample_data/depth.jpg", depth_image);
            cv::imwrite("sample_data/color.jpg", color_image);
            //*/
            break; // whileループから抜ける．
        }
        std::cout << make_spath(color_dir, c, ".jpg") << std::endl;
        // cv::imwrite(make_spath(color_dir, c, ".jpg"), color_image);
        // cv::imwrite(make_spath(depth_dir, c, ".jpg"), depth_image);
        // cv::imwrite(make_spath(edge_dir, c, ".jpg"), lbp);

        c++;
    }

    pipe.stop();

    return EXIT_SUCCESS;
}
catch (const rs2::error &e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

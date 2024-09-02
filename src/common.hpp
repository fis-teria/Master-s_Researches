#ifndef INCLUDE_COMMON_HPP
#define INCLUDE_COMMON_HPP

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

#include "structure.hpp"

namespace common
{
    std::string make_path(std::string dir, int var, std::string tag)
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

    void print_elapsed_time(clock_t begin, clock_t end)
    {
        float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
        printf("Elapsed Time: %15.15f sec\n", elapsed);
    }

    int LBP_filter[3][3] = {{64, 32, 16},
                            {128, 0, 8},
                            {1, 2, 4}};

    //void cvt_ELBP(const cv::Mat &src, cv::Mat &dst, std::vector<int> &UNIFORMED_LUT)
    void cvt_ELBP(const cv::Mat &src, cv::Mat &dst, std::vector<int> &UNIFORMED_LUT)
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

    void my_free(void *data, void *hint)
    {
        free(data);
    }

    void zmq_serve(cv::Mat &image, std::string c_mode)
    {
        int i;

        // system("../pycoral/code/zeroMQ.py");

        // File Open
        /*
        if (!strcmp(c_mode.c_str(), "color"))
        {
            printf("color\n");
            image = cv::imread("ex_data/color/000030.jpg", 1);
        }
        else
        {
            printf("gray\n");
            image = cv::imread("ex_data/color/000030.jpg", 0);
        }
        */
        // Image Info
        int32_t info[3];
        info[0] = (int32_t)image.rows;
        info[1] = (int32_t)image.cols;
        info[2] = (int32_t)image.type();

        // Open ZMQ Connection
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_REQ);
        socket.connect("tcp://localhost:5555");

        // Send Rows, Cols, Type
        for (i = 0; i < 3; i++)
        {
            zmq::message_t msg((void *)&info[i], sizeof(int32_t), NULL);
            socket.send(msg, ZMQ_SNDMORE);
        }

        // Pixel data
        void *data = malloc(image.total() * image.elemSize());
        memcpy(data, image.data, image.total() * image.elemSize());

        // Send Pixel data
        zmq::message_t msg2(data, image.total() * image.elemSize(), my_free, NULL);
        socket.send(msg2);

        return;
    }
    void zmq_check_serve()
    {
        int i;
        // Image Info
        int info_size = 1;
        int32_t info[info_size];
        int32_t check_ok = 1;
        info[0] = (int32_t)check_ok;

        // Open ZMQ Connection
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_REQ);
        socket.connect("tcp://localhost:5557");

        // Send Rows, Cols, Type
        for (i = 0; i < info_size; i++)
        {
            zmq::message_t msg((void *)&info[i], sizeof(int32_t), NULL);
            socket.send(msg);
        }
        std::cout << "send check msg" << std::endl;

        return;
    }

    int zmq_n_recive()
    {
        // Open ZMQ Connection
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_REP);
        socket.bind("tcp://*:5556");

        zmq::message_t rcv_msg;
        socket.recv(&rcv_msg, 0);

        int n = *(int *)rcv_msg.data();

        return n;
    }

    ResultImage zmq_img_recive()
    {
        int cnt = 0;
        int rows, cols, type;
        int xmin, xmax, ymin, ymax;
        int id = -1;
        cv::Mat img;
        void *data;
        ResultImage result;

        // Open ZMQ Connection
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_REP);
        socket.bind("tcp://*:5556");

        while (1)
        {
            zmq::message_t rcv_msg;
            socket.recv(&rcv_msg, 0);

            // Receive Data from ZMQ
            switch (cnt)
            {
            case 0:
                rows = *(int *)rcv_msg.data();
                break;
            case 1:
                cols = *(int *)rcv_msg.data();
                break;
            case 2:
                type = *(int *)rcv_msg.data();
                break;
            case 3:
                data = (void *)rcv_msg.data();
                printf("rows=%d, cols=%d type=%d\n", rows, cols, type);

                if (type == 2)
                {
                    img = cv::Mat(rows, cols, CV_8UC1, data);
                }
                else
                {
                    img = cv::Mat(rows, cols, CV_8UC3, data);
                }
                break;
            case 4:
                xmin = *(int *)rcv_msg.data();
                break;
            case 5:
                xmax = *(int *)rcv_msg.data();
                break;
            case 6:
                ymin = *(int *)rcv_msg.data();
                break;
            case 7:
                ymax = *(int *)rcv_msg.data();
                break;
            case 8:
                id = *(int *)rcv_msg.data();
                
                cv::resize(img, img, cv::Size(xmax-xmin, ymax-ymin));
                cv::imshow("recv.bmp", img);
                img.copyTo(result.image);
                result.xmin = xmin;
                result.xmax = xmax;
                result.ymin = ymin;
                result.ymax = ymax;
                result.id = id;
                cv::waitKey(100);
                break;
            }

            if (!rcv_msg.more())
            {
                // No massage any more
                break;
            }

            cnt++;
        }

        return result;
    }
}
#endif
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

std::mutex m;

cv::Mat src0;
cv::Mat src1;
cv::Mat diff;

int cn1 = 0;
int cn2 = 2;

int get_c1 = 0;
int get_c2 = 0;

int wait_c1 = 0;
int wait_c2 = 0;

int fin = 0;

std::string dir = "images/2023_1128/left";
std::string dir2 = "images/2023_1128/right";
std::string tag = ".jpg";

void print_elapsed_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %15.7f sec\n", elapsed);
}

std::string make_tpath(std::string dir, int dir_num, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + std::to_string(dir_num) + "/00000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 100)
    {
        back = dir + std::to_string(dir_num) + "/0000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 1000)
    {
        back = dir + std::to_string(dir_num) + "/000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 10000)
    {
        back = dir + std::to_string(dir_num) + "/00" + std::to_string(var) + tag;
        return back;
    }
}

void camera1()
{
    std::cout << "start c1" << std::endl;
    /*
    cv::VideoCapture cap0(cn1);
    cap0.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap0.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap0.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap0.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!cap0.isOpened())
    {
        std::cout << "not opend cam1" << std::endl;
        return;
    }
    //*/
    while (1)
    {
        clock_t begin1 = clock();
        // cap0 >> src0;
        src0 = cv::imread("images/test_img/left.JPG", 1);
        clock_t end1 = clock();
        std::cout << "get c1 time ";
        print_elapsed_time(begin1, end1);
        std::cout << std::endl;

        // cv::cvtColor(src0, src0, cv::COLOR_BGR2GRAY);
        m.lock();
        get_c1 = 1;
        m.unlock();
        std::cout << "get_c1 in cam1 " << get_c1 << std::endl;
        while (wait_c1 == 0)
        {
            /* code */
        }
        wait_c1 = 0;

        if (fin == 1)
            break;
    }
    return;
}

void camera2()
{
    std::cout << "start c2" << std::endl;
    /*** USBカメラの初期化(複数台USBカメラが存在する場合引数で選択) ***/
    /*
    cv::VideoCapture cap1(cn2);
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap1.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!cap1.isOpened())
    {
        std::cout << "not opened cam2" << std::endl;
        return;
    }
    */

    while (1)
    {
        clock_t begin2 = clock();
        // cap1 >> src1;
        src1 = cv::imread("images/test_img/right.JPG");
        clock_t end2 = clock();
        std::cout << "get c2 time ";
        print_elapsed_time(begin2, end2);
        std::cout << std::endl;

        // cv::cvtColor(src1, src1, cv::COLOR_BGR2GRAY);
        m.lock();
        get_c2 = 1;
        m.unlock();
        std::cout << "get_c1 in cam2 " << get_c2 << std::endl;
        while (wait_c2 == 0)
        {
            /* code */
        }
        wait_c2 = 0;

        if (fin == 1)
            break;
    }
    return;
}

void test()
{
    int count = 0;

    while (1)
    {
    std::cout << "start test" << std::endl;
        while (1)
        {
            //std::cout << count << std::endl;
            if (count == 100000)
            {
                std::cout << "get_c1, get_c1 " << get_c1 << " " << get_c2 << std::endl;
                count = 0;
            }
            if (get_c1 == 1 & get_c2 == 1)
            {
                break;
            }
            /* code */
            count++;
        }

        clock_t begin = clock();
        // absdiff(src0, src1, diff);
        cv::imshow("preview0", src0);
        cv::imshow("preview1", src1);
        // cv::imshow("diff", diff);
        clock_t end = clock();

        int key = cv::waitKey(0);
        if (key == 'q')
            break;

        print_elapsed_time(begin, end);
        m.lock();
        get_c1 = 0;
        get_c2 = 0;
        wait_c1++;
        wait_c2++;
        m.unlock();
        /*
        cv::imwrite(make_tpath(dir, cn1, count, tag), src0);
        cv::imwrite(make_tpath(dir, cn2, count, tag), src1);

        if (key == 'q')
            break;
        else if (key == 'c')
        {
            cv::imwrite(make_tpath(dir, cn1, count, tag), src0);
            cv::imwrite(make_tpath(dir, cn2, count, tag), src1);
            count++;
        }
        */
    }
    fin = 1;
}

int main()
{
    std::thread thread_c1(camera1);
    std::thread thread_c2(camera2);
    std::thread thread_test(test);

    thread_c1.join();
    thread_c2.join();
    thread_test.join();
    return 0;
}
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

// dir name
std::string dir = "images/Tsukuba0";
std::string tag = ".png";
int DB_dir_num = 0;
int Cam_dir_num = 2;

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

int main(){

        std::string numf = dir + std::to_string(Cam_dir_num) + "/num.txt";
    std::string line;
    int num = 0;
    std::ifstream ifs(numf);
    while (getline(ifs, line))
    {
        num = std::stoi(line);
    }
    std::cout << "start" << std::endl;
    std::cout << num << std::endl;
    for (int i = 2; i < num; i++)
    {
        clock_t begin = clock();
        std::string path = make_tpath(dir, 0, i, tag);
        std::string _path = make_tpath(dir, 0, i-1, tag);
        std::string _path_ = make_tpath(dir, 0, i-2, tag);
        std::cout << path << std::endl;
        cv::Mat img = cv::imread(path, 0);
        if (img.empty())
            return -1;
        cv::Mat _img = cv::imread(_path, 0);
        if (_img.empty())
            return -1;
        cv::Mat _img_ = cv::imread(_path_, 0);
        if (_img.empty())
            return -1;

        cv::Mat diff_image;
        cv::absdiff(img, _img, diff_image);
        cv::Mat diff_image2;
        cv::absdiff(_img, _img_, diff_image2);
        cv::Mat diff;
        cv::absdiff(diff_image, diff_image2, diff);
        cv::Mat back;
        cv::absdiff(img, diff, back);

        clock_t end = clock();
        print_elapsed_time(begin, end);

        cv::imshow("hog", img);
        cv::imshow("diff", diff);
        cv::imshow("back", back);
        // cv::imwrite("output.jpg", img);
        cv::waitKey(10);
    }

    return 0;
}
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

std::string dir = "images/Tsukuba0";
std::string tag = ".png";
int DB_dir_num = 0;
int Cam_dir_num = 2;

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

void print_elapsed_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %15.7f sec\n", elapsed);
}

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_LBP(const cv::Mat &src, cv::Mat &lbp)
{
    lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc;
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);

    // cv::imshow("first", src);
    // cv::imshow("third", lbp);
    for (int x = 1; x < padsrc.cols - 1; x++)
    {
        for (int y = 1; y < padsrc.rows - 1; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= padsrc.at<unsigned char>(y, x))
                        lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
                }
            }
        }
    }
    // cv::imshow("second", lbp);
}

void detective()
{
    cv::Mat lbp, edge, det;
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
    for (int i = 339; i < 340; i++)
    {
        clock_t begin = clock();
        std::string path = make_tpath(dir, Cam_dir_num, i, tag);
        std::cout << path << std::endl;
        cv::Mat img = cv::imread(path, 0);
        if (img.empty())
        {
            std::cout << "not found images" << std::endl;
            break;
        }

        cvt_LBP(img, lbp);

        cv::Canny(img, edge, 100, 200);

        // cv::findContours は第一引数を破壊的に利用するため imshow 用に別変数を用意しておきます。
        cv::Mat canny2 = edge.clone();

        // cv::Point の配列として、輪郭を計算します。
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(canny2, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::cout << contours.size() << std::endl;                  //=> 36
        std::cout << contours[contours.size() - 1][0] << std::endl; //=> [154, 10]

        // 輪郭を可視化してみます。分かりやすさのため、乱数を利用して色付けします。
        cv::Mat drawing = cv::Mat::zeros(canny2.size(), CV_8UC3);
        cv::RNG rng(12345);

        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            cv::drawContours(drawing, contours, (int)i, color);
        }

        det = drawing.clone();
        //cv::add(edge, lbp, det);
        clock_t end = clock();
        print_elapsed_time(begin, end);

        cv::imshow("hog", img);
        cv::imshow("lbp", lbp);
        cv::imshow("canny", edge);
        cv::imshow("result", det);
        // cv::imwrite("output.jpg", img);
        cv::waitKey(0);
    }
}

int main()
{

    detective();
    return 0;
}
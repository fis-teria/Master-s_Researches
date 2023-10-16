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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

std::string dir = "images/Tsukuba0";
std::string tag = ".png";
int DB_dir_num = 0;
int Cam_dir_num = 2;

const int NTSS_GRAY = 0;
const int NTSS_RGB = 1;

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
    cv::Mat padsrc, blur;
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    // cv::bilateralFilter(padsrc, padsrc, 2, 2*2, 2/2);
    cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
    // cv::medianBlur(padsrc, blur, 3);
    // padsrc = blur.clone();
    //  cv::imshow("first", src);
    //  cv::imshow("third", lbp);
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
        // cv::add(edge, lbp, det);
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

class readXml
{
public:
    cv::Mat camera_matrix, distcoeffs;

public:
    readXml(const std::string FILE_NAME)
    {
        cv::FileStorage fs(FILE_NAME, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
    }
};

struct BM
{
    int x;
    int y;
    int sam;
};

void NTSS(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int step)
{
    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int BM_size = 0;
    int sam = 0;
    int k = origin_x - step;
    // if (k < 0)
    //     k = 0;
    int l = origin_y - step;
    // if (l < 0)
    //     l = 0;

    int end_x = origin_x + step;
    if (end_x > src.cols)
        end_x = src.cols;
    int end_y = origin_y + step;
    if (end_y > src.rows)
        end_y = src.rows;
    // std::cout << "before error" << std::endl;
    // std::cout << x << " " << y << std::endl;

    // first step
    // step distance round search
    std::cout << "origin_x " << origin_x << " origin_y " << origin_y << std::endl;
    std::cout << "first step" << std::endl;
    std::cout << "step distance round search" << std::endl;
    for (int x = k; x <= end_x; x += step)
    {
        for (int y = l; y <= end_y; y += step)
        {
            if (x == origin_x && y == origin_y)
            {
                continue;
            }
            else
            {
                std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = x;
                match_Result[BM_size].y = y;
                rect = src.clone();
                cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                cv::imshow("dd", rect);
                const int key = cv::waitKey(10);

                std::cout << "start block matching" << std::endl;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                        {
                            // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << "sam " << sam << std::endl;
                sam = 0;
                BM_size++;
            }
        }
    }

    // origin round search
    std::cout << "origin round search" << std::endl;
    for (int x = origin_x - 1; x <= origin_x + 1; x++)
    {
        for (int y = origin_y - 1; y <= origin_y + 1; y++)
        {
            if (x >= 0 && y >= 0 && x <= src.cols && y <= src.rows)
            {
                std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = x;
                match_Result[BM_size].y = y;
                rect = src.clone();
                cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                cv::imshow("dd", rect);
                const int key = cv::waitKey(10);

                std::cout << "start block matching" << std::endl;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                        {
                            // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << "sam " << sam << std::endl;
                sam = 0;
                BM_size++;
            }
        }
    }
    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sam < beta.sam; });
    std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sam << " " << match_Result.size() << std::endl;

    int sstep = step / 2;
    int sx = match_Result[0].x;
    int sy = match_Result[0].y;
    int tx = origin_x;
    int ty = origin_y;
    match_Result.resize(0);
    BM_size = 0;

    // second steps and more
    if ((sx - tx) * (sx - tx) + (sy - ty) * (sy - ty) > 2)
    {
        std::cout << "far distance" << std::endl;
        for (int x = sx - sstep; x <= sx + sstep; x += sstep)
        {
            for (int y = sy - sstep; y <= sy + sstep; y += sstep)
            {
                if (x == origin_x && y == origin_y)
                {
                    continue;
                }
                else
                {
                    std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                    match_Result.resize(BM_size + 1);
                    match_Result[BM_size].x = x;
                    match_Result[BM_size].y = y;
                    rect = src.clone();
                    cv::rectangle(rect, cv::Point(x, y), cv::Point(x + block.cols, y + block.rows), cv::Scalar(255, 0, 0), 1);
                    cv::imshow("dd", rect);
                    const int key = cv::waitKey(10);

                    std::cout << "start block matching" << std::endl;
                    for (int i = 0; i < block.cols; i++)
                    {
                        for (int j = 0; j < block.rows; j++)
                        {
                            if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                            {
                                // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                                sam += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                            }
                        }
                    }
                    match_Result[BM_size].sam = sam;
                    std::cout << "sam " << sam << std::endl;
                    sam = 0;
                    BM_size++;
                }
            }
        }
        std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
                  { return alpha.sam < beta.sam; });
        std::cout << "origin point (" << sx << " " << sy << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sam << " " << match_Result.size() << std::endl;
    }
    else
    {
        std::cout << "near distance" << std::endl;
        /*
            A P O N M
            B 1 4 7 L
            C 2 5 8 K
            D 3 6 9 J
            E F G H I
        */
        if (abs(sx - origin_x) < 0)
        {
            if (abs(sy - origin_y) < 0)
            {
                std::cout << "match left up" << std::endl;
                // 1
                // A
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // P
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // B
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
            else if (abs(sy - origin_y) == 0)
            {
                std::cout << "match left" << std::endl;
                // 2
                // B
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // C
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // D
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
            else if (abs(sy - origin_y) > 0)
            {
                std::cout << "match left down" << std::endl;
                // 3
                // D
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // E
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // F
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
        }
        else if (abs(sx - origin_x) == 0)
        {
            if (abs(sy - origin_y) < 0)
            {
                std::cout << "match up" << std::endl;
                // 4
                // P
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // O
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // N
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
            else if (abs(sy - origin_y) == 0)
            {
                std::cout << "match second origin" << std::endl;
            }
            else if (abs(sy - origin_y) > 0)
            {
                std::cout << "match down" << std::endl;
                // 6
                // F
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx - 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // G
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // H
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
        }
        else if (abs(sx - origin_x) > 0)
        {
            if (abs(sy - origin_y) < 0)
            {
                std::cout << "match right up" << std::endl;
                // 7
                // N
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // M
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // L
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
            else if (abs(sy - origin_y) == 0)
            {
                std::cout << " match right" << std::endl;
                // 8
                // L
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy - 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // K
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // J
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
            else if (abs(sy - origin_y) > 0)
            {
                std::cout << "match right down" << std::endl;
                // 9
                // H
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // I
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy + 1;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
                // J
                match_Result.resize(BM_size + 1);
                match_Result[BM_size].x = sx + 1;
                match_Result[BM_size].y = sy;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (sx + i >= 0 && sy + j >= 0 && sy + j < src.rows && sx + i < src.cols)
                        {
                            // std::cout << sx + i << " " << sy + j << " " << src.at<unsigned char>(sy + j, sx + i) << " " << block.at<unsigned char>(j, i) << std::endl;
                            sam += abs(src.at<unsigned char>(sy + j, sx + i) - block.at<unsigned char>(j, i));
                        }
                    }
                }
                match_Result[BM_size].sam = sam;
                std::cout << sam << std::endl;
                sam = 0;
                BM_size++;
            }
        }
        std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
                  { return alpha.sam < beta.sam; });
        std::cout << "origin point (" << sx << " " << sy << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sam << " " << match_Result.size() << std::endl;
    }

    std::cout << "origin point (" << origin_x << " " << origin_y << ") -> first match(" << sx << " " << sy << ") -> second match(" << match_Result[0].x << " " << match_Result[0].y << ")" << std::endl;
    return;
}

void block_Matching(const cv::Mat &block, const cv::Mat &src, int block_size, int mode)
{
    int b_size = block_size * 5;
    if (b_size % 2 != 1)
        b_size++;

    for (int x = b_size / 2; x < block.cols; x += b_size)
    {
        for (int y = b_size / 2; y < block.rows; y += b_size)
        {
            if (mode == 0)
                if (y + b_size / 2 < block.rows && x + b_size / 2 < block.cols)
                    NTSS(block(cv::Range(y - b_size / 2, y + b_size / 2), cv::Range(x - b_size / 2, x + b_size / 2)), src, x, y, 4);
        }
    }
}

void xmlRead()
{
    readXml xml00 = readXml("camera/out_camera_data00.xml");
    readXml xml02 = readXml("camera/out_camera_data02.xml");
    /*
    for (int x = 0; x < xml00.camera_matrix.cols; x++)
    {
        for (int y = 0; y < xml00.camera_matrix.rows; y++)
        {
            std::cout << xml00.camera_matrix.at<double>(y, x) << " ";
        }
        std::cout << std::endl;
    }
    */
    cv::VideoCapture cap(0); // デバイスのオープン
                             // cap.open(0);//こっちでも良い．
                             // capの画像の解像度を変える部分 word CaptureChange
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::VideoCapture cap2(2); // デバイスのオープン
                              // cap.open(0);//こっちでも良い．
                              // capの画像の解像度を変える部分 word CaptureChange
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap2.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::Mat frame; // 取得したフレーム
    cv::Mat distort;
    cv::Mat matx, maty;

    cv::Mat frame2; // 取得したフレーム
    cv::Mat distort2;

    cv::Mat matx2, maty2;
    cv::initUndistortRectifyMap(xml00.camera_matrix, xml00.distcoeffs, cv::Mat(), xml00.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx, maty);
    cv::initUndistortRectifyMap(xml02.camera_matrix, xml02.distcoeffs, cv::Mat(), xml02.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx2, maty2);

    cv::Mat f1g, f2g;
    while (1) // 無限ループ
    {
        cap >> frame;
        cap2 >> frame2;
        // cv::imshow("win", frame);   // 画像を表示．
        // cv::imshow("win2", frame2); // 画像を表示．
        cv::cvtColor(frame, f1g, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, f2g, cv::COLOR_BGR2GRAY);
        // f1g.convertTo(f1g, CV_8U);
        // std::cout << f1g << std::endl;
        //  cvt_LBP(frame, distort);
        //  cvt_LBP(frame2, distort2);
        //   clock_t begin = clock();
        //    cv::undistort(frame, distort, xml00.camera_matrix, xml00.distcoeffs);
        //    cv::remap(distort, distort, matx, maty, cv::INTER_LANCZOS4);
        //    cv::remap(distort2, distort2, matx2, maty2, cv::INTER_LANCZOS4);
        cv::remap(frame, distort, matx, maty, cv::INTER_LINEAR);
        cv::remap(frame2, distort2, matx2, maty2, cv::INTER_LINEAR);
        // clock_t end = clock();
        // print_elapsed_time(begin, end);

        block_Matching(f1g, f2g, 5, NTSS_GRAY);
        // cv::imshow("a", distort);
        // cv::imshow("b", distort2);

        const int key = cv::waitKey(10);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            break; // whileループから抜ける．
        }
    }
    cv::destroyAllWindows();

    return;
}

void subMat()
{
    readXml xml00 = readXml("camera/out_camera_data00.xml");
    readXml xml02 = readXml("camera/out_camera_data02.xml");
    /*
    for (int x = 0; x < xml00.camera_matrix.cols; x++)
    {
        for (int y = 0; y < xml00.camera_matrix.rows; y++)
        {
            std::cout << xml00.camera_matrix.at<double>(y, x) << " ";
        }
        std::cout << std::endl;
    }
    */
    cv::VideoCapture cap(0); // デバイスのオープン
                             // cap.open(0);//こっちでも良い．
                             // capの画像の解像度を変える部分 word CaptureChange
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::VideoCapture cap2(2); // デバイスのオープン
                              // cap.open(0);//こっちでも良い．
                              // capの画像の解像度を変える部分 word CaptureChange
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap2.isOpened()) // カメラデバイスが正常にオープンしたか確認．
    {
        // 読み込みに失敗したときの処理
        return;
    }

    cv::Mat frame; // 取得したフレーム
    cv::Mat distort;
    cv::Mat matx, maty;

    cv::Mat frame2; // 取得したフレーム
    cv::Mat distort2;

    cv::Mat matx2, maty2;
    cv::initUndistortRectifyMap(xml00.camera_matrix, xml00.distcoeffs, cv::Mat(), xml00.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx, maty);
    cv::initUndistortRectifyMap(xml02.camera_matrix, xml02.distcoeffs, cv::Mat(), xml02.camera_matrix, cv::Size(1280, 720), CV_32FC1, matx2, maty2);

    cv::Mat f1g, f2g;
    // while (1) // 無限ループ
    //{
    cap >> frame;
    cap2 >> frame2;
    // cv::imshow("win", frame);   // 画像を表示．
    // cv::imshow("win2", frame2); // 画像を表示．
    // f1g.convertTo(f1g, CV_8U);
    // std::cout << f1g << std::endl;
    //  cvt_LBP(frame, distort);
    //  cvt_LBP(frame2, distort2);
    //   clock_t begin = clock();
    //    cv::undistort(frame, distort, xml00.camera_matrix, xml00.distcoeffs);
    //    cv::remap(distort, distort, matx, maty, cv::INTER_LANCZOS4);
    //    cv::remap(distort2, distort2, matx2, maty2, cv::INTER_LANCZOS4);
    cv::remap(frame, distort, matx, maty, cv::INTER_LINEAR);
    cv::remap(frame2, distort2, matx2, maty2, cv::INTER_LINEAR);
    cv::cvtColor(distort, f1g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(distort2, f2g, cv::COLOR_BGR2GRAY);
    // clock_t end = clock();
    // print_elapsed_time(begin, end);

    std::cout << (int)f1g.at<unsigned char>(24, 24) << " " << (int)f2g.at<unsigned char>(24, 24) << " " << abs(f1g.at<unsigned char>(24, 24) - f2g.at<unsigned char>(24, 24)) << std::endl;
    cv::imshow("a", f1g);
    cv::imshow("b", f2g);

    const int key = cv::waitKey(0);
    if (key == 'q' /*113*/) // qボタンが押されたとき
    {
        // break; // whileループから抜ける．
    }
    //}
    cv::destroyAllWindows();

    return;
}

int main()
{

    // detective();
    xmlRead();
    // subMat();
    return 0;
}
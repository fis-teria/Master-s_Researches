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

#if _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_contrib248d.lib")
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_nonfree248d.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_contrib248.lib")
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_nonfree248.lib")
#endif

std::string dir = "images/Tsukuba0";
std::string tag = ".jpg";

int DB_dir_num = 0;
int Cam_dir_num = 2;

/*
    Width   Height
    4032    2268
    1280    720
    1024    576
    896     504
    768     432
    640     360
    512     288
    384     216
*/

const int WIDTH = 4032;  // 1280 896
const int HEIGHT = 2268; // 720 504
int D_MAG = 15;          // H = 距離ｘ倍率(H<150)  ex) 最長距離を10mにしたければ倍率を15にすればよい
static std::mutex m;

int FOCUS = 24;                                                // 焦点 mm
float IS_WIDTH = 4.8;                                          // 撮像素子の横 1/3レンズなら4.8mm
float IS_HEIGHT = 3.6;                                         // 撮像素子の縦 1/3レンズなら3.6mm
float PXL_WIDTH = (IS_WIDTH / WIDTH);                          // 1pixelあたりの横の長さ mm
float PXL_HEIGHT = (IS_HEIGHT / HEIGHT);                       // 1pixelあたりの縦の長さ mm
int CAM_DIS = 10;                                              // カメラ間の距離 10cm
double D_CALI = (FOCUS * (CAM_DIS * 10)) / (PXL_WIDTH * 1000); // 距離を求めるのに必要な定数項 mで換算 式(焦点(mm) * カメラ間距離(cm))/(1pixel長(mm)*画像内の距離)

const int NTSS_GRAY = 0;
const int NTSS_RGB = 1;

const int WIN_SIZE = 9;

const int L2R = -1;
const int R2L = 1;

const int SEARCH_RANGE = 128;

const int COLOR_MODE = NTSS_GRAY;

const int DO_ILBP = 1;

std::vector<int> UNIFORMED_LUT(256, 0);

/*
    test data
    images/20231231/left/004567.jpg
    images/20240220/left/000036.jpg
    images/test_img/left.JPG
*/
const std::string LEFT_IMG = "images/20231231/left/004567.jpg";
const std::string RIGHT_IMG = "images/20231231/right/004567.jpg";

void print_elapsed_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %15.15f sec\n", elapsed);
}

// ブロックマッチングに使うかもしれない構造体
class BM
{
public:
    int x;
    int y;
    int sum;

public:
    BM()
    {
    }

public:
    BM(const int origin_x, const int origin_y, const int s)
    {
        x = origin_x;
        y = origin_y;
        sum = s;
    }

public:
    BM(const BM &BM)
    {
        x = BM.x;
        y = BM.y;
        sum = BM.sum;
    }

public:
    void get_ELEMENTS()
    {
        std::cout << "x, y, sum "
                  << " " << x << " " << y << " " << sum << std::endl;
    }
};

// ブロックマッチングの結果を保持するマルチスレッド用のクラス
class BLOCK_MATCHING
{
public:
    int origin_x, origin_y;
    double depth;

public:
    BLOCK_MATCHING(const int x, const int y, const double d)
    {
        origin_x = x;
        origin_y = y;
        depth = d;
    }

    BLOCK_MATCHING(const BLOCK_MATCHING &B_M)
    {
        origin_x = B_M.origin_x;
        origin_y = B_M.origin_y;
        depth = B_M.depth;
    }

public:
    void get_ELEMENTS()
    {
        std::cout << "x, y, depth "
                  << " " << origin_x << " " << origin_y << " " << depth << std::endl;
    }
};

struct CORRES
{
    int x;
    int y;
    int SAD;
};

struct POSITION
{
    int x;
    int y;
};

struct RESULT_SIM_BM
{
    int x;
    int y;
    int SAD;
    double depth;
};

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_LBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc;
    // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

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
    dst = lbp.clone();
    // cv::imshow("second", lbp);
}

void cvt_ILBP(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
    lbp = cv::Scalar::all(0);
    cv::Mat padsrc;
    // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
    copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

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
            ave /= 9;
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
                    //std::cout << (int)lbp.at<unsigned char>(y - 1, x - 1) << std::endl;
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
            lut[count] = 80 + 5 * lut[count];
        }
        count++;
    }
    std::cout << "make_LUT end" << std::endl;
}
// シンプルなブロックマッチング グレースケール
CORRES sim_BM(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int mode, int LorR)
{
    CORRES result = {0, 0, -1};
    int debug = 0;
    if (debug == 1)
        std::cout << "start sim_BM" << std::endl;

    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int dist;
    double depth;
    int BM_size = 0;
    int sum = 0;
    int y = origin_y - 1;

    int start_x = 0;
    int search_range = 0;
    int end_lange = 0;

    int count = 0;

    /*
    if (DO_ILBP == 1)
    {
        for (int x = 0; x < block.cols; x++)
        {
            for (int y = 0; y < block.rows; y++)
            {
                if (block.at<unsigned char>(y, x) == 0)
                    count++;
            }
        }
    }
    if(debug == 1){
        std::cout << count << std::endl;
    }
    if (count >= 4)
        return result;
    //*/

    if (LorR == R2L)
    {
        start_x = origin_x - 0;
        search_range = WIDTH / 2;
    }
    else if (LorR == L2R)
    {
        start_x = origin_x - WIDTH / 2;
        search_range = 0;
    }
    end_lange = origin_x + search_range;

    if (start_x < 0)
        start_x = 0;
    if (end_lange > src.cols)
        end_lange = src.cols;

    for (int x = start_x; x < end_lange; x += block.cols)
    {
        if (x < src.cols)
        {
            // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
            // match_Result.resize(BM_size + 1);
            // match_Result[BM_size].x = x;
            // match_Result[BM_size].y = y;

            // std::cout << "start block matching" << std::endl;
            for (int i = 0; i < block.cols; i++)
            {
                for (int j = 1; j < block.rows; j++)
                {
                    if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                    {
                        // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                        if (mode == NTSS_GRAY)
                        {
                            sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                        }
                        else if (mode == NTSS_RGB)
                        {
                            sum += abs(src.at<cv::Vec3b>(y + j, x + i)[0] - block.at<cv::Vec3b>(j, i)[0]) + abs(src.at<cv::Vec3b>(y + j, x + i)[1] - block.at<cv::Vec3b>(j, i)[1]) + abs(src.at<cv::Vec3b>(y + j, x + i)[2] - block.at<cv::Vec3b>(j, i)[2]);
                        }
                    }
                }
            }
            // match_Result[BM_size].sum = sum;
            match_Result.push_back(BM(x, y, sum));
            // std::cout << "sum " << sum << std::endl;
            sum = 0;
            BM_size++;
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });

    if (debug == 1)
    {
        std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size();

        dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) + (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
        // if (debug == 1)
        std::cout << " distance = " << dist;

        depth = D_CALI / dist;
        // if (debug == 1)
        std::cout << " depth = " << depth << std::endl;
    }
    if (match_Result.empty() != 1)
    {
        result.x = match_Result[0].x;
        result.y = match_Result[0].y;
        result.SAD = match_Result[0].sum;
        // result.depth = depth;
    }
    return result;
}

RESULT_SIM_BM sim_rBM(const cv::Mat &block, const cv::Mat &src, int origin_x, int origin_y, int mode, int LorR)
{
    RESULT_SIM_BM result = {0, 0, -1, 0};
    int debug = 0;
    if (debug == 1)
        std::cout << "start sim_BM" << std::endl;

    cv::Mat rect = src.clone();
    std::vector<BM> match_Result;
    int dist;
    double depth;
    int BM_size = 0;
    int sum = 0;
    // int y = origin_y;

    int start_x = 0;
    int search_range = 0;
    int end_lange = 0;

    int count = 0;

    /*
    if (DO_ILBP == 1)
    {
        for (int x = 0; x < block.cols; x++)
        {
            for (int y = 0; y < block.rows; y++)
            {
                if (block.at<unsigned char>(y, x) == 0)
                    count++;
            }
        }
    }
    if(debug == 1){
        std::cout << count << std::endl;
    }
    if (count >= 4)
        return result;
    //*/

    if (LorR == R2L)
    {
        start_x = origin_x - 0;
        search_range = WIDTH / 2;
    }
    else if (LorR == L2R)
    {
        start_x = origin_x - WIDTH / 2;
        search_range = 0;
    }
    end_lange = origin_x + search_range;

    if (start_x < 0)
        start_x = 0;
    if (end_lange > src.cols)
        end_lange = src.cols;

    for (int x = start_x; x < end_lange; x += 1)
    {
        for (int y = origin_y; y < origin_y + (0 * block.rows) + 1; y += block.rows)
        {
            if (x < src.cols)
            {
                // std::cout << "matching search point (" << x << " " << y << ")" << std::endl;
                //  match_Result.resize(BM_size + 1);
                //  match_Result[BM_size].x = x;
                //  match_Result[BM_size].y = y;

                // std::cout << "start block matching" << std::endl;
                for (int i = 0; i < block.cols; i++)
                {
                    for (int j = 0; j < block.rows; j++)
                    {
                        if (x + i >= 0 && y + j >= 0 && y + j < src.rows && x + i < src.cols)
                        {
                            // std::cout << x + i << " " << y + j << " " << (int)src.at<unsigned char>(y + j, x + i) << " " << (int)block.at<unsigned char>(j, i) << std::endl;
                            if (mode == NTSS_GRAY)
                            {
                                sum += abs(src.at<unsigned char>(y + j, x + i) - block.at<unsigned char>(j, i));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                sum += abs(src.at<cv::Vec3b>(y + j, x + i)[0] - block.at<cv::Vec3b>(j, i)[0]) + abs(src.at<cv::Vec3b>(y + j, x + i)[1] - block.at<cv::Vec3b>(j, i)[1]) + abs(src.at<cv::Vec3b>(y + j, x + i)[2] - block.at<cv::Vec3b>(j, i)[2]);
                            }
                        }
                    }
                }
                // match_Result[BM_size].sum = sum;
                match_Result.push_back(BM(x, y, sum));
                // std::cout << "sum " << sum << std::endl;
                sum = 0;
                BM_size++;
            }
        }
    }

    std::sort(match_Result.begin(), match_Result.end(), [](const BM &alpha, const BM &beta)
              { return alpha.sum < beta.sum; });

    if (debug == 1)
    {
        std::cout << "origin point (" << origin_x << " " << origin_y << ") matching point (" << match_Result[0].x << " " << match_Result[0].y << ") " << match_Result[0].sum << " " << match_Result.size();

        dist = sqrt(abs((origin_x - match_Result[0].x) * (origin_x - match_Result[0].x) + (origin_y - match_Result[0].y) * (origin_y - match_Result[0].x)));
        // if (debug == 1)
        std::cout << " distance = " << dist;

        depth = D_CALI / dist;
        // if (debug == 1)
        std::cout << " depth = " << depth << std::endl;
    }

    if (match_Result.empty() != 1)
    {
        result.x = match_Result[0].x;
        result.y = match_Result[0].y;
        result.SAD = match_Result[0].sum;
        // result.depth = depth;
    }

    return result;
}

// 深度マップから元画像ないの特定の距離のものを抽出する。
void get_depth(const cv::Mat &src, cv::Mat &dst, const cv::Mat &origin)
{
    cv::Mat copy = cv::Mat(src.rows, src.cols, CV_8UC1);
    copy = cv::Scalar::all(0);
    int H;

    int left = 0;
    int centor = 0;
    int right = 0;

    for (int y = 0; y < copy.rows; y++)
    {
        for (int x = 0; x < copy.cols; x++)
        {
            H = src.at<cv::Vec3b>(y, x)[0];
            if ((H / D_MAG) < 10)
            {
                copy.at<unsigned char>(y, x) = 0;
                if (x < copy.cols / 3)
                {
                    left++;
                }
                else if (x > copy.cols / 3 && x < copy.cols * 2 / 3)
                {
                    centor++;
                }
                else if (x > copy.cols * 2 / 3)
                {
                    right++;
                }
            }
            else
            {
                copy.at<unsigned char>(y, x) = origin.at<unsigned char>(y, x);
            }
        }
    }

    /*
    //左が最小
    if (left < centor && left < right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = copy.cols / 3; x < copy.cols; x++)
            {
                copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    //真ん中が最小
    else if (centor < left && centor < right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = 0; x < copy.cols; x++)
            {
                if (x < copy.cols / 3 || x > copy.cols * 2 / 3)
                    copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    //右が最小
    else if (right < left && centor < right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = 2 * copy.cols / 3; x >= 0; x--)
            {
                copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    //*/
    // 左が最大
    if (left > centor && left > right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = 0; x < copy.cols / 3; x++)
            {
                copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    // 真ん中が最大
    else if (centor > left && centor > right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = 0; x < copy.cols; x++)
            {
                if (x > copy.cols / 3 && x > copy.cols * 2 / 3)
                    copy.at<unsigned char>(y, x) = 0;
            }
        }
    }
    // 右が最大
    else if (right > left && centor < right)
    {
        for (int y = 0; y < copy.rows; y++)
        {
            for (int x = 2 * copy.cols / 3; x < copy.cols; x++)
            {
                copy.at<unsigned char>(y, x) = 0;
            }
        }
    }

    dst = copy.clone();
}

// ブロックマッチングの準備をする関数
void block_Matching(cv::Mat &block, const cv::Mat &src, std::vector<std::vector<CORRES>> &conv_map, int block_size, int mode, int LorR)
{
    int debug = 0;
    int debug_img = 0;

    std::vector<int> time;
    std::vector<BLOCK_MATCHING> vec_bm;
    cv::Mat depth_map = cv::Mat(block.rows, block.cols, CV_8UC3);
    depth_map = cv::Scalar::all(0);
    cv::Mat depth_map_HSV = depth_map.clone();
    int b_size = block_size;
    if (b_size % 2 != 1)
        b_size++;

    int depth_H = 0;
    int x_count = 0;
    int y_count = 0;

    RESULT_SIM_BM result = {0, 0, 0, 0};

    clock_t begin = clock();

    // シングルスレッド
    ///*
    std::cout << "single thread" << std::endl;

    int i = 0;
    int j = 0;
    OMP_PARALLEL_FOR
#pragma omp private(depth_H, result, block, src, conv_map, x, y, i, j)
    for (int x = 0; x < conv_map.size(); x++)
    {
        for (int y = 0; y < conv_map[x].size(); y++)
        {
            if (y + b_size / 2 < block.rows && x + b_size / 2 < block.cols)
            {
                conv_map[x][y] = sim_BM(block(cv::Range(b_size * (y), b_size * (y + 1) - 1), cv::Range(b_size * (x), b_size * (x + 1) - 1)), src, b_size * (x), b_size * (y), COLOR_MODE, LorR);
                // std::cout << "now debug" << x << " " << y << " " << conv_map[x][y].x << std::endl;
                //  vec_bm.push_back(BLOCK_MATCHING(x, y, result.depth));
                //  conv_map[x][y].x = result.x;
                //  conv_map[x][y].y = result.y;
                //  conv_map[x][y].SAD = result.SAD;
            }
        }
    }

    // std::cout << "end loop" << std::endl;
    //*/

    clock_t end = clock();
    print_elapsed_time(begin, end);

    if (debug_img == 1)
    {
        for (int i = 0; i < vec_bm.size(); i++)
        {
            depth_H = vec_bm[i].depth * D_MAG;
            /*
            if (depth_H > 150)
                depth_H = 150;
            if (depth_H < 0)
                depth_H = 0;
            //*/
            if (debug == 1)
                vec_bm[i].get_ELEMENTS();
            // std::cout << vec_bm[i].depth << std::endl;
            // cv::rectangle(depth_map, cv::Point(vec_bm[i].origin_x - b_size / 2, vec_bm[i].origin_y - b_size / 2), cv::Point(vec_bm[i].origin_x + b_size / 2, vec_bm[i].origin_y + b_size / 2), cv::Scalar(depth_H, 255, 255), cv::FILLED);
            ///*
            for (int x = vec_bm[i].origin_x - b_size / 2; x <= vec_bm[i].origin_x + b_size / 2; x++)
            {
                for (int y = vec_bm[i].origin_y - b_size / 2; y <= vec_bm[i].origin_y + b_size / 2; y++)
                {
                    depth_map.at<cv::Vec3b>(y, x)[0] = depth_H;
                    depth_map.at<cv::Vec3b>(y, x)[1] = 255;
                    depth_map.at<cv::Vec3b>(y, x)[2] = 255;
                    // std::cout << depth_map.at<cv::Vec3b>(y,x) << std::endl;
                }
            }
            //*/
        }
        // std::cout << "synchronous" << std::endl;
        cv::cvtColor(depth_map, depth_map_HSV, cv::COLOR_HSV2BGR);
        // cv::Mat nearline;
        // get_depth(depth_map_HSV, nearline, block);

        if (LorR == R2L)
        {
            cv::imshow("depth R2L", depth_map_HSV);
            // cv::imshow("depth R2L nearline", nearline);
        }
        else if (LorR == L2R)
        {
            cv::imshow("depth L2R", depth_map_HSV);
            // cv::imshow("depth L2R nearline", nearline);
        }
        //  std::cout << "block matching time = ";
    }
}

void block_Matching(cv::Mat &block, const cv::Mat &src, std::vector<std::vector<RESULT_SIM_BM>> &conv_map, int block_size, int mode, int LorR)
{
    int debug = 0;
    int debug_img = 0;

    std::vector<int> time;
    std::vector<BLOCK_MATCHING> vec_bm;
    cv::Mat depth_map = cv::Mat(block.rows, block.cols, CV_8UC3);
    depth_map = cv::Scalar::all(0);
    cv::Mat depth_map_HSV = depth_map.clone();
    int b_size = block_size;
    if (b_size % 2 != 1)
        b_size++;

    int depth_H = 0;
    int x_count = 0;
    int y_count = 0;

    RESULT_SIM_BM result = {0, 0, 0, 0};

    clock_t begin = clock();

    // シングルスレッド
    ///*
    std::cout << "single thread" << std::endl;

    int i = 0;
    int j = 0;

    OMP_PARALLEL_FOR
#pragma omp private(depth_H, result, block, src, conv_map, x, y, i, j)
    for (int x = 0; x < conv_map.size(); x++)
    {
        for (int y = 0; y < conv_map[x].size(); y++)
        {
            if (y + b_size / 2 < block.rows && x + b_size / 2 < block.cols)
            {
                /*
                std::cout << "block ( y = " << b_size * (y) << " -> " << b_size * (y + 1) - 1 << ", "
                          << "x = " << b_size * (x) << " -> " << b_size * (x + 1) - 1 << ")" << std::endl;
                //*/
                conv_map[x][y] = sim_rBM(block(cv::Range(b_size * (y), b_size * (y + 1) - 1), cv::Range(b_size * (x), b_size * (x + 1) - 1)), src, b_size * (x), b_size * (y), COLOR_MODE, LorR);
                // std::cout << "now debug" << x << " " << y << " " << conv_map[x][y].x << std::endl;
                //  vec_bm.push_back(BLOCK_MATCHING(x, y, result.depth));
                //  conv_map[x][y].x = result.x;
                //  conv_map[x][y].y = result.y;
                //  conv_map[x][y].SAD = result.SAD;
            }
        }
    }
    clock_t end = clock();
    print_elapsed_time(begin, end);
}

void opticalflow_Initialize(cv::Mat &block, cv::Mat &src, std::vector<std::vector<CORRES>> &conv_map, int block_size, std::vector<std::vector<POSITION>> &random_mask, int mode)
{
    int match_x = 0;
    int match_y = 0;
    int rand_uni = 0;

    int i = 0;
    int j = 0;

    int debug = 0;

    std::cout << "start Initialization" << std::endl;

    // conv_map_cols = 0;
    std::string location_path = "logs/patch_match/initilize.txt";
    std::ofstream outputfile(location_path);
    for (int x = block_size / 2; x < block.cols; x += block_size)
    {
        // conv_map_rows = 0;
        j = 0;
        for (int y = block_size / 2; y < block.rows; y += block_size)
        {
            // std::cout << "a" << std::endl;
            // std::cout << "b" << std::endl;

            if (debug == 1)
            {
                outputfile << "(" << x << " " << y << ") " << rand_uni << "      ";
                outputfile << "     ->      ";
            }
            match_x = 1 + 3 * random_mask[i][j].x;
            match_y = 1 + 3 * random_mask[i][j].y;
            // match_y = y;

            if (debug == 1)
            {
                outputfile << match_x << " " << match_y;
            }

            for (int bx = -1; bx <= 1; bx++)
            {
                for (int by = -1; by <= 1; by++)
                {
                    if (mode == NTSS_GRAY)
                    {
                        conv_map[i][j].SAD += abs(block.at<unsigned char>(y + by, x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                    }
                    else if (mode == NTSS_RGB)
                    {
                        conv_map[i][j].SAD += abs(block.at<cv::Vec3b>(y + by, x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                        conv_map[i][j].SAD += abs(block.at<cv::Vec3b>(y + by, x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                        conv_map[i][j].SAD += abs(block.at<cv::Vec3b>(y + by, x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                    }
                }
            }
            conv_map[i][j].x = match_x;
            conv_map[i][j].y = match_y;

            if (debug == 1)
            {
                outputfile << "    ->     ";
                outputfile << conv_map[i][j].x << " " << conv_map[i][j].y << " " << conv_map[i][j].SAD;
                outputfile << std::endl;
            }
            // conv_map_rows++;
            j++;
        }
        i++;
        // conv_map_cols++;
    }
}

void opticalflow_Random_Search(cv::Mat &block, cv::Mat &src, std::vector<std::vector<CORRES>> &conv_map, int conv_map_cols, int conv_map_rows, std::vector<std::vector<int>> &random_mask, int range, int mode)
{
    // search
    int match_x = 0;
    int match_y = 0;
    int rand_uni = 0;

    std::cout << "start Random Search" << std::endl;
    int search_range = range;
    search_range++;
    // conv_map_cols = 0;

    for (int x = 0; x < conv_map_cols; x++)
    {
        // conv_map_rows = 0;
        for (int y = 0; y < conv_map_rows; y++)
        {
            // std::cout << "a" << std::endl;
            rand_uni = random_mask[x][y];
            // std::cout << "b" << std::endl;
            // std::cout << "(" << x << " " << y << ") " << rand_uni << "      ";
            // std::cout << conv_map[x][y].x << " " << conv_map[x][y].y << " " << conv_map[x][y].SAD << "     ->      ";

            if (conv_map[x][y].x - 3 * search_range / 2 < 0)
            {
                // std::cout << " x1 ";
                if (conv_map[x][y].y - 3 * search_range / 2 < 0) // yが負の時
                {
                    // std::cout << " y1 ";
                    match_x = 1 + 3 * rand_uni / search_range;
                    match_y = 1 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else if (conv_map[x][y].y + 3 * search_range / 2 > block.rows) // yが縦を超えるとき
                {
                    // std::cout << " y2 ";
                    match_x = 1 + 3 * rand_uni / search_range;
                    match_y = (block.rows - 2 - 3 * search_range) + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else
                {
                    // std::cout << " y3 ";
                    match_x = 1 + 3 * rand_uni / search_range;
                    match_y = conv_map[x][y].y - 3 * search_range / 2 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
            }
            else if (conv_map[x][y].x + 3 * search_range / 2 > block.cols)
            {
                // std::cout << " x2 ";
                if (conv_map[x][y].y - 3 * search_range / 2 < 0) // yが負の時
                {
                    // std::cout << " y4 ";
                    match_x = (block.cols - 2 - 3 * search_range / 2) + 3 * rand_uni / search_range;
                    match_y = 1 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else if (conv_map[x][y].y + 3 * search_range / 2 > block.rows) // yが縦を超えるとき
                {
                    // std::cout << " y5 ";
                    match_x = (block.cols - 2 - 3 * search_range / 2) + 3 * rand_uni / search_range;
                    match_y = (block.rows - 2 - 3 * search_range / 2) + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else
                {
                    // std::cout << " y6 ";
                    match_x = (block.cols - 2 - 3 * search_range / 2) + 3 * rand_uni / search_range;
                    match_y = conv_map[x][y].y - 3 * search_range / 2 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
            }
            else
            {
                // std::cout << " x3 ";
                if (conv_map[x][y].y - 3 * search_range / 2 < 0) // yが負の時
                {
                    // std::cout << " y7 ";
                    match_x = conv_map[x][y].x - 3 * search_range / 2 + 3 * rand_uni / search_range;
                    match_y = 1 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else if (conv_map[x][y].y + 3 * search_range / 2 > block.rows) // yが縦を超えるとき
                {
                    // std::cout << " y8 ";
                    match_x = conv_map[x][y].x - 3 * search_range / 2 + 3 * rand_uni / search_range;
                    match_y = (block.rows - 2 - 3 * search_range / 2) + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
                else
                {
                    // std::cout << " y9 ";
                    match_x = conv_map[x][y].x - 3 * search_range / 2 + 3 * rand_uni / search_range;
                    match_y = conv_map[x][y].y - 3 * search_range / 2 + 3 * rand_uni % search_range;
                    conv_map[x][y].SAD = 0;
                    for (int bx = -1; bx <= 1; bx++)
                    {
                        for (int by = -1; by <= 1; by++)
                        {
                            if (mode == NTSS_GRAY)
                            {
                                conv_map[x][y].SAD += abs(block.at<unsigned char>(conv_map[x][y].y + by, conv_map[x][y].x + bx) - src.at<unsigned char>(match_y + by, match_x + bx));
                            }
                            else if (mode == NTSS_RGB)
                            {
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[0] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[0]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[1] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[1]);
                                conv_map[x][y].SAD += abs(block.at<cv::Vec3b>(conv_map[x][y].y + by, conv_map[x][y].x + bx)[2] - src.at<cv::Vec3b>(match_y + by, match_x + bx)[2]);
                            }
                        }
                    }
                    conv_map[x][y].x = match_x;
                    conv_map[x][y].y = match_y;
                }
            }
            // std::cout << conv_map[x][y].x << " " << conv_map[x][y].y << " " << conv_map[x][y].SAD;
            // std::cout << std::endl;

            // conv_map_rows++;
        }
        // conv_map_cols++;
    }
}

void opticalflow_Propagation(std::vector<std::vector<CORRES>> &conv_map, const cv::Mat &block, int conv_map_cols, int conv_map_rows, int r)
{
    std::cout << "start Propagation" << std::endl;
    for (int i = 0; i < conv_map_cols; i++)
    {
        for (int j = 0; j < conv_map_rows; j++)
        {
            if (r % 2 == 1)
            {
                if (i - 1 > 0)
                {
                    if (conv_map[i - 1][j].SAD < conv_map[i][j].SAD)
                    {
                        if (conv_map[i - 1][j].x + 3 < block.cols)
                        {
                            conv_map[i][j].x = conv_map[i - 1][j].x + 3;
                        }
                        else
                        {
                            conv_map[i][j].x = conv_map[i - 1][j].x;
                        }
                        conv_map[i][j].y = conv_map[i - 1][j].y;
                        conv_map[i][j].SAD = conv_map[i - 1][j].SAD;
                    }
                }
                if (j - 1 > 0)
                {
                    if (conv_map[i][j - 1].SAD < conv_map[i][j].SAD)
                    {
                        conv_map[i][j].x = conv_map[i][j - 1].x;
                        if (conv_map[i][j - 1].y + 3 < block.rows)
                        {
                            conv_map[i][j].y = conv_map[i][j - 1].y + 3;
                        }
                        else
                        {
                            conv_map[i][j].y = conv_map[i][j - 1].y;
                        }
                        conv_map[i][j].SAD = conv_map[i][j - 1].SAD;
                    }
                }
            }
            else
            {
                if (i + 1 < conv_map_cols)
                {
                    if (conv_map[i + 1][j].SAD < conv_map[i][j].SAD)
                    {
                        if (conv_map[i + 1][j].x - 3 > 0)
                        {
                            conv_map[i][j].x = conv_map[i + 1][j].x - 3;
                        }
                        else
                        {
                            conv_map[i][j].x = conv_map[i + 1][j].x;
                        }
                        conv_map[i][j].y = conv_map[i + 1][j].y;
                        conv_map[i][j].SAD = conv_map[i + 1][j].SAD;
                    }
                }
                if (j + 1 < conv_map_rows)
                {
                    if (conv_map[i][j + 1].SAD < conv_map[i][j].SAD)
                    {
                        conv_map[i][j].x = conv_map[i][j + 1].x;
                        if (conv_map[i][j + 1].y - 3 > 0)
                        {
                            conv_map[i][j].y = conv_map[i][j + 1].y - 3;
                        }
                        else
                        {
                            conv_map[i][j].y = conv_map[i][j + 1].y;
                        }
                        conv_map[i][j].SAD = conv_map[i][j + 1].SAD;
                    }
                }
            }
        }
    }
}

void opticalflow_Propagation(std::vector<std::vector<RESULT_SIM_BM>> &conv_map, const cv::Mat &block, int conv_map_cols, int conv_map_rows, int r)
{
    std::cout << "start Propagation" << std::endl;
    for (int i = 20; i < conv_map_cols; i++)
    {
        for (int j = 0; j < conv_map_rows; j++)
        {
            if (r % 2 == 1)
            {
                if (i - 1 > 0)
                {
                    if (conv_map[i - 1][j].depth != 0)
                    {
                        if (conv_map[i - 1][j].SAD < conv_map[i][j].SAD)
                        {
                            if (conv_map[i - 1][j].x + WIN_SIZE < block.cols)
                            {
                                conv_map[i][j].x = conv_map[i - 1][j].x + WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].x = conv_map[i - 1][j].x;
                            }
                            conv_map[i][j].y = conv_map[i - 1][j].y;
                            conv_map[i][j].SAD = conv_map[i - 1][j].SAD;
                        }
                    }
                }
                if (i - 2 > 0)
                {
                    if (conv_map[i - 2][j].depth != 0)
                    {
                        if (conv_map[i - 2][j].SAD < conv_map[i][j].SAD)
                        {
                            if (conv_map[i - 2][j].x + WIN_SIZE < block.cols)
                            {
                                conv_map[i][j].x = conv_map[i - 2][j].x + WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].x = conv_map[i - 2][j].x;
                            }
                            conv_map[i][j].y = conv_map[i - 2][j].y;
                            conv_map[i][j].SAD = conv_map[i - 2][j].SAD;
                        }
                    }
                }
                if (j - 1 > 0)
                {
                    if (conv_map[i][j - 1].depth != 0)
                    {
                        if (conv_map[i][j - 1].SAD < conv_map[i][j].SAD)
                        {
                            conv_map[i][j].x = conv_map[i][j - 1].x;
                            if (conv_map[i][j - 1].y + WIN_SIZE < block.rows)
                            {
                                conv_map[i][j].y = conv_map[i][j - 1].y + WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].y = conv_map[i][j - 1].y;
                            }
                            conv_map[i][j].SAD = conv_map[i][j - 1].SAD;
                        }
                    }
                }
                if (j - 2 > 0)
                {
                    if (conv_map[i][j - 1].depth != 0)
                    {
                        if (conv_map[i][j - 2].SAD < conv_map[i][j].SAD)
                        {
                            conv_map[i][j].x = conv_map[i][j - 2].x;
                            if (conv_map[i][j - 2].y + WIN_SIZE < block.rows)
                            {
                                conv_map[i][j].y = conv_map[i][j - 2].y + WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].y = conv_map[i][j - 2].y;
                            }
                            conv_map[i][j].SAD = conv_map[i][j - 2].SAD;
                        }
                    }
                }
            }
            else
            {
                if (i + 1 < conv_map_cols)
                {
                    if (conv_map[i + 1][j].depth != 0)
                    {
                        if (conv_map[i + 1][j].SAD < conv_map[i][j].SAD)
                        {
                            if (conv_map[i + 1][j].x - WIN_SIZE > 0)
                            {
                                conv_map[i][j].x = conv_map[i + 1][j].x - WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].x = conv_map[i + 1][j].x;
                            }
                            conv_map[i][j].y = conv_map[i + 1][j].y;
                            conv_map[i][j].SAD = conv_map[i + 1][j].SAD;
                        }
                    }
                }
                if (i + 2 < conv_map_cols)
                {
                    if (conv_map[i + 2][j].depth != 0)
                    {
                        if (conv_map[i + 2][j].SAD < conv_map[i][j].SAD)
                        {
                            if (conv_map[i + 2][j].x - WIN_SIZE > 0)
                            {
                                conv_map[i][j].x = conv_map[i + 2][j].x - WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].x = conv_map[i + 2][j].x;
                            }
                            conv_map[i][j].y = conv_map[i + 2][j].y;
                            conv_map[i][j].SAD = conv_map[i + 2][j].SAD;
                        }
                    }
                }
                if (j + 1 < conv_map_rows)
                {
                    if (conv_map[i][j + 1].depth != 0)
                    {
                        if (conv_map[i][j + 1].SAD < conv_map[i][j].SAD)
                        {
                            conv_map[i][j].x = conv_map[i][j + 1].x;
                            if (conv_map[i][j + 1].y - WIN_SIZE > 0)
                            {
                                conv_map[i][j].y = conv_map[i][j + 1].y - WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].y = conv_map[i][j + 1].y;
                            }
                            conv_map[i][j].SAD = conv_map[i][j + 1].SAD;
                        }
                    }
                }
                if (j + 2 < conv_map_rows)
                {
                    if (conv_map[i][j + 2].depth != 0)
                    {
                        if (conv_map[i][j + 2].SAD < conv_map[i][j].SAD)
                        {
                            conv_map[i][j].x = conv_map[i][j + 2].x;
                            if (conv_map[i][j + 2].y - WIN_SIZE > 0)
                            {
                                conv_map[i][j].y = conv_map[i][j + 2].y - WIN_SIZE;
                            }
                            else
                            {
                                conv_map[i][j].y = conv_map[i][j + 2].y;
                            }
                            conv_map[i][j].SAD = conv_map[i][j + 2].SAD;
                        }
                    }
                }
            }
        }
    }
}

void make_Random_Map(std::vector<std::vector<POSITION>> &random_map, int conv_map_cols, int conv_map_rows)
{
    std::mt19937 rng(1); // 乱数
    std::uniform_int_distribution<int> col_dist(0, conv_map_cols - 1);
    std::uniform_int_distribution<int> row_dist(0, conv_map_rows - 1);
    for (int x = 0; x < conv_map_cols; x++)
    {
        for (int y = 0; y < conv_map_rows; y++)
        {
            random_map[x][y].x = col_dist(rng);
            random_map[x][y].y = row_dist(rng);
        }
    }
}

void make_Random_Map(std::vector<std::vector<int>> &random_map, int range, int conv_map_cols, int conv_map_rows)
{
    std::mt19937 rng(1); // 乱数
    std::uniform_int_distribution<int> dist(0, range - 1);
    int rand_uni = 0;
    for (int x = 0; x < conv_map_cols; x++)
    {
        for (int y = 0; y < conv_map_rows; y++)
        {
            random_map[x][y] = dist(rng);
        }
    }
}

void conv_map_to_depth_map(std::vector<std::vector<CORRES>> &conv_map, cv::Mat &block, cv::Mat &dst)
{
    int depth_H = 0;
    double dist = 0;
    int depth = 0;
    cv::Mat depth_img = cv::Mat(block.rows, block.cols, CV_8UC3);
    for (int i = 0; i < block.cols / 3; i++)
    {
        for (int j = 0; j < block.rows / 3; j++)
        {
            if (conv_map[i][j].SAD == -1)
            {
                for (int bx = -1; bx <= 1; bx++)
                {
                    for (int by = -1; by <= 1; by++)
                    {
                        depth_img.at<cv::Vec3b>(3 * (j + 1) - 2 + by, 3 * (i + 1) - 2 + bx) = cv::Vec3b(0, 0, 0);
                    }
                }
            }
            else
            {
                dist = sqrt((3 * (i + 1) - 2 - conv_map[i][j].x) * (3 * (i + 1) - 2 - conv_map[i][j].x) + (3 * (j + 1) - 2 - conv_map[i][j].y) * (3 * (j + 1) - 2 - conv_map[i][j].y));
                // std::cout << 3 * (i + 1) - 2 << " " << conv_map[i][j].x << " " << 3 * (j + 1) - 2 << " " << conv_map[i][j].y << " " << dist << "    ->    ";
                depth = D_CALI / dist;
                // std::cout << depth << "    ->    ";
                depth_H = depth * D_MAG;
                if (depth_H > 150)
                    depth_H = 150;
                if (depth_H < 0)
                    depth_H = 0;
                // std::cout << depth_H << "\n";
                //  cv::rectangle(dst, cv::Point(3 * (i), 3 * (j)), cv::Point(3 * (i + 1) - 1, 3 * (j + 1) - 1), cv::Scalar(depth_H, 255, 255), cv::FILLED);
                ///*
                for (int bx = -1; bx <= 1; bx++)
                {
                    for (int by = -1; by <= 1; by++)
                    {
                        if (block.at<unsigned char>(3 * (j + 1) - 2 + by, 3 * (i + 1) - 2 + bx) != 0)
                        {
                            depth_img.at<cv::Vec3b>(3 * (j + 1) - 2 + by, 3 * (i + 1) - 2 + bx) = cv::Vec3b(depth_H, 255, 255);
                        }
                        else
                        {
                            depth_img.at<cv::Vec3b>(3 * (j + 1) - 2 + by, 3 * (i + 1) - 2 + bx) = cv::Vec3b(0, 0, 0);
                        }
                    }
                }
                //*/
                // std::cout << "(" << 3 * (i + 1) - 2 << ", " << 3 * (j + 1) - 2 << ") " << conv_map[i][j].x << " " << conv_map[i][j].y << " " << conv_map[i][j].SAD << " " << depth_H << "\n";
            }
        }
    }
    cv::cvtColor(depth_img, dst, cv::COLOR_HSV2BGR);
}

void conv_map_to_depth_map(std::vector<std::vector<RESULT_SIM_BM>> &conv_map, cv::Mat &block, cv::Mat &dst)
{
    int depth_H = 0;
    double dist = 0;
    int depth = 0;
    int half_win = WIN_SIZE / 2;
    int div = WIN_SIZE / 2 + 1;
    cv::Mat depth_img = cv::Mat(block.rows, block.cols, CV_8UC3);
    for (int i = 0; i < block.cols / WIN_SIZE; i++)
    {
        for (int j = 0; j < block.rows / WIN_SIZE; j++)
        {
            if (conv_map[i][j].SAD == -1)
            {
                for (int bx = -1; bx <= 1; bx++)
                {
                    for (int by = -1; by <= 1; by++)
                    {
                        depth_img.at<cv::Vec3b>(3 * (j + 1) - 2 + by, 3 * (i + 1) - 2 + bx) = cv::Vec3b(0, 0, 0);
                    }
                }
            }
            else
            {
                dist = sqrt((WIN_SIZE * (i + 1) - div - conv_map[i][j].x) * (WIN_SIZE * (i + 1) - div - conv_map[i][j].x) + (WIN_SIZE * (j + 1) - div - conv_map[i][j].y) * (WIN_SIZE * (j + 1) - div - conv_map[i][j].y));
                // std::cout << 3 * (i + 1) - 2 << " " << conv_map[i][j].x << " " << 3 * (j + 1) - 2 << " " << conv_map[i][j].y << " " << dist << "    ->    ";
                depth = D_CALI / dist;
                // std::cout << depth << "    ->    ";
                depth_H = depth * D_MAG;
                if (depth_H > 150)
                    depth_H = 150;
                if (depth_H < 0)
                    depth_H = 0;
                // std::cout << depth_H << "\n";
                //  cv::rectangle(dst, cv::Point(3 * (i), 3 * (j)), cv::Point(3 * (i + 1) - 1, 3 * (j + 1) - 1), cv::Scalar(depth_H, 255, 255), cv::FILLED);
                ///*
                for (int bx = -half_win; bx <= half_win; bx++)
                {
                    for (int by = -half_win; by <= half_win; by++)
                    {
                        if (block.at<unsigned char>(WIN_SIZE * (j + 1) - div + by, WIN_SIZE * (i + 1) - div + bx) != 0)
                        {
                            depth_img.at<cv::Vec3b>(WIN_SIZE * (j + 1) - div + by, WIN_SIZE * (i + 1) - div + bx) = cv::Vec3b(depth_H, 255, 255);
                            conv_map[i][j].depth = depth_H;
                        }
                        else
                        {
                            depth_img.at<cv::Vec3b>(WIN_SIZE * (j + 1) - div + by, WIN_SIZE * (i + 1) - div + bx) = cv::Vec3b(0, 0, 0);
                            conv_map[i][j].depth = 0;
                        }
                    }
                }
                //*/
                // std::cout << "(" << 3 * (i + 1) - 2 << ", " << 3 * (j + 1) - 2 << ") " << conv_map[i][j].x << " " << conv_map[i][j].y << " " << conv_map[i][j].SAD << " " << depth_H << "\n";
            }
        }
    }
    cv::cvtColor(depth_img, dst, cv::COLOR_HSV2BGR);
}

void Outlier_Rejection(cv::Mat &depth, cv::Mat &origin)
{
    for (int x = 0; x < depth.cols; x++)
    {
        for (int y = 0; y < depth.rows; y++)
        {
            if (origin.at<unsigned char>(y, x) == 0)
            {
                depth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
}

void debug_matching_point(std::vector<std::vector<CORRES>> &conv_map, cv::Mat &right)
{
    cv::Mat mp = cv::Mat(right.rows, right.cols, CV_8UC3);
    for (int x = 0; x < right.cols / 3; x++)
    {
        for (int y = 0; y < right.rows / 3; y++)
        {
            for (int i = -1; i < 2; i++)
            {
                for (int j = -1; j < 2; j++)
                {
                    if (conv_map[x][y].x + i < right.cols && conv_map[x][y].y + j < right.rows && conv_map[x][y].x + i > 0 && conv_map[x][y].y + j > 0)
                        mp.at<cv::Vec3b>(conv_map[x][y].y + j, conv_map[x][y].x + i) += cv::Vec3b(10, 0, 0);
                }
            }
        }
    }

    Outlier_Rejection(mp, right);
    cv::imshow("match point", mp);
}

void check_pixel(cv::Mat &src)
{
    std::string location_path = "logs/patch_match/pixel.txt";
    std::ofstream outputfile(location_path);

    for (int x = WIN_SIZE / 2; x < src.cols; x += WIN_SIZE)
    {
        for (int y = WIN_SIZE / 2; y < src.rows; y += WIN_SIZE)
        {
            for (int by = -1; by < 2; by++)
            {
                for (int bx = -1; bx < 2; bx++)
                {
                    if (COLOR_MODE == NTSS_GRAY)
                        outputfile << (int)src.at<unsigned char>(y + by, x + bx) << "\t";
                    else if (COLOR_MODE == NTSS_RGB)
                        outputfile << src.at<cv::Vec3b>(y + by, x + bx) << "\t";
                }
                outputfile << "\n";
            }
            outputfile << "\n\n";
        }
    }
    outputfile.close();
}

void opticalflow_PM(cv::Mat &block, cv::Mat &src, cv::Mat &dst, int block_size, int mode, int LorR)
{
    /*
        block       深度推定したい画像
        src         対応を探したい画像
        dst         深度マップを出力する画像
        block_size  探索フレームのサイズ基本3x3
        mode        グレースケールorカラー
        LorR        左から右か右から左
    */

    dst = cv::Mat(block.rows, block.cols, CV_8UC3);
    std::cout << "start opticalflow_PM" << std::endl;
    cv::Mat frame = block.clone();
    cv::Mat frame2 = src.clone();
    if (block.cols % 3 != 0 || block.rows % 3 != 0)
    {
        cv::Mat padblock, padsrc;
        int r = 0;
        int l = 0;
        if (block.cols % 3 == 2)
        {
            r = 1;
        }
        else if (block.cols % 3 == 1)
        {
            r = 2;
        }

        if (block.rows % 3 == 2)
        {
            l = 1;
        }
        else if (block.rows % 3 == 1)
        {
            l = 2;
        }
        // 1280x720 -> 1281x720 1281 can devide by 3.
        copyMakeBorder(block, padblock, 0, l, 0, r, cv::BORDER_REPLICATE);
        copyMakeBorder(src, padsrc, 0, l, 0, r, cv::BORDER_REPLICATE);
        frame = padblock.clone();
        frame2 = padsrc.clone();
    }
    std::vector<std::vector<CORRES>> conv_map(frame.cols / 3, (std::vector<CORRES>(frame.rows / 3, {0, 0, 0})));

    std::cout << conv_map.size() << std::endl;
    int conv_map_cols = frame.cols / 3;
    int conv_map_rows = frame.rows / 3;

    std::vector<std::vector<POSITION>> random_ini(conv_map_cols, (std::vector<POSITION>(conv_map_rows, {0, 0})));

    make_Random_Map(random_ini, conv_map_cols, conv_map_rows);

    // Inirializarion

    opticalflow_Initialize(block, src, conv_map, block_size, random_ini, mode);

    std::cout << conv_map_cols << " " << conv_map_rows << std::endl;
    int range = SEARCH_RANGE;

    // Propagation
    ///*
    std::vector<std::vector<int>> random_map(conv_map_cols, (std::vector<int>(conv_map_rows, 0)));
    for (int r = 1; r < 6; r++)
    {
        opticalflow_Propagation(conv_map, block, conv_map_cols, conv_map_rows, r);

        make_Random_Map(random_map, range * range, conv_map_cols, conv_map_rows);
        opticalflow_Random_Search(block, src, conv_map, conv_map_cols, conv_map_rows, random_map, range, mode);
        range /= 2;
    }
    //*/

    conv_map_to_depth_map(conv_map, block, dst);
}

void opticalflow_BM(cv::Mat &block, cv::Mat &src, cv::Mat &dst, int block_size, int mode, int LorR)
{
    /*
    block       深度推定したい画像
    src         対応を探したい画像
    dst         深度マップを出力する画像
    block_size  探索フレームのサイズ基本3x3
    mode        グレースケールorカラー
    LorR        左から右か右から左
*/

    dst = cv::Mat(block.rows, block.cols, CV_8UC3);
    std::cout << "start opticalflow_PM" << std::endl;
    cv::Mat frame = block.clone();
    cv::Mat frame2 = src.clone();
    if (block.cols % WIN_SIZE != 0 || block.rows % WIN_SIZE != 0)
    {
        cv::Mat padblock, padsrc;
        int r = 0;
        int l = 0;

        if (block.cols % block_size != 0)
            r = block_size - block.cols % block_size;

        if (block.rows % block_size != 0)
            l = block_size - block.rows % block_size;
        // 1280x720 -> 1281x720 1281 can devide by 3.
        copyMakeBorder(block, padblock, 0, l, 0, r, cv::BORDER_REPLICATE);
        copyMakeBorder(src, padsrc, 0, l, 0, r, cv::BORDER_REPLICATE);
        frame = padblock.clone();
        frame2 = padsrc.clone();
    }
    // std::vector<std::vector<CORRES>> conv_map(frame.cols / block_size, (std::vector<CORRES>(frame.rows / block_size, {0, 0, 0})));
    std::vector<std::vector<RESULT_SIM_BM>> conv_map(frame.cols / block_size, (std::vector<RESULT_SIM_BM>(frame.rows / block_size, {0, 0, 0, 0})));

    std::cout << conv_map.size() << std::endl;
    int conv_map_cols = frame.cols / block_size;
    int conv_map_rows = frame.rows / block_size;

    block_Matching(frame, frame2, conv_map, WIN_SIZE, COLOR_MODE, L2R);

    conv_map_to_depth_map(conv_map, frame, dst);
    cv::imshow("test1", dst);

    for (int i = 0; i < 0; i++)
    {
        opticalflow_Propagation(conv_map, frame, conv_map_cols, conv_map_rows, i);
        conv_map_to_depth_map(conv_map, frame, dst);
        cv::imshow("test2", dst);
        cv::waitKey(1000);
    }
    // Outlier_Rejection(dst, block);

    // debug_matching_point(conv_map, frame2);
    // get_depth(dst, dst, block);
}

void test_PM()
{
    int debug = 1;
    cv::Mat left = cv::imread(LEFT_IMG, COLOR_MODE);
    cv::Mat right = cv::imread(RIGHT_IMG, COLOR_MODE);
    if (left.empty() == 1)
    {
        return;
    }
    if (right.empty() == 1)
    {
        return;
    }

    cv::Mat dst;
    std::chrono::system_clock::time_point start, fin;
    clock_t begin = clock();
    start = std::chrono::system_clock::now();

    // cv::resize(left, left, cv::Size(WIDTH, HEIGHT));
    // cv::resize(right, right, cv::Size(WIDTH, HEIGHT));

    if (DO_ILBP == 1)
    {
        cvt_ILBP(left, left);
        // cv::erode(left, left, cv::Mat(), cv::Point(-1, -1), 2);
        // cv::dilate(left, left, cv::Mat(), cv::Point(-1, -1), 2);

        cvt_ILBP(right, right);
        // cv::erode(right, right, cv::Mat(), cv::Point(-1, -1), 2);
        // cv::dilate(right, right, cv::Mat(), cv::Point(-1, -1), 2);
    }
    if (debug == 1)
        check_pixel(left);

    opticalflow_BM(left, right, dst, WIN_SIZE, COLOR_MODE, L2R);
    // opticalflow_BM(right, left, dst, WIN_SIZE, COLOR_MODE, R2L);
    //  opticalflow_PM(left, right, dst, WIN_SIZE, COLOR_MODE, L2R);

    fin = std::chrono::system_clock::now();

    clock_t end = clock();
    print_elapsed_time(begin, end);

    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(fin - start).count() / 1000.0);
    printf("time %lf[s]\n", time / 1000);

    if (debug == 1)
    {
        cv::imshow("a", dst);
        cv::imwrite("images/match_sample/test.jpg", dst);
        cv::imshow("left", left);
        cv::imshow("right", right);
        const int key = cv::waitKey(0);
        if (key == 'q' /*113*/) // qボタンが押されたとき
        {
            // break; // whileループから抜ける．
        }
    }
}

int main()
{
    unsigned int thread_num = std::thread::hardware_concurrency();
    std::cout << "This CPU has " << thread_num << " threads" << std::endl;

    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    make_LUT(UNIFORMED_LUT);
    test_PM();
    return 0;
}

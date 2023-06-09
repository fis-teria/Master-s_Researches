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

// Thread Flags
std::mutex mtx_;
int thread_finish = 0;
int get_src = 0;
int read_src = 0;
int next_Dtbase = 0;
int make_DB = 0;

// grobal variables

// get camera images
cv::Mat src;
// get Pre-run images Dataase
std::vector<cv::Mat> img_dtbase;
int start_check = 0;

// get Position result
std::vector<int> position;
int position_size = 0;
// output
std::string SAVE = "SAVE_Postion.txt";

struct resulT
{
    double max;
    int num;
};

struct location
{
    double px;
    double py;
};

void print_elapsed_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed Time: %15.7f sec\n", elapsed);
}

std::string make_path(std::string dir, int dir_num, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + std::to_string(dir_num) + "_template/00000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 100)
    {
        back = dir + std::to_string(dir_num) + "_template/0000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 1000)
    {
        back = dir + std::to_string(dir_num) + "_template/000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 10000)
    {
        back = dir + std::to_string(dir_num) + "_template/00" + std::to_string(var) + tag;
        return back;
    }
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

int read_save()
{
    int save_num;
    std::ifstream ifs;

    ifs = std::ifstream(SAVE.c_str());
    std::string str;
    getline(ifs, str);
    std::cout << str << std::endl;
    save_num = atoi(str.c_str());
    return save_num;
}

void save(int m)
{
    std::ofstream savefile;
    savefile = std::ofstream(SAVE.c_str());
    savefile << m << std::endl;
    savefile.close();
}

void make_locate_image(const cv::Mat &src, const cv::Mat &dst, cv::Mat &locate_image)
{
    std::cout << "start make locate image" << std::endl;
    cv::Mat srcr, dstr;
    cv::resize(src, srcr, cv::Size(), 0.5, 0.5);
    cv::resize(dst, dstr, cv::Size(), 0.5, 0.5);
    locate_image = cv::Mat(srcr.rows, srcr.cols * 2, CV_8UC1);
    locate_image = cv::Scalar::all(0);
    std::cout << srcr.size << "\n"
              << locate_image.size << "\nstart make locate image first" << std::endl;

    for (int x = 0; x < locate_image.cols / 2; x++)
    {
        for (int y = 0; y < locate_image.rows; y++)
        {
            locate_image.at<float>(y, x) = srcr.at<float>(y, x);
        }
        std::cout << " x " << x << std::endl;
    }
    cv::imshow("locate", locate_image);

    std::cout << "start make locate image second\n"
              << locate_image.cols / 2 << std::endl;

    for (int x = 0; x < locate_image.cols / 2; x++)
    {
        for (int y = 0; y < locate_image.rows; y++)
        {
            // std::cout << y << " x " << x << std::endl;
            locate_image.at<float>(y, x + locate_image.cols / 2) = dstr.at<float>(y, x);
        }
    }

    std::cout << "clear make locate image" << std::endl;
    cv::imshow("locate2", locate_image);
}

std::vector<std::string> split(std::string &input, char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter))
    {
        result.push_back(field);
    }
    return result;
}

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_LBP(const cv::Mat &src, cv::Mat &lbp)
{
    lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
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
    // cv::imshow("second", lbp);
}

void test_thread()
{
    std::cout << "test_thread start" << std::endl;
    int num = 0;
    while (num < 100)
    {
        if (src.empty())
        {
            continue;
        }
        else
        {
            while (read_src != 1)
                read_src = 0;
            std::cout << get_src << std::endl;
            cv::imshow("get_src", src);
            cv::waitKey(100);
            get_src++;
            num++;
        }
    }
    std::cout << "test_thread end" << std::endl;
    thread_finish++;
}

void read_image()
{
    std::cout << "thread read_image() start" << std::endl;
    std::string img_path;
    std::string dir = "images/Test0";
    std::string tag = ".jpg";
    cv::Mat dst;
    int num = 0;
    while (thread_finish == 0)
    {
        img_path = make_path(dir, 1, num, tag);
        std::cout << img_path << std::endl;

        cvt_LBP(cv::imread(img_path, 0), dst);
        std::cout << "LBP" << std::endl;
        // std::lock_guard<std::mutex> lock(mtx_);
        src = dst;
        read_src++;
        while (read_src == 1)
        {
            if (thread_finish == 1)
                break;
        }

        while (get_src == 0)
        {
            if (thread_finish == 1)
                break;
        }
        get_src = 0;
        num++;
    }
    std::cout << "thread read_image() end" << std::endl;
}

void make_Dtbase()
{
    std::cout << "make_Dtbase() start" << std::endl;
    std::string path;
    std::string dir = "images/Test0";
    int dir_num = 0;
    std::string tag;
    cv::Mat dst;
    tag = ".jpg";
    for (int i = 0; i < 200; i++)
    {
        path = make_path(dir, dir_num, i, tag);
        // std::cout << path << std::endl;
        cvt_LBP(cv::imread(path, 0), dst);
        img_dtbase.resize(i + 1);
        img_dtbase[i] = dst;
        if (img_dtbase[i].empty())
        {
            std::cout << "nothing to image " << path << std::endl;
        }
    }

    make_DB = 1;

    while (next_Dtbase != 1)
    {
    }

    img_dtbase.clear();
    img_dtbase.shrink_to_fit();

    int count = 0;
    std::cout << "start position " << std::endl;
    for (int i = position[position.size() - 1] - 3; i < position[position.size() - 1] + 3; i++)
    {
        path = make_path(dir, dir_num, i, tag);
        img_dtbase.resize(count + 1);
        cvt_LBP(cv::imread(path, 0), dst);
        img_dtbase[count] = dst;
        count++;
    }

    std::cout << "make_Dtbase() end" << std::endl;
}

void update_Dtbase()
{
    std::string path;
    std::string dir = "images/Test0";
    int dir_num = 0;
    std::string tag;
    path = make_path(dir, dir_num, position[position.size() - 1], tag);
    img_dtbase.erase(img_dtbase.begin());
    img_dtbase.push_back(cv::imread(path, 0));
}

void position_check()
{
    std::cout << "position_check() start" << std::endl;
    std::ifstream ifs("images/Test01/num.txt");
    std::string line;
    std::vector<int> check;
    std::vector<resulT> result;
    int result_size = 0;
    cv::Mat sl_tim;
    double min_sl, max_sl;
    cv::Point p_min_sl, p_max_sl;
    int num = 0;
    cv::Mat temp;
    while (getline(ifs, line))
    {
        num = std::stoi(line);
        std::cout << num << std::endl;
    }

    while (make_DB == 0)
    {
    }
    make_DB = 0;

    for (int i = 0; i < num; i++)
    {
        while (read_src != 1)
        {
        }
        read_src = 0;

        std::cout << "match position " << std::endl;

        clock_t begin = clock();
        for (int j = 0; j < img_dtbase.size(); j++)
        {
            clock_t begin = clock();
            temp = img_dtbase[i].clone();
            // std::cout << get_src << std::endl;
            cv::matchTemplate(src, temp(cv::Range(temp.rows / 10, (9 * temp.rows) / 10), cv::Range(temp.cols / 10, (9 * temp.cols) / 10)), sl_tim, cv::TM_CCOEFF_NORMED);
            cv::minMaxLoc(sl_tim, &min_sl, &max_sl, &p_min_sl, &p_max_sl);

            result.resize(result_size + 1);
            result[result_size].max = max_sl;
            result[result_size].num = j;
            result_size++;
            clock_t end = clock();
            print_elapsed_time(begin, end);
        }

        std::sort(result.begin(), result.end(), [](const resulT &alpha, const resulT &beta)
                  { return alpha.max < beta.max; });

        position.resize(position_size + 1);
        position[position_size];
        if (position_size > 0)
        {
            if (position[position_size] - position[position_size - 1] > 2 || position[position_size] - position[position_size - 1] < -2)
            {
                start_check++;
            }
            else
            {
                start_check = 0;
            }
        }
        position_size++;

        if (start_check == 10)
        {
            next_Dtbase++;
        }

        std::cout << "now locate " << result[result.size() - 1].num << std::endl;

        update_Dtbase();

        clock_t end = clock();
        print_elapsed_time(begin, end);

        std::cout << "show" << std::endl;
        cv::imshow("get_src", cv::imread(make_tpath("images/Test0", 0, result[result.size() - 1].num, ".jpg")));
        cv::waitKey(10);
        get_src++;
    }

    std::cout << "position_check() end" << std::endl;
}

int main()
{
    // std::thread test(test_thread);
    std::thread thread_read_image(read_image);
    std::thread thread_make_Dtbase(make_Dtbase);
    std::thread thread_position(position_check);

    // test.join();
    thread_read_image.join();
    thread_make_Dtbase.join();
    thread_position.join();
    return 0;
}
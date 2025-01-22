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

void normalizeUC_MinMax(std::vector<float> &input_vector, std::vector<unsigned char> &output_vector);
int getFileNums(std::string folderPath, std::vector<std::string> &file_names);

int main(int argc, char **argv)
{

    std::string actual_dir = argv[1];
    std::string actual_run = actual_dir + "/color";
    std::string test_dir = argv[2];
    std::string test_run = test_dir + "/color";

    int skip = 12;
    std::vector<std::string> actual_vec, test_vec;

    int matrix_cols = getFileNums(actual_run, actual_vec) - 2;
    int matrix_rows = getFileNums(test_run, test_vec) - 2;
    std::cout << "result matrix size " << matrix_cols << " x " << matrix_rows << std::endl;

    cv::Mat seq_matrix = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);

    if (seq_matrix.empty())
    {
        std::cout << "cannot make images" << std::endl;
        return -1;
    }
    seq_matrix = cv::Scalar::all(255);

    for (int i = 0; i < matrix_cols; i++)
    {
        cv::Mat actual_img = cv::imread(common::make_path(actual_run, i, ".jpg"), 0);
        cv::resize(actual_img, actual_img, cv::Size(84, 48));

        for (int x = 0; x < actual_img.cols; x += skip)
        {
            for (int y = 0; y < actual_img.rows; y += skip)
            {
                cv::normalize(actual_img(cv::Range(y, y + skip - 1), cv::Range(x, x + skip - 1)), actual_img(cv::Range(y, y + skip - 1), cv::Range(x, x + skip - 1)), 0, 255, cv::NORM_MINMAX, CV_8UC1);
            }
        }

        for (int j = 0; j < matrix_rows; j++)
        {
            double local_ave = 0;
            double global_ave = 0;
            int global_count = 0;
            cv::Mat test_img = cv::imread(common::make_path(test_run, j, ".jpg"), 0);
            cv::resize(test_img, test_img, cv::Size(84, 48));

            for (int x = 0; x < test_img.cols; x += skip)
            {
                for (int y = 0; y < test_img.rows; y += skip)
                {
                    cv::normalize(test_img(cv::Range(y, y + skip - 1), cv::Range(x, x + skip - 1)), test_img(cv::Range(y, y + skip - 1), cv::Range(x, x + skip - 1)), 0, 255, cv::NORM_MINMAX, CV_8UC1);
                }
            }
            for (int x = 0; x < test_img.cols; x += skip)
            {
                for (int y = 0; y < test_img.rows; y += skip)
                {
                    local_ave = 0;
                    for (int u = 0; u < skip; u++)
                    {
                        for (int v = 0; v < skip; v++)
                        {
                            local_ave += actual_img.at<unsigned char>(y + v, x + u) - test_img.at<unsigned char>(y + v, x + u);
                        }
                    }
                    local_ave = local_ave / (skip * skip);
                    global_ave += (int)local_ave;
                    global_count++;
                    //fprintf(stderr, "\r%f", local_ave);
                }
            }
            global_ave /= global_count;
            //std::cout << abs(global_ave) << " " << local_ave << std::endl;

            seq_matrix.at<unsigned char>(j, i) = abs((int)global_ave);
        }

        fprintf(stderr, "\r%s\n\r[%3.2f / 100]", common::make_path(actual_run, i, ".jpg").c_str() ,(float)(((float)i/(float)matrix_cols)*100));
        cv::imshow("a", seq_matrix);
        cv::waitKey(50);
    }

    cv::imwrite("../data/images/seqslam/old/day_run4-3_color.jpg", seq_matrix);

    return 0;
}

void normalizeUC_MinMax(std::vector<float> &input_vector, std::vector<unsigned char> &output_vector)
{
    std::vector<float> vec(input_vector.size(), 0);
    // output_vector.resize(input_vector.size());
    std::vector<unsigned char> output;

    std::copy(input_vector.begin(), input_vector.end(), vec.begin());

    float min = *std::min_element(begin(vec), end(vec));
    float max = *std::max_element(begin(vec), end(vec));
    float dif = max - min;

    for (int i = 0; i < vec.size(); i++)
    {
        output_vector.push_back((int)(((vec[i] - min) / dif) * 255));
        // std::cout << (int)output[i] << " " << (int)((vec[i] - min) / dif * 255) << std::endl;
    }

    // std::copy(output.begin(), output.end(), output_vector.begin());
}

/**
 * @brief フォルダ以下のファイル一覧を取得する関数
 * @param[in]	folderPath	フォルダパス
 * @param[out]	file_names	ファイル名一覧
 * return		true:成功, false:失敗
 */
int getFileNums(std::string folderPath, std::vector<std::string> &file_names)
{
    using namespace std::filesystem;
    directory_iterator iter(folderPath), end;
    std::error_code err;

    for (; iter != end && !err; iter.increment(err))
    {
        const directory_entry entry = *iter;

        file_names.push_back(entry.path().string());
        // printf("%s\n", file_names.back().c_str());
    }

    /* エラー処理 */
    if (err)
    {
        std::cout << err.value() << std::endl;
        std::cout << err.message() << std::endl;
        return -1;
    }
    return file_names.size();
}
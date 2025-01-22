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

void normalizeUC_MinMax(std::vector<float> &input_vector, cv::Mat &output);
int getFileNums(std::string folderPath, std::vector<std::string> &file_names);

int main(int argc, char **argv)
{
    int start = atoi(argv[1]);
    int end = atoi(argv[2]);
    int anno = 0;
    int before_anno = 0;
    int count = 0;
    int lcount = 0;
    std::string loop_ans;

    std::string csv_dir = argv[3];
    std::string cosine_csv;
    std::string obstacle_csv;
    std::string anno_dir = argv[4];
    std::string input = anno_dir + "/annotation.txt";
    std::string actual_run = anno_dir + "/depth";
    std::string test_dir = argv[5];
    std::string test_run = test_dir + "/depth";
    std::ifstream ifs(input);
    std::string line;
    std::string str_buf;
    std::string str_conma_buf;

    std::vector<std::string> actual_vec, test_vec;

    int matrix_cols = getFileNums(actual_run, actual_vec) - 2;
    int matrix_rows = getFileNums(test_run, test_vec) - 2;
    std::cout << "result matrix size " << matrix_cols << " x " << matrix_rows << std::endl;

    cv::Mat seq_matrix_color = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_color2 = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_dedge = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_d_color = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_d_dedge = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_s_color = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_s_dedge = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_i_color = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);
    cv::Mat seq_matrix_i_dedge = cv::Mat(matrix_rows, matrix_cols, CV_8UC1);

    if (seq_matrix_color.empty())
    {
        std::cout << "cannot make images" << std::endl;
        return -1;
    }
    seq_matrix_color = cv::Scalar::all(255);
    seq_matrix_color2 = cv::Scalar::all(255);
    seq_matrix_dedge = cv::Scalar::all(255);
    seq_matrix_d_color = cv::Scalar::all(255);
    seq_matrix_d_dedge = cv::Scalar::all(255);
    seq_matrix_s_color = cv::Scalar::all(255);
    seq_matrix_s_dedge = cv::Scalar::all(255);
    seq_matrix_i_color = cv::Scalar::all(255);
    seq_matrix_i_dedge = cv::Scalar::all(255);

    std::vector<int> annotation;

    std::cout
        << "annotation file " << input << std::endl;

    while (getline(ifs, line))
    {
        annotation.push_back(atoi(line.c_str()));
        // std::cout << annotation[annotation.size() - 1] << std::endl;
    }

    for (int i = start; i < end; i++)
    {
        std::string csv_num = std::__cxx11::to_string(i);
        cosine_csv = csv_dir + "/cosine_" + csv_num.insert(0, 6 - csv_num.length(), '0') + ".csv";
        obstacle_csv = csv_dir + "/obstacle" + csv_num.insert(0, 6 - csv_num.length(), '0') + ".csv";

        std::ifstream cosine(cosine_csv);
        std::ifstream obstacle(obstacle_csv);

        anno = annotation[i];
        // std::cout << "grand true" << anno << std::endl;

        std::vector<float> buf(6);
        std::vector<float> st_clr;
        std::vector<float> st_dedg;
        std::vector<float> st_d_clr;
        std::vector<float> st_d_dedg;
        std::vector<float> st_s_clr;
        std::vector<float> st_s_dedg;
        std::vector<float> st_i_clr;
        std::vector<float> st_i_dedg;

        cv::Mat color = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat dedge = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat d_color = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat d_dedge = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat s_color = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat s_dedge = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat i_color = cv::Mat(matrix_rows, 1, CV_32F);
        cv::Mat i_dedge = cv::Mat(matrix_rows, 1, CV_32F);

        int loop_count = 0;
        int data_num = 0;

        if (!cosine)
        {
            std::cout << "not found file" << std::endl;
            break;
        }

        while (getline(cosine, str_buf))
        {
            // 「,」区切りごとにデータを読み込むためにistringstream型にする
            std::istringstream i_stream(str_buf);
            // header skip
            if (loop_count == 0)
            {
                loop_count++;
                continue;
            }
            // 「,」区切りごとにデータを読み込む
            while (getline(i_stream, str_conma_buf, ','))
            {
                // csvファイルに書き込む
                buf[data_num++] = atof(str_conma_buf.c_str());
            }
            data_num = 0;
            st_clr.push_back(buf[0]);
            st_dedg.push_back(buf[1]);
            // std::cout << color.at<float>(loop_count - 1, 1) << " " << buf[0] << std::endl;
            loop_count++;
            // std::cout << database[loop_count - 1].frame_num << " " << database[loop_count - 1].color << " " << database[loop_count - 1].dedge << std::endl;
        }

        /*
        for (int x = 0; x < color.rows; x++)
        {
            std::cout << color.at<float>(x, 1) << " ";
            if (x % 10 == 0)
            {
                std::cout << std::endl;
            }
        }
        std::cout << "\n**********************************" << std::endl;
        //*/

        normalizeUC_MinMax(st_clr, color);
        normalizeUC_MinMax(st_dedg, dedge);
        // cv::normalize(dedge, dedge, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        /*
        for (int x = 0; x < color.rows; x++)
        {
            //std::cout << (int)color.at<unsigned char>(x, 1) << " ";
            if (x % 10 == 0 && x != 0)
            {
                std::cout << std::endl;
            }
        }
        std::cout << "\n**********************************" << std::endl;
        */
        //*/
        for (int j = 0; j < matrix_rows; j++)
        {
            seq_matrix_color.at<unsigned char>(j, i) = (int)color.at<unsigned char>(j, 1);
            seq_matrix_dedge.at<unsigned char>(j, i) = (int)dedge.at<unsigned char>(j, 1);
            seq_matrix_d_color.at<unsigned char>(j, i) = (int)color.at<unsigned char>(j, 1);
            seq_matrix_d_dedge.at<unsigned char>(j, i) = (int)dedge.at<unsigned char>(j, 1);
            seq_matrix_s_color.at<unsigned char>(j, i) = (int)color.at<unsigned char>(j, 1);
            seq_matrix_s_dedge.at<unsigned char>(j, i) = (int)dedge.at<unsigned char>(j, 1);
            seq_matrix_i_color.at<unsigned char>(j, i) = (int)color.at<unsigned char>(j, 1);
            seq_matrix_i_dedge.at<unsigned char>(j, i) = (int)dedge.at<unsigned char>(j, 1);
            //seq_matrix_color2.at<unsigned char>(j, i) = 255 - (int)clr[j];
            // std::cout << 255 - seq_matrix_color.at<unsigned char>(j, i) << " " << (int)color.at<unsigned char>(j, 1) << std::endl;
        }
        //*/
        /*
        color.copyTo(seq_matrix_color.col(i));
        dedge.copyTo(seq_matrix_dedge.col(i));
        color.copyTo(seq_matrix_d_color.col(i));
        dedge.copyTo(seq_matrix_d_dedge.col(i));
        color.copyTo(seq_matrix_s_color.col(i));
        dedge.copyTo(seq_matrix_s_dedge.col(i));
        color.copyTo(seq_matrix_i_color.col(i));
        dedge.copyTo(seq_matrix_i_dedge.col(i));
        */
        if (!obstacle)
        {
        }
        else
        {
            loop_count = 0;
            while (getline(obstacle, str_buf))
            {
                // 「,」区切りごとにデータを読み込むためにistringstream型にする
                std::istringstream i_stream(str_buf);
                // header skip
                if (loop_count == 0)
                {
                    loop_count++;
                    continue;
                }
                // 「,」区切りごとにデータを読み込む
                while (getline(i_stream, str_conma_buf, ','))
                {
                    // csvファイルに書き込む
                    buf[data_num++] = atof(str_conma_buf.c_str());
                }
                data_num = 0;

                st_d_clr.push_back(buf[0]);
                st_s_clr.push_back(buf[1]);
                st_d_dedg.push_back(buf[2]);
                st_s_dedg.push_back(buf[3]);
                st_i_clr.push_back(buf[4]);
                st_i_dedg.push_back(buf[5]);
                /*
                d_color.at<float>(loop_count - 1, 1) = buf[0];
                s_color.at<float>(loop_count - 1, 1) = buf[1];
                d_dedge.at<float>(loop_count - 1, 1) = buf[2];
                s_dedge.at<float>(loop_count - 1, 1) = buf[3];
                i_color.at<float>(loop_count - 1, 1) = buf[4];
                i_dedge.at<float>(loop_count - 1, 1) = buf[5];
                */
                // std::cout << i_color.at<float>(loop_count - 1, 1) << " " << buf[4] << std::endl;

                // std::cout << database[loop_count - 1].frame_num << " " << " " << database[loop_count - 1].color << " " << database[loop_count - 1].dedge <<" " << database[loop_count - 1].det_color << " " << database[loop_count - 1].det_dedge << " " << database[loop_count - 1].seg_color << " " << database[loop_count - 1].seg_dedge << " " << database[loop_count - 1].ip_color << " " << database[loop_count - 1].ip_dedge << std::endl;
                loop_count++;
            }

            /*
            for (int x = 0; x < color.rows; x++)
            {
                std::cout << i_color.at<float>(x, 1) << " ";
                if (x % 10 == 0)
                {
                    std::cout << std::endl;
                }
            }
            std::cout << "\n**********************************" << std::endl;
            //*/
            /*
            cv::normalize(d_color, d_color, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::waitKey(10);
            cv::normalize(d_dedge, d_dedge, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::waitKey(10);
            cv::normalize(s_color, s_color, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::waitKey(10);
            cv::normalize(s_dedge, s_dedge, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::waitKey(10);
            cv::normalize(i_color, i_color, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::waitKey(10);
            cv::normalize(i_dedge, i_dedge, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            //*/
            normalizeUC_MinMax(st_d_clr, d_color);
            normalizeUC_MinMax(st_d_dedg, d_dedge);
            normalizeUC_MinMax(st_s_clr, s_color);
            normalizeUC_MinMax(st_s_dedg, s_dedge);
            normalizeUC_MinMax(st_i_clr, i_color);
            normalizeUC_MinMax(st_i_dedg, i_dedge);

            /*
             for (int x = 0; x < color.rows; x++)
             {
                 std::cout << (int)i_color.at<unsigned char>(x, 1) << " ";
                 if (x % 10 == 0 && x != 0)
                 {
                     std::cout << std::endl;
                 }
             }
             std::cout << "\n**********************************" << std::endl;
            */
           //*/
            for (int j = 0; j < matrix_rows; j++)
            {
                seq_matrix_d_color.at<unsigned char>(j, i) = d_color.at<unsigned char>(j, 1);
                seq_matrix_d_dedge.at<unsigned char>(j, i) = d_dedge.at<unsigned char>(j, 1);
                seq_matrix_s_color.at<unsigned char>(j, i) = s_color.at<unsigned char>(j, 1);
                seq_matrix_s_dedge.at<unsigned char>(j, i) = s_dedge.at<unsigned char>(j, 1);
                seq_matrix_i_color.at<unsigned char>(j, i) = i_color.at<unsigned char>(j, 1);
                seq_matrix_i_dedge.at<unsigned char>(j, i) = i_dedge.at<unsigned char>(j, 1);
                // std::cout << 255 - d_color.at<unsigned char>(j, 1) << std::endl;
            }
             //*/
             /*
            d_color.copyTo(seq_matrix_d_color.col(i));
            d_dedge.copyTo(seq_matrix_d_dedge.col(i));
            s_color.copyTo(seq_matrix_s_color.col(i));
            s_dedge.copyTo(seq_matrix_s_dedge.col(i));
            i_color.copyTo(seq_matrix_i_color.col(i));
            i_dedge.copyTo(seq_matrix_i_dedge.col(i));
            */
        }
        fprintf(stderr, "\r[%3.2f / 100]", (float)(((float)i / (float)matrix_cols) * 100));
    }
    for (int x = 0; x < matrix_cols; x++)
    {
        // std::cout << (int)seq_matrix_i_color.at<unsigned char>(0, x) << std::endl;
    }

    cv::imshow("a", seq_matrix_color);
    cv::imshow("a2", seq_matrix_color2);
    cv::imshow("b", seq_matrix_dedge);
    cv::imshow("c", seq_matrix_d_color);
    cv::imshow("d", seq_matrix_d_dedge);
    cv::imshow("e", seq_matrix_s_color);
    cv::imshow("f", seq_matrix_s_dedge);
    cv::imshow("g", seq_matrix_i_color);
    cv::imshow("h", seq_matrix_i_dedge);
    cv::waitKey(0);

    return 0;
}

void normalizeUC_MinMax(std::vector<float> &input_vector, cv::Mat &output)
{
    std::vector<float> vec(input_vector.size(), 0);
    // output_vector.resize(input_vector.size());
    // std::vector<unsigned char> output;

    std::copy(input_vector.begin(), input_vector.end(), vec.begin());

    float min = *std::min_element(begin(vec), end(vec));
    float max = *std::max_element(begin(vec), end(vec));
    float dif = max - min;

    for (int i = 0; i < vec.size(); i++)
    {
        // output_vector.push_back((int)(((vec[i] - min) / dif) * 255));
        output.at<unsigned char>(i, 1) = 255 - (int)(((vec[i] - min) / dif) * 255);
        //std::cout << (int)output[i] << " " << (int)((vec[i] - min) / dif * 255) << std::endl;
        //std::cout <<(int)output.at<unsigned char>(i, 1) <<" " << 255 - (int)(((vec[i] - min) / dif) * 255) << std::endl;
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
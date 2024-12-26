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

struct CSV_READ
{
    int frame_num = 0;
    double color = 0;
    double dedge = 0;
    double det_color = 0;
    double det_dedge = 0;
    double seg_color = 0;
    double seg_dedge = 0;
    double ip_color = 0;
    double ip_dedge = 0;
};

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
    std::string input = argv[4];
    std::ifstream ifs(input);
    std::string line;
    std::string str_buf;
    std::string str_conma_buf;

    std::vector<int> annotation;

    std::cout
        << "annotation file " << input << std::endl;

    while (getline(ifs, line))
    {
        annotation.push_back(atoi(line.c_str()));
        std::cout << annotation[annotation.size() - 1] << std::endl;
    }

    int normal_frame = 0;
    int obstacle_frame = 0;

    //+-2frame
    double color_top1_pm2 = 0;
    double color_top3_pm2 = 0;
    double color_top5_pm2 = 0;
    double color_top10_pm2 = 0;
    //+-5frame
    double color_top1_pm5 = 0;
    double color_top3_pm5 = 0;
    double color_top5_pm5 = 0;
    double color_top10_pm5 = 0;

    //+-2frame
    double dedge_top1_pm2 = 0;
    double dedge_top3_pm2 = 0;
    double dedge_top5_pm2 = 0;
    double dedge_top10_pm2 = 0;
    //+-5frame
    double dedge_top1_pm5 = 0;
    double dedge_top3_pm5 = 0;
    double dedge_top5_pm5 = 0;
    double dedge_top10_pm5 = 0;

    //+-2frame
    double o_color_top1_pm2 = 0;
    double o_color_top3_pm2 = 0;
    double o_color_top5_pm2 = 0;
    double o_color_top10_pm2 = 0;
    //+-5frame
    double o_color_top1_pm5 = 0;
    double o_color_top3_pm5 = 0;
    double o_color_top5_pm5 = 0;
    double o_color_top10_pm5 = 0;

    //+-2frame
    double o_dedge_top1_pm2 = 0;
    double o_dedge_top3_pm2 = 0;
    double o_dedge_top5_pm2 = 0;
    double o_dedge_top10_pm2 = 0;
    //+-5frame
    double o_dedge_top1_pm5 = 0;
    double o_dedge_top3_pm5 = 0;
    double o_dedge_top5_pm5 = 0;
    double o_dedge_top10_pm5 = 0;

    //+-2frame
    double det_color_top1_pm2 = 0;
    double det_color_top3_pm2 = 0;
    double det_color_top5_pm2 = 0;
    double det_color_top10_pm2 = 0;
    //+-5frame
    double det_color_top1_pm5 = 0;
    double det_color_top3_pm5 = 0;
    double det_color_top5_pm5 = 0;
    double det_color_top10_pm5 = 0;

    //+-2frame
    double det_dedge_top1_pm2 = 0;
    double det_dedge_top3_pm2 = 0;
    double det_dedge_top5_pm2 = 0;
    double det_dedge_top10_pm2 = 0;
    //+-5frame
    double det_dedge_top1_pm5 = 0;
    double det_dedge_top3_pm5 = 0;
    double det_dedge_top5_pm5 = 0;
    double det_dedge_top10_pm5 = 0;

    //+-2frame
    double seg_color_top1_pm2 = 0;
    double seg_color_top3_pm2 = 0;
    double seg_color_top5_pm2 = 0;
    double seg_color_top10_pm2 = 0;
    //+-5frame
    double seg_color_top1_pm5 = 0;
    double seg_color_top3_pm5 = 0;
    double seg_color_top5_pm5 = 0;
    double seg_color_top10_pm5 = 0;

    //+-2frame
    double seg_dedge_top1_pm2 = 0;
    double seg_dedge_top3_pm2 = 0;
    double seg_dedge_top5_pm2 = 0;
    double seg_dedge_top10_pm2 = 0;
    //+-5frame
    double seg_dedge_top1_pm5 = 0;
    double seg_dedge_top3_pm5 = 0;
    double seg_dedge_top5_pm5 = 0;
    double seg_dedge_top10_pm5 = 0;

    //+-2frame
    double ip_color_top1_pm2 = 0;
    double ip_color_top3_pm2 = 0;
    double ip_color_top5_pm2 = 0;
    double ip_color_top10_pm2 = 0;
    //+-5frame
    double ip_color_top1_pm5 = 0;
    double ip_color_top3_pm5 = 0;
    double ip_color_top5_pm5 = 0;
    double ip_color_top10_pm5 = 0;

    //+-2frame
    double ip_dedge_top1_pm2 = 0;
    double ip_dedge_top3_pm2 = 0;
    double ip_dedge_top5_pm2 = 0;
    double ip_dedge_top10_pm2 = 0;
    //+-5frame
    double ip_dedge_top1_pm5 = 0;
    double ip_dedge_top3_pm5 = 0;
    double ip_dedge_top5_pm5 = 0;
    double ip_dedge_top10_pm5 = 0;

    for (int i = start; i < end; i++)
    {
        normal_frame++;
        std::string csv_num = std::__cxx11::to_string(i);
        cosine_csv = csv_dir + "/cosine_" + csv_num.insert(0, 6 - csv_num.length(), '0') + ".csv";
        obstacle_csv = csv_dir + "/obstacle" + csv_num.insert(0, 6 - csv_num.length(), '0') + ".csv";

        std::ifstream cosine(cosine_csv);
        std::ifstream obstacle(obstacle_csv);

        anno = annotation[i];
        // std::cout << "grand true" << anno << std::endl;

        std::vector<CSV_READ> database;
        std::vector<double> buf(6);
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
            database.resize(loop_count);
            database[loop_count - 1].frame_num = loop_count - 1;
            database[loop_count - 1].color = buf[0];
            database[loop_count - 1].dedge = buf[1];
            loop_count++;
            // std::cout << database[loop_count - 1].frame_num << " " << database[loop_count - 1].color << " " << database[loop_count - 1].dedge << std::endl;
        }
        data_num = 0;
        loop_count = 0;

        //+-2frame
        int top1_pm2 = 0;
        int top3_pm2 = 0;
        int top5_pm2 = 0;
        int top10_pm2 = 0;
        //+-5frame
        int top1_pm5 = 0;
        int top3_pm5 = 0;
        int top5_pm5 = 0;
        int top10_pm5 = 0;
        std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                  { return alpha.color > beta.color; });
        for (int x = 0; x < 10; x++)
        {
            if (x == 0)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top1_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top1_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top3_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top3_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (0 < x && x <= 3)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top3_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top3_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (3 < x && x <= 5)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (5 < x)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
        }
        if (top1_pm2 > 0)
        {
            color_top1_pm2++;
        }
        if (top1_pm5 > 0)
        {
            color_top1_pm5++;
        }
        if (top3_pm2 > 0)
        {
            color_top3_pm2++;
        }
        if (top3_pm5 > 0)
        {
            color_top3_pm5++;
        }
        if (top5_pm2 > 0)
        {
            color_top5_pm2++;
        }
        if (top5_pm5 > 0)
        {
            color_top5_pm5++;
        }
        if (top10_pm2 > 0)
        {
            color_top10_pm2++;
        }
        if (top10_pm5 > 0)
        {
            color_top10_pm5++;
        }
        top1_pm2 = 0;
        top3_pm2 = 0;
        top5_pm2 = 0;
        top10_pm2 = 0;
        top1_pm5 = 0;
        top3_pm5 = 0;
        top5_pm5 = 0;
        top10_pm5 = 0;
        /*
        for (int j = 0; j < 10; j++)
        {
            std::cout << database[j].frame_num << " " << database[j].color << " " << database[j].dedge << std::endl;
        }
        std::cout << "------------------------" << std::endl;
       */
        std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                  { return alpha.dedge > beta.dedge; });
        for (int x = 0; x < 10; x++)
        {
            if (x == 0)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top1_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top1_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top3_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top3_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (0 < x && x <= 3)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top3_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top3_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (3 < x && x <= 5)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top5_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top5_pm5++;
                }
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
            else if (5 < x)
            {
                if (abs(database[x].frame_num - anno) < 3)
                {
                    top10_pm2++;
                }
                if (abs(database[x].frame_num - anno) < 6)
                {
                    top10_pm5++;
                }
            }
        }
        if (top1_pm2 > 0)
        {
            dedge_top1_pm2++;
        }
        if (top1_pm5 > 0)
        {
            dedge_top1_pm5++;
        }
        if (top3_pm2 > 0)
        {
            dedge_top3_pm2++;
        }
        if (top3_pm5 > 0)
        {
            dedge_top3_pm5++;
        }
        if (top5_pm2 > 0)
        {
            dedge_top5_pm2++;
        }
        if (top5_pm5 > 0)
        {
            dedge_top5_pm5++;
        }
        if (top10_pm2 > 0)
        {
            dedge_top10_pm2++;
        }
        if (top10_pm5 > 0)
        {
            dedge_top10_pm5++;
        }
        top1_pm2 = 0;
        top3_pm2 = 0;
        top5_pm2 = 0;
        top10_pm2 = 0;
        top1_pm5 = 0;
        top3_pm5 = 0;
        top5_pm5 = 0;
        top10_pm5 = 0;

        /*
        for (int j = 0; j < 10; j++)
        {
            std::cout << database[j].frame_num << " " << database[j].color << " " << database[j].dedge << std::endl;
        }
        */
        if (!obstacle)
        {
        }
        else
        {
            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.color > beta.color; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                o_color_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                o_color_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                o_color_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                o_color_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                o_color_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                o_color_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                o_color_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                o_color_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;
            /*
            for (int j = 0; j < 10; j++)
            {
                std::cout << database[j].frame_num << " " << database[j].color << " " << database[j].dedge << std::endl;
            }
            std::cout << "------------------------" << std::endl;
           */
            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.dedge > beta.dedge; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                o_dedge_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                o_dedge_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                o_dedge_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                o_dedge_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                o_dedge_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                o_dedge_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                o_dedge_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                o_dedge_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            obstacle_frame++;
            std::cout << "get obstacle file" << std::endl;
            std::cout << obstacle_csv << std::endl;

            // std::cout << "------------------------" << std::endl;
            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.frame_num < beta.frame_num; });
            /*
            for (int j = 0; j < 10; j++)
            {
                std::cout << database[j].frame_num << " " << database[j].color << " " << database[j].dedge << std::endl;
            }*/
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
                database[loop_count - 1].det_color = buf[0];
                database[loop_count - 1].seg_color = buf[1];
                database[loop_count - 1].det_dedge = buf[2];
                database[loop_count - 1].seg_dedge = buf[3];
                database[loop_count - 1].ip_color = buf[4];
                database[loop_count - 1].ip_dedge = buf[5];
                // std::cout << database[loop_count - 1].frame_num << " " << " " << database[loop_count - 1].color << " " << database[loop_count - 1].dedge <<" " << database[loop_count - 1].det_color << " " << database[loop_count - 1].det_dedge << " " << database[loop_count - 1].seg_color << " " << database[loop_count - 1].seg_dedge << " " << database[loop_count - 1].ip_color << " " << database[loop_count - 1].ip_dedge << std::endl;
                loop_count++;
            }

            std::cout << "Obstacle data analysis" << std::endl;
            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.det_color > beta.det_color; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                det_color_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                det_color_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                det_color_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                det_color_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                det_color_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                det_color_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                det_color_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                det_color_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.seg_color > beta.seg_color; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                seg_color_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                seg_color_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                seg_color_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                seg_color_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                seg_color_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                seg_color_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                seg_color_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                seg_color_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.det_dedge > beta.det_dedge; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                det_dedge_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                det_dedge_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                det_dedge_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                det_dedge_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                det_dedge_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                det_dedge_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                det_dedge_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                det_dedge_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.seg_dedge > beta.seg_dedge; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                seg_dedge_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                seg_dedge_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                seg_dedge_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                seg_dedge_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                seg_dedge_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                seg_dedge_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                seg_dedge_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                seg_dedge_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.ip_color > beta.ip_color; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                ip_color_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                ip_color_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                ip_color_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                ip_color_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                ip_color_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                ip_color_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                ip_color_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                ip_color_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;

            std::sort(database.begin(), database.end(), [](const CSV_READ &alpha, const CSV_READ &beta)
                      { return alpha.ip_dedge > beta.ip_dedge; });
            for (int x = 0; x < 10; x++)
            {
                if (x == 0)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top1_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top1_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (0 < x && x <= 3)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top3_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top3_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (3 < x && x <= 5)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top5_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top5_pm5++;
                    }
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
                else if (5 < x)
                {
                    if (abs(database[x].frame_num - anno) < 3)
                    {
                        top10_pm2++;
                    }
                    if (abs(database[x].frame_num - anno) < 6)
                    {
                        top10_pm5++;
                    }
                }
            }
            if (top1_pm2 > 0)
            {
                ip_dedge_top1_pm2++;
            }
            if (top1_pm5 > 0)
            {
                ip_dedge_top1_pm5++;
            }
            if (top3_pm2 > 0)
            {
                ip_dedge_top3_pm2++;
            }
            if (top3_pm5 > 0)
            {
                ip_dedge_top3_pm5++;
            }
            if (top5_pm2 > 0)
            {
                ip_dedge_top5_pm2++;
            }
            if (top5_pm5 > 0)
            {
                ip_dedge_top5_pm5++;
            }
            if (top10_pm2 > 0)
            {
                ip_dedge_top10_pm2++;
            }
            if (top10_pm5 > 0)
            {
                ip_dedge_top10_pm5++;
            }
            top1_pm2 = 0;
            top3_pm2 = 0;
            top5_pm2 = 0;
            top10_pm2 = 0;
            top1_pm5 = 0;
            top3_pm5 = 0;
            top5_pm5 = 0;
            top10_pm5 = 0;
        }
    }
    std::cout << "all frame " << normal_frame << " : obstacle_frame " << obstacle_frame << std::endl;
    std::cout << "color +-2:  top1:" << color_top1_pm2 << " top3:" << color_top3_pm2 << " top5:" << color_top5_pm2 << " top10:" << color_top10_pm2 << std::endl;
    std::cout << "color +-5:  top1:" << color_top1_pm5 << " top3:" << color_top3_pm5 << " top5:" << color_top5_pm5 << " top10:" << color_top10_pm5 << std::endl;
    std::cout << "dedge +-2:  top1:" << dedge_top1_pm2 << " top3:" << dedge_top3_pm2 << " top5:" << dedge_top5_pm2 << " top10:" << dedge_top10_pm2 << std::endl;
    std::cout << "dedge +-5:  top1:" << dedge_top1_pm5 << " top3:" << dedge_top3_pm5 << " top5:" << dedge_top5_pm5 << " top10:" << dedge_top10_pm5 << std::endl;
    std::cout << "o_color +-2:  top1:" << o_color_top1_pm2 << " top3:" << o_color_top3_pm2 << " top5:" << o_color_top5_pm2 << " top10:" << o_color_top10_pm2 << std::endl;
    std::cout << "o_color +-5:  top1:" << o_color_top1_pm5 << " top3:" << o_color_top3_pm5 << " top5:" << o_color_top5_pm5 << " top10:" << o_color_top10_pm5 << std::endl;
    std::cout << "o_dedge +-2:  top1:" << o_dedge_top1_pm2 << " top3:" << o_dedge_top3_pm2 << " top5:" << o_dedge_top5_pm2 << " top10:" << o_dedge_top10_pm2 << std::endl;
    std::cout << "o_dedge +-5:  top1:" << o_dedge_top1_pm5 << " top3:" << o_dedge_top3_pm5 << " top5:" << o_dedge_top5_pm5 << " top10:" << o_dedge_top10_pm5 << std::endl;
    std::cout << "det_color +-2:  top1:" << det_color_top1_pm2 << " top3:" << det_color_top3_pm2 << " top5:" << det_color_top5_pm2 << " top10:" << det_color_top10_pm2 << std::endl;
    std::cout << "det_color +-5:  top1:" << det_color_top1_pm5 << " top3:" << det_color_top3_pm5 << " top5:" << det_color_top5_pm5 << " top10:" << det_color_top10_pm5 << std::endl;
    std::cout << "seg_color +-2:  top1:" << seg_color_top1_pm2 << " top3:" << seg_color_top3_pm2 << " top5:" << seg_color_top5_pm2 << " top10:" << seg_color_top10_pm2 << std::endl;
    std::cout << "seg_color +-5:  top1:" << seg_color_top1_pm5 << " top3:" << seg_color_top3_pm5 << " top5:" << seg_color_top5_pm5 << " top10:" << seg_color_top10_pm5 << std::endl;
    std::cout << "det_dedge +-2:  top1:" << det_dedge_top1_pm2 << " top3:" << det_dedge_top3_pm2 << " top5:" << det_dedge_top5_pm2 << " top10:" << det_dedge_top10_pm2 << std::endl;
    std::cout << "det_dedge +-5:  top1:" << det_dedge_top1_pm5 << " top3:" << det_dedge_top3_pm5 << " top5:" << det_dedge_top5_pm5 << " top10:" << det_dedge_top10_pm5 << std::endl;
    std::cout << "seg_dedge +-2:  top1:" << seg_dedge_top1_pm2 << " top3:" << seg_dedge_top3_pm2 << " top5:" << seg_dedge_top5_pm2 << " top10:" << seg_dedge_top10_pm2 << std::endl;
    std::cout << "seg_dedge +-5:  top1:" << seg_dedge_top1_pm5 << " top3:" << seg_dedge_top3_pm5 << " top5:" << seg_dedge_top5_pm5 << " top10:" << seg_dedge_top10_pm5 << std::endl;
    std::cout << "ip_color +-2:  top1:" << ip_color_top1_pm2 << " top3:" << ip_color_top3_pm2 << " top5:" << ip_color_top5_pm2 << " top10:" << ip_color_top10_pm2 << std::endl;
    std::cout << "ip_color +-5:  top1:" << ip_color_top1_pm5 << " top3:" << ip_color_top3_pm5 << " top5:" << ip_color_top5_pm5 << " top10:" << ip_color_top10_pm5 << std::endl;
    std::cout << "ip_dedge +-2:  top1:" << ip_dedge_top1_pm2 << " top3:" << ip_dedge_top3_pm2 << " top5:" << ip_dedge_top5_pm2 << " top10:" << ip_dedge_top10_pm2 << std::endl;
    std::cout << "ip_dedge +-5:  top1:" << ip_dedge_top1_pm5 << " top3:" << ip_dedge_top3_pm5 << " top5:" << ip_dedge_top5_pm5 << " top10:" << ip_dedge_top10_pm5 << std::endl;

    std::cout << "all frame " << normal_frame << " : obstacle_frame " << obstacle_frame << std::endl;
    //+-2
    std::cout << "カラー画像 &" << color_top1_pm2 / normal_frame * 100 << " &" << color_top3_pm2 / normal_frame * 100 << " &" << color_top5_pm2 / normal_frame * 100 << " &" << color_top10_pm2 / normal_frame * 100 << std::endl;
    std::cout << "深度エッジ画像 &" << dedge_top1_pm2 / normal_frame * 100 << " &" << dedge_top3_pm2 / normal_frame * 100 << " &" << dedge_top5_pm2 / normal_frame * 100 << " &" << dedge_top10_pm2 / normal_frame * 100 << std::endl;
    std::cout << "color" << std::endl;
    std::cout << "前処理なし&" << o_color_top1_pm2 / obstacle_frame * 100 << " &" << o_color_top3_pm2 / obstacle_frame * 100 << " &" << o_color_top5_pm2 / obstacle_frame * 100 << " &" << o_color_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "矩形領域除去&" << det_color_top1_pm2 / obstacle_frame * 100 << " &" << det_color_top3_pm2 / obstacle_frame * 100 << " &" << det_color_top5_pm2 / obstacle_frame * 100 << " &" << det_color_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "マスク画像&" << seg_color_top1_pm2 / obstacle_frame * 100 << " &" << seg_color_top3_pm2 / obstacle_frame * 100 << " &" << seg_color_top5_pm2 / obstacle_frame * 100 << " &" << seg_color_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "インペインティング&" << ip_color_top1_pm2 / obstacle_frame * 100 << " &" << ip_color_top3_pm2 / obstacle_frame * 100 << " &" << ip_color_top5_pm2 / obstacle_frame * 100 << " &" << ip_color_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "depth edge" << std::endl;
    std::cout << "前処理なし&" << o_dedge_top1_pm2 / obstacle_frame * 100 << " &" << o_dedge_top3_pm2 / obstacle_frame * 100 << " &" << o_dedge_top5_pm2 / obstacle_frame * 100 << " &" << o_dedge_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "矩形領域除去&" << det_dedge_top1_pm2 / obstacle_frame * 100 << " &" << det_dedge_top3_pm2 / obstacle_frame * 100 << " &" << det_dedge_top5_pm2 / obstacle_frame * 100 << " &" << det_dedge_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "マスク領域除去&" << seg_dedge_top1_pm2 / obstacle_frame * 100 << " &" << seg_dedge_top3_pm2 / obstacle_frame * 100 << " &" << seg_dedge_top5_pm2 / obstacle_frame * 100 << " &" << seg_dedge_top10_pm2 / obstacle_frame * 100 << std::endl;
    std::cout << "インペインティング&" << ip_dedge_top1_pm2 / obstacle_frame * 100 << " &" << ip_dedge_top3_pm2 / obstacle_frame * 100 << " &" << ip_dedge_top5_pm2 / obstacle_frame * 100 << " &" << ip_dedge_top10_pm2 / obstacle_frame * 100 << std::endl;
    //+-5
    std::cout << "カラー画像&" << color_top1_pm5 / normal_frame * 100 << " &" << color_top3_pm5 / normal_frame * 100 << " &" << color_top5_pm5 / normal_frame * 100 << " &" << color_top10_pm5 / normal_frame * 100 << std::endl;
    std::cout << "深度エッジ画像&" << dedge_top1_pm5 / normal_frame * 100 << " &" << dedge_top3_pm5 / normal_frame * 100 << " &" << dedge_top5_pm5 / normal_frame * 100 << " &" << dedge_top10_pm5 / normal_frame * 100 << std::endl;
    std::cout << "color" << std::endl;
    std::cout << "前処理なし&" << o_color_top1_pm5 / obstacle_frame * 100 << " &" << o_color_top3_pm5 / obstacle_frame * 100 << " &" << o_color_top5_pm5 / obstacle_frame * 100 << " &" << o_color_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "矩形領域除去&" << det_color_top1_pm5 / obstacle_frame * 100 << " &" << det_color_top3_pm5 / obstacle_frame * 100 << " &" << det_color_top5_pm5 / obstacle_frame * 100 << " &" << det_color_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "マスク領域除去&" << seg_color_top1_pm5 / obstacle_frame * 100 << " &" << seg_color_top3_pm5 / obstacle_frame * 100 << " &" << seg_color_top5_pm5 / obstacle_frame * 100 << " &" << seg_color_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "インペインティング&" << ip_color_top1_pm5 / obstacle_frame * 100 << " &" << ip_color_top3_pm5 / obstacle_frame * 100 << " &" << ip_color_top5_pm5 / obstacle_frame * 100 << " &" << ip_color_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "depth edge" << std::endl;
    std::cout << "前処理なし&" << o_dedge_top1_pm5 / obstacle_frame * 100 << " &" << o_dedge_top3_pm5 / obstacle_frame * 100 << " &" << o_dedge_top5_pm5 / obstacle_frame * 100 << " &" << o_dedge_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "矩形領域除去&" << det_dedge_top1_pm5 / obstacle_frame * 100 << " &" << det_dedge_top3_pm5 / obstacle_frame * 100 << " &" << det_dedge_top5_pm5 / obstacle_frame * 100 << " &" << det_dedge_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "マスク領域除去&" << seg_dedge_top1_pm5 / obstacle_frame * 100 << " &" << seg_dedge_top3_pm5 / obstacle_frame * 100 << " &" << seg_dedge_top5_pm5 / obstacle_frame * 100 << " &" << seg_dedge_top10_pm5 / obstacle_frame * 100 << std::endl;
    std::cout << "インペインティング&" << ip_dedge_top1_pm5 / obstacle_frame * 100 << " &" << ip_dedge_top3_pm5 / obstacle_frame * 100 << " &" << ip_dedge_top5_pm5 / obstacle_frame * 100 << " &" << ip_dedge_top10_pm5 / obstacle_frame * 100 << std::endl;

    std::cout << "カラー画像 &" << (color_top1_pm2 - o_color_top1_pm2 + ip_color_top1_pm2) / normal_frame * 100 << " &" << (color_top3_pm2 - o_color_top3_pm2 + ip_color_top3_pm2) / normal_frame * 100 << " &" << (color_top5_pm2 - o_color_top5_pm2 + ip_color_top5_pm2) / normal_frame * 100 << " &" << (color_top10_pm2 - o_color_top10_pm2 + ip_color_top10_pm2) / normal_frame * 100 << std::endl;
    std::cout << "深度エッジ画像 &" << (dedge_top1_pm2 - o_dedge_top1_pm2 + seg_dedge_top1_pm2) / normal_frame * 100 << " &" << (dedge_top3_pm2 - o_dedge_top3_pm2 + seg_dedge_top3_pm2)/ normal_frame * 100 << " &" << (dedge_top5_pm2 - o_dedge_top5_pm2 + seg_dedge_top5_pm2)/ normal_frame * 100 << " &" << (dedge_top10_pm2 -o_dedge_top10_pm2 + seg_dedge_top10_pm2)/ normal_frame * 100 << std::endl;
    std::cout << "カラー画像&" << (color_top1_pm5 - o_color_top1_pm5 + ip_color_top1_pm5) / normal_frame * 100 << " &" << (color_top3_pm5 - o_color_top3_pm5 + ip_color_top3_pm5) / normal_frame * 100 << " &" << (color_top5_pm5 - o_color_top5_pm5 + ip_color_top5_pm5) / normal_frame * 100 << " &" << (color_top10_pm5 - o_color_top10_pm5 + ip_color_top10_pm5) / normal_frame * 100 << std::endl;
    std::cout << "深度エッジ画像 &" << (dedge_top1_pm5 - o_dedge_top1_pm5 + seg_dedge_top1_pm5) / normal_frame * 100 << " &" << (dedge_top3_pm5 - o_dedge_top3_pm5 + seg_dedge_top3_pm5)/ normal_frame * 100 << " &" << (dedge_top5_pm5 - o_dedge_top5_pm5 + seg_dedge_top5_pm5)/ normal_frame * 100 << " &" << (dedge_top10_pm5 -o_dedge_top10_pm5 + seg_dedge_top10_pm5)/ normal_frame * 100 << std::endl;
    return 0;
}
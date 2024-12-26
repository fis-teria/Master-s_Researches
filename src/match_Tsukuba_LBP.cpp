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

// #define rootGunDai
#define MATCH00to01
#define root1001_1430match2_917
#define root1001_1435match2_1001_1030
std::string SAVE = "save_tsukuba_LBP.txt";
/*
0917_1349
start 50
end 4546

1001_1030
start 33
end 2469

1001_1435
start 27
end 4548
*/cosine_det_img,cosine_seg_img,cosine_det_dedge,cosine_seg_dedge,cosine_ip_img,cosine_ip_dedge
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

int position_Check(const cv::Mat src, std::string path, int now_locate)
{
    cv::Mat rim, sim, tim;
    cv::Mat sl_rim, sl_sim, sl_tim;
    /*
    rim = base image
    sim = normal template iamge
    tim = result normal template matching
    sl_rim = LBP  after resize rim
    sl_sim = LBP  after resize sim
    sl_tim = result cim template matching

    */
    int i, j; // seed num
    int num_m;
    int path_m;
    // int load_num = read_save();
    std::vector<resulT> result;
    int average = 0;
    int result_size = 0;
    int start, end;
    std::string dir;
    std::string tag;
    std::string dpath;

    double min_sl, max_sl;
    cv::Point p_min_sl, p_max_sl;

    // Match  0917_1349 to 1001_1435
    num_m = 4820; // 0917_1349
    path_m = 2;
    dir = "images/Test0";
    tag = ".jpg";

    start = now_locate - 100;
    if (start < 2)
    {
        start = 2;
    }
    end = now_locate + 100;
    if (end > num_m)
    {
        end = num_m;
    }
    std::cout << start << " : " << end << std::endl;
    // LBP
    cvt_LBP(src, sl_rim);
    //  std::cout << "start matching" << std::endl;
    for (int k = start; k < end; k++)
    {
        dpath = make_tpath(dir, 1, k, tag);
        // std::cout << "match " << path << " : " << dpath << std::endl;

        sim = cv::imread(dpath, cv::IMREAD_GRAYSCALE);
        cv::resize(sim, sim, cv::Size(), 0.1, 0.1);

        //clock_t begin = clock();
        // cvt_LBP(sim, sl_sim);
        //  small LBP
        cvt_LBP(sim, sl_sim);
        cv::matchTemplate(sl_rim, sl_sim(cv::Range(sl_sim.rows / 10, (9 * sl_sim.rows) / 10), cv::Range(sl_sim.cols / 10, (9 * sl_sim.cols) / 10)), sl_tim, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(sl_tim, &min_sl, &max_sl, &p_min_sl, &p_max_sl);
        //clock_t endc = clock();
        // print_elapsed_time(begin, endc);

        result.resize(result_size + 1);
        result[result_size].max = max_sl;
        result[result_size].num = k;
        result_size++;
        // std::cout << k << std::endl;
    }
    std::sort(result.begin(), result.end(), [](const resulT &alpha, const resulT &beta)
              { return alpha.max < beta.max; });
    /*
    std::cout << result[result_size - 1].num << " " << result_size << " " << result.size() << std::endl;
    for (int j = 0; j < result_size - 1; j++)
    {
        std::cout << result[j].num << std::endl;
    }
    */
    return result[result.size() - 1].num;
}

int main(int argc, char **argv)
{
    /*
    if (argc != 2)
    {
        std::cout << "Error Too many or too few command line arguments" << std::endl;
        return 0;
    }
    */

    cv::Mat img, src, dst, locate_image;
    cv::Mat img_lst[2];
    cv::Mat srcr, dstr;
    std::string dir, rdir;
    std::string tag;
    std::string path;
    int now_locate = 0;
    int time = 0;
    int load_num = read_save();
    if (load_num == -1)
    {
        std::cout << "Error command line num only use 0, 1, 2" << std::endl;
        return 0;
    }

    dir = "images/Test0";
    rdir = "gra/result_Tsukuba0";
    tag = ".jpg";

    std::ifstream ifs("gra/Tsukuba00_locate/location.csv");
    std::ifstream jfs("gra/Tsukuba02_locate/location.csv");

    std::string line;
    std::vector<location> locate_main;
    std::vector<location> locate_tmp;
    int lm_size = 0;
    int lt_size = 0;
    std::cout << "a" <<std::endl;
    while (getline(ifs, line))
    {

        std::vector<std::string> strvec = split(line, ',');

        locate_main.resize(lm_size + 1);
        locate_main[lm_size].px = std::stod(strvec[0]);
        locate_main[lm_size].py = std::stod(strvec[1]);
        // std::cout << locate_main[lm_size].px << " x " << locate_main[lm_size].py << std::endl;
        lm_size++;
    }
    while (getline(jfs, line))
    {

        std::vector<std::string> strvec2 = split(line, ',');

        locate_tmp.resize(lt_size + 1);
        locate_tmp[lt_size].px = std::stod(strvec2[0]);
        locate_tmp[lt_size].py = std::stod(strvec2[1]);
        // std::cout << locate_tmp[lt_size].px << " x " << locate_tmp[lt_size].py << std::endl;
        lt_size++;
    }
    ///*
    std::string location_path;
    location_path = "gra/r_file/location_LBP.csv";
    //std::ofstream outputfile(location_path, std::ios::app);
    std::ofstream ofs("logs/location.txt");
    for (int i = load_num; i < 4546; i++)
    {
        std::cout << "start positioning " << i << " times" << std::endl;
        path = make_tpath(dir, 0, i, tag);
        // std::cout << path << std::endl;
        img = cv::imread(path, 1);
        cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
        cv::resize(src, src, cv::Size(), 0.1, 0.1);

        clock_t begin = clock();
        now_locate = position_Check(src, path, i);
        std::cout << now_locate << std::endl;
        ofs << now_locate <<std::endl;
        clock_t end = clock();
        print_elapsed_time(begin, end);

        // time = 200000 - ((end - begin) / CLOCKS_PER_SEC) * 1000000;

        //dst = cv::imread(make_tpath(dir, 2, now_locate, tag), 1);
        // locate_image = cv::imread(make_path(dir, 2, now_locate, tag), 0);
        // make_locate_image(img, dst, locate_image);
        // cv::imshow("get image", img);
        //cv::resize(img, srcr, cv::Size(), 0.5, 0.5);
        //cv::resize(dst, dstr, cv::Size(), 0.5, 0.5);
        //img_lst[0] = srcr.clone();
        //img_lst[1] = dstr.clone();
        //std::cout << img_lst[0].size << " " << img_lst[1].size << std::endl;
        //cv::hconcat(img_lst, 2, locate_image);
        // std::cout << "show locate image " << locate_image.size << std::endl;
        // cv::imshow("locate image", locate_image);
        //cv::imwrite(make_tpath(rdir, 0, i, tag), locate_image);
        //outputfile << locate_tmp[now_locate].px << "," << locate_tmp[now_locate].py << "," << now_locate << std::endl;
        //  usleep(10000000);
        //  cv::waitKey(0);
       //save(i, execute_num);
    }
    ofs.close();
    //outputfile.close();
    //*/
    return 0;
}

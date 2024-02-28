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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char **argv)
{
    cv::Mat temp = cv::imread(argv[1], 0);
    cv::Mat origin = cv::imread(argv[2], 0);
    int mode = atoi(argv[3]); // 0 monoral 1 stereo

    int SAD = 0;
    int count = 0;
    double match_point = 0;

    if (mode == 0)
    {
        for (int x = 0; x < temp.cols; x++)
        {
            for (int y = 0; y < temp.rows; y++)
            {
                SAD += abs(temp.at<unsigned char>(y, x) - origin.at<unsigned char>(y, x));
            }
        }
    }
    else if (mode == 1)
    {
        for (int x = 0; x < temp.cols; x++)
        {
            for (int y = 0; y < temp.rows; y++)
            {
                if (temp.at<unsigned char>(y, x) != 0)
                {
                    SAD += abs(temp.at<unsigned char>(y, x) - origin.at<unsigned char>(y, x));
                }else{
                    count++;
                }
            }
        }
    }

    std::cout << SAD << " " << temp.cols * temp.rows * 255 - count * 255 << std ::endl;

    match_point = (1.0 - ((double)SAD/(double)(temp.cols*temp.rows*255 - count * 255))) * 100;

    std::cout << match_point << std::endl;

}
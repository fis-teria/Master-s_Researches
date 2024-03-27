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

#ifdef __SSE2__
#include <emmintrin.h>
#else
#include <arm_neon.h>
#endif

void sse2_func()
{
    cv::Mat left = cv::imread("images/result/left.jpg", 0);
    cv::Mat right = cv::imread("images/result/right.jpg", 0);
    unsigned char src[9] = {0};
    unsigned char rrc[9] = {0};
    int x, y;
    x = 0;
    y = 0;
    
    for (int i = 0; i < 3; i++)
    {
        src[3 * i] = left.at<unsigned char>(y, x + i);
        src[3 * i + 1] = left.at<unsigned char>(y + 1, x + i);
        src[3 * i + 2] = left.at<unsigned char>(y + 2, x + i);
    }

    int st = x;
    int lim = 300;
    if (x + lim > left.cols)
    {
        st = left.cols - lim;
    }
    for (int add = st; add < x + lim; add++)
    {
        for (int i = 0; i < 3; i++)
        {
            rrc[3 * i] = right.at<unsigned char>(y, x + i);
            rrc[3 * i + 1] = right.at<unsigned char>(y + 1, x + i);
            rrc[3 * i + 2] = right.at<unsigned char>(y + 2, x + i);
        }
        for (int i = 0; i < sizeof(unsigned char) * 9; i += 16)
        {
            std::cout << "calicurate simd" << std::endl;
            __m128i row01Reg = _mm_load_si128((__m128i *)(src + i));
            __m128i row02Reg = _mm_load_si128((__m128i *)(rrc + i));
        }
    }
}

void neon_func()
{
}

int main()
{

#ifdef __SSE2__
    std::cout << "This CPU follow SSE2" << std::endl;
    sse2_func();
#else
    std::cout << "This CPU don't follow SSE2" neon_func();
#endif

    return 0;
}
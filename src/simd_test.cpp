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
    int WIN_SIZE = 5;
    cv::Mat left = cv::imread("images/result/left.jpg", 0);
    cv::Mat right = cv::imread("images/result/right.jpg", 0);
    unsigned char src[16 * ((WIN_SIZE * WIN_SIZE) / 16 + 1)] = {0};
    unsigned char rrc[16 * ((WIN_SIZE * WIN_SIZE) / 16 + 1)] = {0};
    unsigned char mem[16] = {0};

    int x, y;
    x = 0;
    y = 0;

    int k = 0;
    for (int i = 0; i < WIN_SIZE; i++)
    {
        for (int j = 0; j < WIN_SIZE; j++)
        {
            src[k] = left.at<unsigned char>(j, i);
            k++;
        }
    }
    k = 0;

    for (int i = 0; i < sizeof(src); i++)
    {
        std::cout << (int)src[i] << " ";
        
        if(i % 4 == 3)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    int st = x;
    int lim = 10;
    if (x + lim > left.cols)
    {
        st = left.cols - lim;
    }
    for (int add = st; add < x + lim; add++)
    {
        for (int i = 0; i < WIN_SIZE; i++)
        {
            for (int j = 0; j < WIN_SIZE; j++)
            {
                rrc[k] = right.at<unsigned char>(j, i);
                k++;
            }
        }
        k = 0;

        std::cout << "calicurate simd " << add << std::endl;
        int a[2] = {0};
        int store = 0;

        //SSE2 SAD CALICURATION
        for (int i = 0; i < sizeof(src); i += 8)
        {
            // 128bitのメモリに8bit毎にunsigned char配列の15~0番目が格納される
            __m128i row01Reg = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, src[i + 7], src[i + 6], src[i + 5], src[i + 4], src[i + 3], src[i + 2], src[i + 1], src[i]);
            __m128i row02Reg = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, rrc[i + 7], rrc[i + 6], rrc[i + 5], rrc[i + 4], rrc[i + 3], rrc[i + 2], rrc[i + 1], rrc[i]);

            __m128i row03Reg = _mm_load_si128((__m128i *)(mem));

            // SSE2のSADは8bitごとに差の絶対値を求めてくれるが合計してくれるのは0~63bitまでだから、
            // 別で9番目の値のみを取り出したRegを用意する必要がある。
            __m128i sad = _mm_sad_epu8(row01Reg, row02Reg);

            __m128i adder = _mm_add_epi8(row01Reg, row03Reg);

            _mm_store_si128((__m128i *)(a), sad);
            _mm_store_si128((__m128i *)(mem), row01Reg);

            for(int j = 0; j < sizeof(mem);j++)
                std::cout << (int)mem[j] << " ";
            std::cout << std::endl;
            store += a[0];
        }


        int sum = 0;
        for (int i = 0; i < sizeof(src); i++)
        {
            sum += abs(src[i] - rrc[i]);
        }

        std::cout << "rrc \t\t mem \n";
        for (int i = 0; i < sizeof(rrc); i+=4)
        {
            std::cout << (int)rrc[i] << " " << (int)rrc[i + 1] << " " << (int)rrc[i + 2] << " " << (int)rrc[i + 3] << "\t";
            std::cout << (int)mem[i] << " " << (int)mem[i + 1] << " " << (int)mem[i + 2] << " " << (int)mem[i + 3] << std::endl;
        }
        std::cout << "sse2 : normal = " << store << " " << sum << std::endl;
        std::cout << std::endl;
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
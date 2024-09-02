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

std::string make_path(std::string dir, int var, std::string tag)
{
    std::string back;
    if (var < 10)
    {
        back = dir + "/00000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 100)
    {
        back = dir + "/0000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 1000)
    {
        back = dir + "/000" + std::to_string(var) + tag;
        return back;
    }
    else if (var < 10000)
    {
        back = dir + "/00" + std::to_string(var) + tag;
        return back;
    }
}

int main(int argc, char **argv)
{
    // 動画ファイルを取り込むためのオブジェクトを宣言する
    if(argc != 3){
        std::cout << "ファイル名と出力画像を入れるフォルダ名を書いてください\n(exe\n./mov2img douga/douga.mp4 gazou" << std::endl;
        return 0;
    }
    cv::VideoCapture cap;
    std::string filename = argv[1];
    std::string output = argv[2];
    std::string dst;

    // 動画ファイルを開く
    cap.open(filename);
    if (cap.isOpened() == false)
    {
        // 動画ファイルが開けなかったときは終了する
        return 0;
    }

    // 画像を格納するオブジェクトを宣言する
    cv::Mat frame;
    int count = 0;
    std::string tag = ".jpg";

    for (;;)
    {
        // cap から frame へ1フレームを取り込む
        cap >> frame;

        if (frame.empty() == true)
        {
            // 画像が読み込めなかったとき、無限ループを抜ける
            break;
        }

        dst = make_path(output, count, tag);

        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        cv::imwrite(dst, frame);

        // ウィンドウに画像を表示する
        cv::imshow("再生中", frame);

        // 30ミリ秒待つ
        // # waitKeyの引数にキー入力の待ち時間を指定できる（ミリ秒単位）
        // # 引数が 0 または何も書かない場合は、キー入力があるまで待ち続ける
        cv::waitKey(30);
        std::cout << dst + " now comvert" << std::endl;
        count++;
    }

    return 0;
}
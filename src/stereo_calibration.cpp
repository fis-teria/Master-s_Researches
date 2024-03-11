#include <opencv2/opencv.hpp>
#include <opencv2/core/persistence.hpp>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
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

/*
    test data
    images/20231231/left/004567.jpg
    images/20240220/left/000036.jpg
*/
const std::string LEFT_IMG = "images/20231231/left/004222.jpg";
const std::string RIGHT_IMG = "images/20231231/right/004222.jpg";

// キャリブレーションから得られた外部関数を読み込むクラス
class readXml
{
public:
    cv::Mat camera_matrix, distcoeffs, extrinsic_param, img_points;
    std::string FILE_NAME;

public:
    readXml()
    {
    }

public:
    readXml(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        fs["extrinsic_parameters"] >> extrinsic_param;
        fs["image_points"] >> img_points;

        std::cout << "constractor" << std::endl;
    }

public:
    readXml(const readXml &xml)
    {
        std::cout << "copy constractor" << std::endl;
    }

public:
    void read(std::string f_name)
    {
        cv::FileStorage fs(f_name, cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distcoeffs;
        fs["extrinsic_parameters"] >> extrinsic_param;
        fs["image_points"] >> img_points;

        std::cout << "member function" << std::endl;
        return;
    }
};

int main(){
    readXml xml00 = readXml("camera/out_camera_data00.xml");//left
    readXml xml02 = readXml("camera/out_camera_data02.xml");//right

    cv::Mat left = cv::imread(LEFT_IMG, 0);
    cv::Mat right = cv::imread(RIGHT_IMG, 0);
    if (left.empty() == 1)
    {
        return 0;
    }
    if (right.empty() == 1)
    {
        return 0;
    }

    std::vector<std::vector<cv::Point3f>> worldPoints;
    cv::Mat R, T, E, F, perViewError;
    cv::TermCriteria criteria{10000, 10000, 0.0001};
    double Re_Projection_Error = 0;
    Re_Projection_Error = cv::stereoCalibrate(worldPoints, xml00.img_points, xml02.img_points,
                        xml00.camera_matrix, xml00.distcoeffs, xml02.camera_matrix, xml02.distcoeffs,
                        left.size(), R, T, E, F, perViewError, cv::CALIB_FIX_INTRINSIC, criteria);

    // 外部パラメータを書き込む
    cv::Mat ros;
    cv::Rodrigues(R, ros);
    /*
    cv::FileStorage fs("camera/stereo_camera.xml", cv::FileStorage::WRITE);

    time_t tm;
    time(&tm);
    struct tm *t2 = localtime(&tm);
    char buf[1024];
    strftime(buf, sizeof(buf), "%c", t2);

    fs << "calibration_time" << buf;
    fs << "worldPoint" << worldPoints;
    fs << "xml00_Images_Points" << xml00.img_points;
    fs << "xml02_Images_Points" << xml02.img_points;
    fs << "xml00_Camera_Matrix" << xml00.camera_matrix;
    fs << "xml00_Dist_Coeffs" << xml00.distcoeffs;
    fs << "xml02_Camera_Matrix" << xml02.camera_matrix;
    fs << "xml02_Dist_Coeffs" << xml02.distcoeffs;
    fs << "Images_Size" << left.size();
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "perViewError" << perViewError;
    fs << "Criteria" << criteria;
    fs << "RE_Projection_Error" << Re_Projection_Error;

    //*/
    return 0;
}
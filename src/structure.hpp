#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP
#include<opencv2/opencv.hpp>

struct ResultImage{
    cv::Mat image;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    int id; 
};

#endif
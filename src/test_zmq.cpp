#include <opencv2/opencv.hpp>
#include "common.hpp"


int main(){
    cv::Mat img = cv::imread("../data/images/amalab/lab_root_1/000600.jpg", 1);
    std::cout << "img serve" << std::endl;
    common::zmq_serve(img, "color");
    std::cout << "img serve complete" << std::endl;

    std::cout << "loop n recive" << std::endl;
    common::zmq_n_recive();
    std::cout << "loop n recive complete" << std::endl;

    return 0;

}
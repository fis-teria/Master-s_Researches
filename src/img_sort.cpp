#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "common.hpp"

int main(int argc, char**argv){
    if(argc < 4){
        std::cout << "./img_sort begin_num end_num parent_path child_dir_name" << std::endl;
        std::cout << "parent_path : images/parent/path/" << std::endl;
        std::cout << "child_dir : images/parent/path/child_dir/" << std::endl;
        return -1;
    }
    int begin = atoi(argv[1]);
    int end = atoi(argv[2]);
    std::string parent_path = argv[3];
    std::string child_dir = argv[4];
    std::string color_dir = parent_path + "color";
    std::string depth_dir = parent_path + "depth";
    std::string edge_dir = parent_path + "edge";
    std::string dst_dir = parent_path + "dst";
    
    std::string ccolor_dir = parent_path + child_dir + "color";
    std::string cdepth_dir = parent_path + child_dir + "depth";
    std::string cedge_dir = parent_path + child_dir + "edge";
    std::string cdst_dir = parent_path + child_dir + "dst";

    for(int i = 0; i < end - begin; i++){
        cv::Mat img_color = cv::imread(common::make_path(color_dir, begin + i, ".jpg"), 1);
        cv::Mat img_depth = cv::imread(common::make_path(depth_dir, begin + i, ".jpg"), 1);
        cv::Mat img_edge = cv::imread(common::make_path(edge_dir, begin + i, ".jpg"), 1);
        cv::Mat img_dst = cv::imread(common::make_path(dst_dir, begin + i, ".jpg"), 1);

        cv::imwrite(common::make_path(ccolor_dir, i, ".jpg"), img_color);
        cv::imwrite(common::make_path(cdepth_dir, i, ".jpg"), img_depth);
        cv::imwrite(common::make_path(cedge_dir, i, ".jpg"), img_edge);
        cv::imwrite(common::make_path(cdst_dir, i, ".jpg"), img_dst);
    }
    return 0;
}

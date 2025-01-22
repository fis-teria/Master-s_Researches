#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"

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

class Matches_Point{
    public:
    int query_x;
    int query_y;
    int map_x;
    int map_y;

    Matches_Point(int qx, int qy, int mx, int my){
        this->query_x = qx;
        this->query_y = qy;
        this->map_x = mx;
        this->map_y = my;
    }

};

int main(int argc, char** argv){
    std::string query_path = argv[1];
    std::string map_path = argv[2];

    cv::Mat query = cv::imread(query_path, 1);
    cv::Mat map = cv::imread(map_path, 1);

    float query_pt[3];
    float map_pt[3];

    std::vector<Matches_Point> match_pt;
    match_pt.push_back(Matches_Point(367, 302, 361, 306));
    match_pt.push_back(Matches_Point(133, 89, 116, 91));
    match_pt.push_back(Matches_Point(691, 338, 683, 340));

    std::cout << "start" << std::endl;
    std::cout << "query 3d points" << std::endl;
    common::doDeprojectPosition(query, match_pt[0].query_x, match_pt[0].query_y, query_pt);
    common::doDeprojectPosition(query, match_pt[1].query_x, match_pt[2].query_y, query_pt);
    common::doDeprojectPosition(query, match_pt[2].query_x, match_pt[1].query_y, query_pt);
    std::cout << "map 3d points" << std::endl;
    common::doDeprojectPosition(map, match_pt[0].map_x, match_pt[0].map_y, map_pt);
    common::doDeprojectPosition(map, match_pt[1].map_x, match_pt[2].map_y, map_pt);
    common::doDeprojectPosition(map, match_pt[2].map_x, match_pt[1].map_y, map_pt);
    std::cout << "finish" << std::endl;
    return 0;

}
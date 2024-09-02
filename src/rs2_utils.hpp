#ifndef INCLUDE_RS2UTILS_HPP
#define INCLUDE_RS2UTILS_HPP
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <librealsense2/hpp/rs_device.hpp>

// リアルセンスの基本機能をまとめたクラス
class rs2_utils
{
private:
    int WIDTH;
    int HEIGHT;
    int FPS;
    int DEPTH_WIDTH;
    int DEPTH_HEIGHT;
    int DEPTH_FPS;
    int enable_color, enable_depth, enable_gyro, enable_accel;

    rs2::config config;
    rs2::pipeline pipe;
    rs2::colorizer color_map;
    rs2::device device;

public:
    cv::Mat color_image;
    cv::Mat depth_image;
    rs2_vector gyro;
    rs2_vector accel;

    rs2_utils()
    {
        initialize();
    }

    void info()
    {
        std::cout << device.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        std::cout << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
        std::cout << device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
    }

    void initialize()
    {
        std::cout << "realsense setup" << std::endl;
        cv::FileStorage fs("../data/realsense_config/config.xml", cv::FileStorage::READ);
        fs["image_width"] >> WIDTH;
        fs["image_height"] >> HEIGHT;
        fs["image_fps"] >> FPS;
        fs["depth_width"] >> DEPTH_WIDTH;
        fs["depth_height"] >> DEPTH_HEIGHT;
        fs["depth_fps"] >> DEPTH_FPS;
        fs["enable_color"] >> enable_color;
        fs["enable_depth"] >> enable_depth;
        fs["enable_gyro"] >> enable_gyro;
        fs["enable_accel"] >> enable_accel;
        std::cout << "config images";
        if (enable_color == 1)
            config.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
        if (enable_depth == 1)
            config.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);
        if (enable_gyro == 1)
            config.enable_stream(RS2_STREAM_GYRO);
        if (enable_accel == 1)
            config.enable_stream(RS2_STREAM_ACCEL);
        std::cout << "\tOK" << std::endl;
    }

    void pipe_start()
    {
        std::cout << "pipe start \t";
        this->pipe.start(this->config);
        std::cout << "OK" << std::endl;
    }

    void test_get_frames()
    {
        std::cout << "start up check";
        for (int i = 0; i < 3; i++)
        {
            rs2::frameset test_frames = pipe.wait_for_frames();
            cv::waitKey(10);
        }
        std::cout << "\tOK" << std::endl;
    }

    void get_frames()
    {
        rs2::align align(RS2_STREAM_COLOR);
        rs2::frameset frames = this->pipe.wait_for_frames();
        auto aligned_frames = align.process(frames);

        rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
        rs2::video_frame depth_frame = aligned_frames.get_depth_frame().apply_filter(color_map);

        color_image = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        depth_image = cv::Mat(cv::Size(WIDTH, HEIGHT), CV_8UC3, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        if (rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL))
        {
            accel = accel_frame.get_motion_data();
        }

        if (rs2::motion_frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO))
        {
            gyro = gyro_frame.get_motion_data();
        }
    }
};
#endif
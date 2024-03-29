cmake_minimum_required(VERSION 3.0)
project(Master-s_Rsearches VERSION 1.1.0 LANGUAGES CXX)
# ビルドフラグの設定
find_package( OpenMP )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FRAGS "-lstdc++fs -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_CXX_FLAGS "-O2 -Wall")

# インクルードパスの追加

#add include Path
set(WIN_INCLUDE
    C:/local/include
    C:/Python310/include
)
set(WIN_LIB
    C:/local/lib
    C:/Python310/libs
)

if(WIN32)
  include_directories(${WIN_INCLUDE})
  link_directories(${WIN_LIB})
else()
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()
# ビルドに関する設定
set(EXECUTABLES "positioning_MT;match_Tsukuba_LBP;hog_detector;test;sub_image;stereo_camera;calibration;stereoGUI;belief_propagation;attach_img2mp4;test_pm;template_match;camera_check;stereo_calibration;simd_test")
set(DEPENDENT_LIBS opencv_core;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_videoio;opencv_objdetect;opencv_calib3d;opencv_features2d;pthread)
foreach(target IN LISTS EXECUTABLES)
  add_executable(${target} src/${target}.cpp)
  target_link_libraries(${target} ${DEPENDENT_LIBS})
endforeach()

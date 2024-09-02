cmake_minimum_required(VERSION 3.0)
project(Master-s_Rsearches VERSION 1.1.0 LANGUAGES CXX)
# ビルドフラグの設定
find_package( OpenMP )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-lstdc++fs -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-DPYTHON3_EXECUTABLE=/home/isight/.pyenv/versions/3.8.19/bin/python3")
set(CMAKE_CXX_FLAGS "-DPYTHON3_INCLUDE_DIR=/home/isight/.pyenv/versions/3.8.19/include/")
set(CMAKE_CXX_FLAGS "-DPYTHON3_LIBRARY=/home/isight/.pyenv/versions/3.8.19/lib/")
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

set(LINUX_INCLUDE
	/usr/local/include
	/home/isight/.pyenv/versions/3.8.19/include/python3.8
)
set(LINUX_LIB
	/usr/local/lib
	/home/isight/.pyenv/versions/3.8.19/lib
)



#ライブラリの追加



if(WIN32)
  include_directories(${WIN_INCLUDE})
  link_directories(${WIN_LIB})
else()
	include_directories(${LINUX_INCLUDE})
	link_directories(${LINUX_LIB})
endif()
# ビルドに関する設定
set(EXECUTABLES "realsense;test;eraser;img2cvt;img2mp4;attach_img2mp4;mv2img;embeded_python;localization_main")
set(DEPENDENT_LIBS opencv_core;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_videoio;opencv_objdetect;opencv_calib3d;opencv_features2d;pthread;realsense2;python3.8;zmq)
foreach(target IN LISTS EXECUTABLES)
  add_executable(${target} src/${target}.cpp)
  target_link_libraries(${target} ${DEPENDENT_LIBS})
endforeach()

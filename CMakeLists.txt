cmake_minimum_required(VERSION 3.0)
project(Guraduation_Resarch VERSION 1.1.0 LANGUAGES CXX)
# ビルドフラグの設定
find_package( OpenMP )

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# インクルードパスの追加
if(WIN32)
  include_directories(C:/local/include)
  link_directories(C:/local/lib)
else()
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()
# ビルドに関する設定
set(EXECUTABLES "positioning_MT;match_Tsukuba_LBP;hog_detector;test")
set(DEPENDENT_LIBS opencv_core;opencv_highgui;opencv_imgcodecs;opencv_imgproc;exiv2;opencv_videoio;opencv_objdetect)
foreach(target IN LISTS EXECUTABLES)
  add_executable(${target} src/${target}.cpp)
  target_link_libraries(${target} ${DEPENDENT_LIBS})
endforeach()
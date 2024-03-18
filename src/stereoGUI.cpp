#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
 
// initialize values for StereoSGBM parameters
int numDisparities = 8;
int blockSize = 5;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;
int dispType = CV_16S;
 
// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
 
cv::Mat imgL;
cv::Mat imgR;
cv::Mat imgL_gray;
cv::Mat imgR_gray;

int LBP_filter[3][3] = {{64, 32, 16},
                        {128, 0, 8},
                        {1, 2, 4}};

void cvt_ILBP(const cv::Mat &src, cv::Mat &dst)
{
  cv::Mat lbp = cv::Mat(src.rows, src.cols, CV_8UC1);
  lbp = cv::Scalar::all(0);
  cv::Mat padsrc;
  // cv::cvtColor(padsrc, padsrc, cv::COLOR_BGR2GRAY);
  copyMakeBorder(src, padsrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

  // cv::imshow("first", src);
  // cv::imshow("third", lbp);

  int ave = 0;
  for (int x = 1; x < padsrc.cols - 1; x++)
  {
    for (int y = 1; y < padsrc.rows - 1; y++)
    {
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          ave += (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i);
        }
      }
      ave /= 9.3;
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          // std::cout << ave << " " << (int)padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) <<std::endl;
          if (padsrc.at<unsigned char>(y - 1 + j, x - 1 + i) >= ave)
            lbp.at<unsigned char>(y - 1, x - 1) += LBP_filter[i][j];
        }
      }
    }
  }
  dst = lbp.clone();
  // cv::imshow("second", lbp);
}

// Defining callback functions for the trackbars to update parameter values
 
static void on_trackbar1( int, void* )
{
  stereo->setNumDisparities(numDisparities*16);
  numDisparities = numDisparities*16;
}
 
static void on_trackbar2( int, void* )
{
  stereo->setBlockSize(blockSize*2+5);
  blockSize = blockSize*2+5;
}
 
static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}
 
static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}
 
static void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(preFilterCap);
}
 
static void on_trackbar6( int, void* )
{
  stereo->setTextureThreshold(textureThreshold);
}
 
static void on_trackbar7( int, void* )
{
  stereo->setUniquenessRatio(uniquenessRatio);
}
 
static void on_trackbar8( int, void* )
{
  stereo->setSpeckleRange(speckleRange);
}
 
static void on_trackbar9( int, void* )
{
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  speckleWindowSize = speckleWindowSize*2;
}
 
static void on_trackbar10( int, void* )
{
  stereo->setDisp12MaxDiff(disp12MaxDiff);
}
 
static void on_trackbar11( int, void* )
{
  stereo->setMinDisparity(minDisparity);
}
 
 
int main()
{

    
  // Initialize variables to store the maps for stereo rectification
  cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
  cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;
 
  // Reading the mapping values for stereo image rectification
  /*
  */
  cv::FileStorage cv_file2 = cv::FileStorage("data/stereo_rectify_maps.xml", cv::FileStorage::READ);
  cv_file2["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
  cv_file2["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
  cv_file2["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
  cv_file2["Right_Stereo_Map_y"] >> Right_Stereo_Map2;
  cv_file2.release();
  // Check for left and right camera IDs
  // These values can change depending on the system
  /*
  int CamL_id{2}; // Camera ID for left camera
  int CamR_id{0}; // Camera ID for right camera
 
  cv::VideoCapture camL(CamL_id), camR(CamR_id);
 
  // Check if left camera is attached
  if (!camL.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    return -1;
  }
 
  // Check if right camera is attached
  if (!camL.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    return -1;
  }
  */
  cv::Mat imgL = cv::imread("images/20231231/left/004567.jpg", 1);
  cv::Mat imgR = cv::imread("images/20231231/right/004567.jpg", 1);
  // Creating a named window to be linked to the trackbars
  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);
 
  // Creating trackbars to dynamically update the StereoBM parameters
  cv::createTrackbar("numDisparities", "disparity", &numDisparities, 18, on_trackbar1);
  cv::createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
  cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
  cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
  cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
  cv::createTrackbar("textureThreshold", "disparity", &textureThreshold, 100, on_trackbar6);
  cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar7);
  cv::createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar8);
  cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar9);
  cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar10);
  cv::createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar11);
 
  cv::Mat disp, disparity;
 
  while (true)
  {
    // Capturing and storing left and right camera images
    //camL >> imgL;
    //camR >> imgR;
 
    // Converting images to grayscale
    cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);
    cvt_ILBP(imgL_gray, imgL_gray);
    cvt_ILBP(imgR_gray, imgR_gray);
 
    // Initialize matrix for rectified stereo images
    cv::Mat Left_nice, Right_nice;
 
    // Applying stereo image rectification on the left image
    cv::remap(imgL_gray,
              Left_nice,
              Left_Stereo_Map1,
              Left_Stereo_Map2,
              cv::INTER_LANCZOS4,
              cv::BORDER_CONSTANT,
              0);
 
    // Applying stereo image rectification on the right image
    cv::remap(imgR_gray,
              Right_nice,
              Right_Stereo_Map1,
              Right_Stereo_Map2,
              cv::INTER_LANCZOS4,
              cv::BORDER_CONSTANT,
              0);
 
    // Calculating disparith using the StereoBM algorithm
    stereo->compute(Left_nice,Right_nice,disp);
 
    // NOTE: Code returns a 16bit signed single channel image,
    // CV_16S containing a disparity map scaled by 16. Hence it 
    // is essential to convert it to CV_32F and scale it down 16 times.
 
    // Converting disparity values to CV_32F from CV_16S
    disp.convertTo(disparity,CV_32F, 1.0);
 
    // Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
 
    // Displaying the disparity map
    cv::imshow("disparity",disparity);
 
    // Close window using esc key
    if (cv::waitKey(1) == 27) break;
  }
  return 0;
}
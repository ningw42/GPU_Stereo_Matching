#ifndef device_h
#define device_h

#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "opencv2\core\core.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\gpu\gpu.hpp>
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include "opencv2\core\internal.hpp"
#include "opencv2\features2d\features2d.hpp"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdio.h>

using namespace cv;

class Device
{
public:
	Size sz;
	int rows;
	int cols;
	int totalPixel;
	int minDisparity;
	int windowSize;
	int windowLength;
	int windowArea;
	float *d_x1;
	float *d_y1;
	float *d_x2;
	float *d_y2;
	uchar *d_difference;
	uchar *d_disparity;
	uchar3 *d_left;
	uchar3 *d_right;
	uchar *d_left_remapped;
	uchar *d_right_remapped;
	uchar *d_left_cvted;
	uchar *d_right_cvted;
	uchar *h_disparity;
public:
	Device(){};
	Device(Size size, int minDisp, int wsz, Mat &mx1, Mat &my1, Mat &mx2, Mat &my2);
	~Device();
	// Proxy functions for standard C++ 
	void blockMatching_gpu(Mat &h_left, Mat &h_right, Mat &h_disparity, int SADWindowSize, int searchRange);
	void remap_gpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int rows, int cols, int total, uchar *result);
	void cvtColor_gpu(uchar3 *src, uchar *dst, int rows, int cols);

	void pipeline(Mat &left, Mat &right);
};

// pre-calculate the difference
__global__ void kernalPreCal(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total);
__global__ void kernalPreCal_V2(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total);

// find the corresponding point in those two photos
__global__ void kernalFindCorr(uchar *difference, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize);

// a pair of function to get the matched position
__global__ void kernalFindAllSAD(uchar *left, uchar *right, uchar *difference, Point3i *relativeLocation, uchar *SAD_data, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowLength, int SADWinwdowSize);
__global__ void kernalFindMinSAD(uchar *SAD_data, uchar *disparity, int numberOfCols, int searchRange);

// convert color
__global__ void kernalCvtColor(uchar3 *src, uchar *dst, int rows, int cols);

// GPU remap
__global__ void kernalRemap(uchar *src, uchar *dst, float *mapx, float *mapy, int rows, int cols);
__device__ float BilinearInterpolation(uchar *src, int rows, int cols, float x, float y);
__device__ __forceinline__ uchar float2uchar(float a);

#endif device_h
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

#include "guidedFilter.cuh"

using namespace cv;

class Device
{
public:
	Size sz;
	int rows;
	int cols;
	int totalPixel;
	int numDisparity;
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
	uchar *d_filtered_disp;

	// test 
	float *ftemp1;
	float *ftemp2;
	float *fresult;
	float *h_fresult;
	uchar *utemp1;
	uchar *utemp2;
	uchar *uresult;
	uchar *h_uresult;

	guidedFilterGPU filter;
public:
	Device(){};
	Device(Size size, int numDisp, int wsz, Mat &mx1, Mat &my1, Mat &mx2, Mat &my2);
	~Device();
	// Proxy functions for standard C++ 
	void blockMatching_gpu(Mat &h_left, Mat &h_right, Mat &h_disparity, int SADWindowSize, int searchRange);
	void remap_gpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int rows, int cols, int total, uchar *result);
	void cvtColor_gpu(uchar3 *src, uchar *dst, int rows, int cols);

	void pipeline(Mat &left, Mat &right);
	void pipeline2(Mat &left, Mat &right);
};


#endif device_h
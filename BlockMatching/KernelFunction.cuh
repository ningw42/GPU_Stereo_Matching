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

__global__ void kernelBoxFilter(float *src, float *dst, int r, int c, int rows, int cols);
__global__ void kernelMul(float *first, float *second, float *result);
__global__ void kernelDivide(float *first, float *second, float *result);
__global__ void kernelSub(float *first, float *second, float *result);
__global__ void kernelAddEle(float *first, float e, float *result);
__global__ void kernelAdd(float *first, float *second, float *result);
__global__ void kernelConvertToFloat(uchar *src, float *dst);
__global__ void kernelConvertToUchar(float *src, uchar *dst);
__device__ __forceinline__ float uchar2float(uchar a);
__device__ __forceinline__ uchar float2uchar(float a);

// pre-calculate the difference
__global__ void kernelPreCal(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total);
__global__ void kernelPreCal_V2(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total);

// find the corresponding point in those two photos
__global__ void kernelFindCorr(uchar *difference, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize);
__global__ void kernelFindCorrNonPreCal(uchar *left, uchar *right, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize);

// a pair of function to get the matched position
__global__ void kernelFindAllSAD(uchar *left, uchar *right, uchar *difference, uchar *SAD_data, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowLength, int SADWinwdowSize);
__global__ void kernelFindMinSAD(uchar *SAD_data, uchar *disparity, int numberOfCols, int searchRange);

// convert color
__global__ void kernelCvtColor(uchar3 *src, uchar *dst, int rows, int cols);

// GPU remap
__global__ void kernelRemap(uchar *src, uchar *dst, float *mapx, float *mapy, int rows, int cols);
__device__ float BilinearInterpolation(uchar *src, int rows, int cols, float x, float y);
//__device__ __forceinline__ uchar float2uchar(float a);

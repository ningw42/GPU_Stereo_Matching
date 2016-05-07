#ifndef guidedfilter_gpu_h
#define guidedfilter_gpu_h

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

class guidedFilterGPU
{
public:
	guidedFilterGPU(int rows, int cols, int r, int c, float eps);
	guidedFilterGPU();
	~guidedFilterGPU();
	void filter(uchar *I, uchar *p, uchar *result);
private:
	int r;
	int c;
	float eps;
	int rows;
	int cols;
	int total;

	// temp 
	float *I;
	float *p;
	float *sq_I;
	float *sq_mean_I;
	float *mul_Ip;
	float *mul_mean_Ip_mean_p;
	float *sum_varI_eps;
	float *mul_a_meanI;

	float *mul_meana_I;
	float *result_float;

	// useful
	float *mean_I;
	float *mean_II;
	float *mean_p;
	float *var_I;
	float *mean_Ip;
	float *cov_Ip;
	float *a;
	float *b;
	float *mean_a;
	float *mean_b;
};

__global__ void kernalBoxFilter(float *src, float *dst, int r, int c, int rows, int cols);
__global__ void kernalMul(float *first, float *second, float *result);
__global__ void kernalDivide(float *first, float *second, float *result);
__global__ void kernalSub(float *first, float *second, float *result);
__global__ void kernalAddEle(float *first, float e, float *result);
__global__ void kernalAdd(float *first, float *second, float *result);
__global__ void kernalConvertToFloat(uchar *src, float *dst);
__global__ void kernalConvertToUchar(float *src, uchar *dst);
__device__ __forceinline__ float uchar2float(uchar a);
__device__ __forceinline__ uchar float2uchar(float a);

#endif guidedfilter_gpu_h
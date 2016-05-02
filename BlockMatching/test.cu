#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

using namespace std;
using namespace cv;

void GPUTest()
{
	int N = 1 << 25;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements

	clock_t begin = clock();
	saxpy <<<(N + 255) / 256, 256 >>>(N, 2.0f, d_x, d_y);
	clock_t end = clock();

	double time = double(end - begin) / CLOCKS_PER_SEC;
	cout << time << endl;


	begin = clock();
	for (size_t i = 0; i < N; i++)
	{
		y[i] = 2.0f*x[i] + y[i];
	}
	end = clock();

	time = double(end - begin) / CLOCKS_PER_SEC;
	cout << time << endl;

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	getchar();
}

int CamTest()
{
	Mat img;

	VideoCapture cam;
	cam.open(0);
	if (!cam.isOpened())
	{
		cout << "Fuck" << endl;
		return -1;
	}

	while (true)
	{
		cam >> img;
		imshow("image", img);
		waitKey(0);
	}
	return 0;
}

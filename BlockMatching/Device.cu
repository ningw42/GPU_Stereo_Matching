#include "Device.cuh"
#include "BlockMatching.h"
#include "guidedFilter.cuh"

using namespace std;
using namespace cv;

__global__ void kernalPreCal(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total)
{
	int index = threadIdx.x;
	int th = index * total;

	for (size_t i = 0; i < total; i++)
	{
		int c = i % numberOfCols - index;
		if (c < 0) continue;
		difference[i + th] = (uchar)std::abs(left[i] - right[i - index]);
	}
}

__global__ void kernalPreCal_V2(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total)
{
	int colIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int frameBias = rowIndex * numberOfCols + colIndex;
	int frameIndex = blockIdx.z;
	int index = frameIndex * total + frameBias;

	// calculate difference only if two pixels are at the same line 
	int refCol = colIndex - frameIndex;
	if (refCol >= 0)
	{
		
		//difference[index] = (uchar)__usad(left[frameBias] - right[frameBias - frameIndex]);
		difference[index] = (uchar)std::abs(left[frameBias] - right[frameBias - frameIndex]);
	}
}

__global__ void kernalFindCorr(uchar *difference, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int currentMinSAD = 50 * windowArea;
	int matchedPosDisp = 0;
	int col = threadIndex % numberOfCols;
	int row = threadIndex / numberOfCols;
	int th = 0;

	for (int _search = 0; _search < searchRange; _search++, th += total) {
		if (col + _search > numberOfCols) break;
		int SAD = 0;
		// calculate the SAD of the current disparity
		for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
		{
			for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
			{
				int _col = col + j;
				if (_col >= numberOfCols || _col < 0) continue;
				int _row = row + i;
				if (_row >= numberOfRows || _row < 0) continue;
				SAD += difference[th + threadIndex + numberOfCols * i + j];
			}
		}
		if (SAD < currentMinSAD) {
			matchedPosDisp = _search;
			currentMinSAD = SAD;
		}
	}

	disparity[threadIndex] = matchedPosDisp;
}

__global__ void kernalFindCorrNonPreCal(uchar *left, uchar *right, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize)
{
	// grid and block should be <rows, cols> respectively
	int col = threadIdx.x;
	int row = blockIdx.x;
	int threadIndex = row * blockDim.x + col;
	int currentMinSAD = 20 * windowArea;
	int matchedPosDisp = 0;

	for (int _search = 0; _search < searchRange; _search++) {
		if (col + _search > numberOfCols) break;
		int SAD = 0;
		// calculate the SAD of the current disparity
		for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
		{
			for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
			{
				int _col = col + j;
				if (_col >= numberOfCols || _col < 0) continue;
				int _row = row + i;
				if (_row >= numberOfRows || _row < 0) continue;
				int base = threadIndex + numberOfCols * i + j;
				SAD += (uchar)std::abs(left[base + _search] - right[base]);
			}
		}
		if (SAD < currentMinSAD) {
			matchedPosDisp = _search;
			currentMinSAD = SAD;
		}
	}

	disparity[threadIndex] = matchedPosDisp;
}

// a pair of function to get the matched position
__global__ void kernalFindAllSAD(uchar *left, uchar *right, uchar *difference, Point3i *relativeLocation, uchar *SAD_data, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowLength, int SADWinwdowSize)
{
	int colIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int frameBias = rowIndex * numberOfCols + colIndex;
	int frameIndex = blockIdx.z;
	int index = frameIndex * total + frameBias;

	if (colIndex + frameIndex > numberOfCols)
	{
		SAD_data[frameBias * searchRange + frameIndex] = 255;
		return;
	}

	int SAD = 0;
	int currCol, currRow;

	for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
	{
		for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
		{
			currCol = colIndex + j;
			if (currCol >= numberOfCols || currCol < 0) continue;
			currRow = rowIndex + i;
			if (currRow >= numberOfRows || currRow < 0) continue;
			SAD += difference[index + i * numberOfCols + j];
		}
	}

	SAD_data[frameBias * searchRange + frameIndex] = SAD;
}

__global__ void kernalFindMinSAD(uchar *SAD_data, uchar *disparity, int numberOfCols, int searchRange)
{
	// TO-DO: return the original index of the min element as the matched position
	int step = threadIdx.x;
	int frameBias = blockIdx.x * numberOfCols + blockIdx.y;
	int base = searchRange * frameBias;
	int matchedPos = 0;

	int index = step + base;
	for (size_t i = blockDim.x / 2; i > 0; i = i >> 1)
	{
		if (step < i)
		{
			if (SAD_data[index] < SAD_data[index + i])
				SAD_data[index] = SAD_data[index];
			else
				SAD_data[index] = SAD_data[index + i];
			// SAD_data[index] = min(SAD_data[index], SAD_data[index + i]);
		}
		__syncthreads();
	}

	if (step == 0)
	{
		disparity[frameBias] = matchedPos;
	}
}

__global__ void kernalRemap(uchar *src, uchar *dst, float *mapx, float *mapy, int rows, int cols)
{
	int index = blockIdx.x * cols + threadIdx.x;

	const float xcoo = mapx[index];
	const float ycoo = mapy[index];
	dst[index] = float2uchar(BilinearInterpolation(src, rows, cols, ycoo, xcoo));
}

__global__ void kernalCvtColor(uchar3 *src, uchar *dst, int rows, int cols)
{
	int index = blockIdx.x * cols + threadIdx.x;

	uchar3 rgb = src[index];
	float channelSum = .299f * rgb.x + .587f * rgb.y + .114f * rgb.z;
	dst[index] = float2uchar(channelSum);
}

//__device__ __forceinline__ uchar float2uchar(float a)
//{
//	unsigned int res = 0;
//	asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(a));
//	return res;
//}

__device__ float BilinearInterpolation(uchar *src, int rows, int cols, float x, float y)
{
	int x1 = floorf(x), y1 = floorf(y), x2 = x1 + 1, y2 = y1 + 1;
	if (x1 < 0 || x2 >= rows || y1 < 0 || y2 >= cols) {
		return 0;
	}

	int baseIndex = x1 * cols + y1;
	uchar Q11 = src[baseIndex], Q12 = src[baseIndex + 1], Q21 = src[baseIndex + cols], Q22 = src[baseIndex + cols + 1];

	float left = (x2 - x) * Q11 + (x - x1) * Q21;
	float right = (x2 - x) * Q12 + (x - x1) * Q22;

	float result = (y2 - y) * left + (y - y1) * right;
	return result;
}



Device::Device(Size size, int numDisp, int wsz, Mat &mx1, Mat &my1, Mat &mx2, Mat &my2)
{
	sz = size;
	windowSize = wsz;
	windowLength = 2 * wsz + 1;
	windowArea = windowLength * windowLength;
	rows = size.height;
	cols = size.width;
	totalPixel = rows * cols;
	numDisparity = numDisp;

	// allocate memory for internal production
	cudaMalloc(&d_difference, numDisparity * totalPixel * sizeof(uchar));

	cudaMalloc(&d_left, totalPixel * sizeof(uchar3));
	cudaMalloc(&d_right, totalPixel * sizeof(uchar3));
	cudaMalloc(&d_left_remapped, totalPixel * sizeof(uchar));
	cudaMalloc(&d_right_remapped, totalPixel * sizeof(uchar));
	cudaMalloc(&d_left_cvted, totalPixel * sizeof(uchar));
	cudaMalloc(&d_right_cvted, totalPixel * sizeof(uchar));

	// allocate memory for result
	cudaMalloc(&d_disparity, totalPixel * sizeof(uchar));
	cudaMalloc(&d_filtered_disp, totalPixel * sizeof(uchar));
	h_disparity = new uchar[totalPixel];

	// allocate memory for calib data
	cudaMalloc(&d_x1, totalPixel * sizeof(float));
	cudaMalloc(&d_y1, totalPixel * sizeof(float));
	cudaMalloc(&d_x2, totalPixel * sizeof(float));
	cudaMalloc(&d_y2, totalPixel * sizeof(float));

	// copy data to GPU
	cudaMemcpy(d_x1, mx1.data, totalPixel * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, my1.data, totalPixel * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, mx2.data, totalPixel * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, my2.data, totalPixel * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	filter = guidedFilterGPU(rows, cols, 2, 2, 255 * 255 * 0.02 * 0.02);
}

Device::~Device()
{
}

// proxy function
void Device::blockMatching_gpu(Mat &h_left, Mat &h_right, Mat &h_disparity, int SADWindowSize, int searchRange)
{
	uchar *d_disparity, *d_left, *d_right, *d_difference, *d_sad_data;
	uchar *h_disparity_data;
	Point3i *h_relativeLocation, *d_relativeLocation;
	int cols = h_left.cols;
	int rows = h_left.rows;
	int total = cols * rows;
	int windowLength = 2 * SADWindowSize + 1;
	int windowArea = windowLength * windowLength;

	// malloc data
	h_disparity_data = new uchar[total];
	h_relativeLocation = new Point3i[windowArea];
	cudaMalloc(&d_sad_data, total * searchRange * sizeof(uchar));
	cudaMalloc(&d_relativeLocation, windowArea * sizeof(Point3i));
	cudaMalloc(&d_left, total * sizeof(uchar));
	cudaMalloc(&d_right, total * sizeof(uchar));
	cudaMalloc(&d_disparity, total * sizeof(uchar));
	cudaMemset(d_disparity, 0, total * sizeof(uchar));
	cudaMalloc(&d_difference, searchRange * total * sizeof(uchar));
	cudaMemset(d_difference, 0, searchRange * total * sizeof(uchar));

	// clock_t start, end;
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 1. upload data to GPU memory
	/**************************************************************************************/
	/**************************************************************************************/
	cudaEventRecord(start, 0);
	cudaMemcpy(d_left, h_left.data, total * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, h_right.data, total * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "upload data : " << elapsedTime << endl;
	/**************************************************************************************/
	/**************************************************************************************/


	// 2. pre-calculate difference
	/**************************************************************************************/
	/**************************************************************************************/
	cudaEventRecord(start, 0);
	// naive pre-calculation
	// kernalPreCal << <1, searchRange >> >(d_left, d_right, d_difference, cols, rows, total);

	// optimized pre-calculation
	dim3 block = dim3(32, 32, 1);
	dim3 grid = dim3(8, 10, searchRange);
	kernalPreCal_V2 << <grid, block >> >(d_left, d_right, d_difference, cols, rows, total);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "pre calculation : " << elapsedTime << endl;

	// DEBUG : compare the difference to CPU's result
	// uchar *result = new uchar[searchRange * total];
	// cudaMemcpy(result, d_difference, searchRange * total * sizeof(uchar), cudaMemcpyDeviceToHost);
	// compareDiff(h_left, h_right, result, SADWindowSize, searchRange, total);
	/**************************************************************************************/
	/**************************************************************************************/


	// 3. find correspondance
	/**************************************************************************************/
	/**************************************************************************************/
	cudaEventRecord(start, 0);
	// naive method to find correspondance
	kernalFindCorr << <rows, cols >> >(d_difference, d_disparity, cols, rows, windowArea, searchRange, total, windowLength, SADWindowSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "find corr : " << elapsedTime << endl;

	// optimized method
	/*
	dim3 resolution = dim3(rows, cols, 1);
	start = clock();
	kernalFindAllSAD << <grid, block >> >(d_left, d_right, d_difference, d_relativeLocation, d_sad_data, cols, rows, windowArea, searchRange, total, windowLength, SADWindowSize);
	cudaDeviceSynchronize();
	// uchar *h_sad_data = new uchar[total * searchRange];
	// cudaMemcpy(h_sad_data, d_sad_data, total * searchRange * sizeof(uchar), cudaMemcpyDeviceToHost);
	// compareSAD(h_left, h_right, h_sad_data, SADWindowSize, searchRange, cols, rows);
	// getAllSAD(h_left, h_right, h_sad_data, SADWindowSize, searchRange);
	// cudaMemcpy(d_sad_data, h_sad_data, total * searchRange * sizeof(uchar), cudaMemcpyHostToDevice);
	end = clock();
	cout << "find corr V2 : " << (double)(end - start) / CLOCKS_PER_SEC << endl;


	start = clock();
	kernalFindMinSAD << <resolution, searchRange >> >(d_sad_data, d_disparity, cols, searchRange);
	cudaDeviceSynchronize();
	end = clock();
	cout << "find min : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
	*/
	/**************************************************************************************/
	/**************************************************************************************/


	// 4. download data
	/**************************************************************************************/
	/**************************************************************************************/
	cudaEventRecord(start, 0);
	cudaMemcpy(h_disparity_data, d_disparity, total * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "download data : " << elapsedTime << endl;
	/**************************************************************************************/
	/**************************************************************************************/

	// DEBUG : compare the disparity to CPU's result
	// compareDisp(h_left, h_right, h_disparity_data, SADWindowSize, searchRange, cols, rows);

	// 5. return data
	h_disparity = Mat(rows, cols, CV_8UC1, h_disparity_data);
}

void Device::remap_gpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int rows, int cols, int total, uchar *result)
{
	uchar *d_left_gpu_data, *d_right_gpu_data, *d_left, *d_right;
	float *d_mapx1, *d_mapx2, *d_mapy1, *d_mapy2;
	cudaMalloc(&d_left_gpu_data, total * sizeof(uchar));
	cudaMalloc(&d_right_gpu_data, total * sizeof(uchar));
	cudaMalloc(&d_left, total * sizeof(uchar));
	cudaMalloc(&d_right, total * sizeof(uchar));
	cudaMalloc(&d_mapx1, total * sizeof(float));
	cudaMalloc(&d_mapx2, total * sizeof(float));
	cudaMalloc(&d_mapy1, total * sizeof(float));
	cudaMalloc(&d_mapy2, total * sizeof(float));

	cudaMemcpy(d_mapx1, mapX1.data, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mapx2, mapX2.data, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mapy1, mapY1.data, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mapy2, mapY2.data, total * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_left, left.data, total * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right.data, total * sizeof(uchar), cudaMemcpyHostToDevice);

	
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernalRemap << <rows, cols >> >(d_left, d_left_gpu_data, d_mapx1, d_mapy1, rows, cols);
	kernalRemap << <rows, cols >> >(d_right, d_right_gpu_data, d_mapx2, d_mapy2, rows, cols);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "GPU Remap : " << elapsedTime << endl;

	cudaMemcpy(result, d_left_gpu_data, total * sizeof(uchar), cudaMemcpyDeviceToHost);
}

void Device::cvtColor_gpu(uchar3 *src, uchar *dst, int rows, int cols)
{
	uchar3 *d_src;
	uchar *d_dst;
	int total = rows * cols;
	cudaEvent_t start, end;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMalloc(&d_src, total * sizeof(uchar3));
	cudaMalloc(&d_dst, total * sizeof(uchar));
	cudaMemcpy(d_src, src, total * sizeof(uchar3), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	for (size_t i = 0; i < 1000; i++)
		kernalCvtColor << <rows, cols >> >(d_src, d_dst, rows, cols);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cout << "GPU cvtColor : " << time << endl;

	cudaMemcpy(dst, d_dst, total * sizeof(uchar), cudaMemcpyDeviceToHost);
}

void Device::pipeline(Mat &left, Mat &right)
{
	// resize
	resize(left, left, sz);
	resize(right, right, sz);
	imshow("Left", left);
	imshow("Right", right);

	// upload data
	cudaMemcpy(d_left, left.data, totalPixel * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right.data, totalPixel * sizeof(uchar3), cudaMemcpyHostToDevice);

	// convert color
	kernalCvtColor << <rows, cols >> >(d_left, d_left_cvted, rows, cols);
	kernalCvtColor << <rows, cols >> >(d_right, d_right_cvted, rows, cols);

	// remap
	kernalRemap << <rows, cols >> >(d_left_cvted, d_left_remapped, d_x1, d_y1, rows, cols);
	kernalRemap << <rows, cols >> >(d_right_cvted, d_right_remapped, d_x2, d_y2, rows, cols);

	// stereo matching
	dim3 block = dim3(24, 32, 1);
	dim3 grid = dim3(10, 10, numDisparity);
	kernalPreCal_V2 << <grid, block >> >(d_left_remapped, d_right_remapped, d_difference, cols, rows, totalPixel);
	kernalFindCorr << <rows, cols >> >(d_difference, d_disparity, cols, rows, windowArea, numDisparity, totalPixel, windowLength, windowSize);

	// download data
	//cudaMemcpy(h_disparity, d_disparity, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	//imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));

	filter.filter(d_left_remapped, d_disparity, d_filtered_disp);
	cudaMemcpy(h_disparity, d_filtered_disp, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));
}

void Device::pipeline2(Mat &left, Mat &right)
{
	// resize
	resize(left, left, sz);
	resize(right, right, sz);
	imshow("Left", left);
	imshow("Right", right);

	// upload data
	cudaMemcpy(d_left, left.data, totalPixel * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right.data, totalPixel * sizeof(uchar3), cudaMemcpyHostToDevice);

	// convert color
	kernalCvtColor << <rows, cols >> >(d_left, d_left_cvted, rows, cols);
	kernalCvtColor << <rows, cols >> >(d_right, d_right_cvted, rows, cols);

	// remap
	kernalRemap << <rows, cols >> >(d_left_cvted, d_left_remapped, d_x1, d_y1, rows, cols);
	kernalRemap << <rows, cols >> >(d_right_cvted, d_right_remapped, d_x2, d_y2, rows, cols);

	// stereo matching
	//dim3 block = dim3(24, 32, 1);
	//dim3 grid = dim3(10, 10, numDisparity);
	//kernalPreCal_V2 << <grid, block >> >(d_left_remapped, d_right_remapped, d_difference, cols, rows, totalPixel);
	kernalFindCorrNonPreCal << <rows, cols >> >(d_left_remapped, d_right_remapped, d_disparity, cols, rows, windowArea, numDisparity, totalPixel, windowLength, windowSize);

	// download data
	cudaMemcpy(h_disparity, d_disparity, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));
}
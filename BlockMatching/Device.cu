#include "Device.cuh"
#include "BlockMatching.h"
#include "guidedfilter.h"
#include "KernelFunction.cuh"

using namespace std;
using namespace cv;




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

	// test
	cudaMalloc(&ftemp1, totalPixel * sizeof(float));
	cudaMalloc(&ftemp2, totalPixel * sizeof(float));
	cudaMalloc(&fresult, totalPixel * sizeof(float));
	h_fresult = new float[totalPixel];
	cudaMalloc(&utemp1, totalPixel * sizeof(uchar));
	cudaMalloc(&utemp2, totalPixel * sizeof(uchar));
	cudaMalloc(&uresult, totalPixel * sizeof(uchar));
	h_uresult = new uchar[totalPixel];
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
	// kernelPreCal << <1, searchRange >> >(d_left, d_right, d_difference, cols, rows, total);

	// optimized pre-calculation
	dim3 block = dim3(32, 32, 1);
	dim3 grid = dim3(8, 10, searchRange);
	kernelPreCal_V2 << <grid, block >> >(d_left, d_right, d_difference, cols, rows, total);

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
	kernelFindCorr << <rows, cols >> >(d_difference, d_disparity, cols, rows, windowArea, searchRange, total, windowLength, SADWindowSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "find corr : " << elapsedTime << endl;

	// optimized method
	/*
	dim3 resolution = dim3(rows, cols, 1);
	start = clock();
	kernelFindAllSAD << <grid, block >> >(d_left, d_right, d_difference, d_relativeLocation, d_sad_data, cols, rows, windowArea, searchRange, total, windowLength, SADWindowSize);
	cudaDeviceSynchronize();
	// uchar *h_sad_data = new uchar[total * searchRange];
	// cudaMemcpy(h_sad_data, d_sad_data, total * searchRange * sizeof(uchar), cudaMemcpyDeviceToHost);
	// compareSAD(h_left, h_right, h_sad_data, SADWindowSize, searchRange, cols, rows);
	// getAllSAD(h_left, h_right, h_sad_data, SADWindowSize, searchRange);
	// cudaMemcpy(d_sad_data, h_sad_data, total * searchRange * sizeof(uchar), cudaMemcpyHostToDevice);
	end = clock();
	cout << "find corr V2 : " << (double)(end - start) / CLOCKS_PER_SEC << endl;


	start = clock();
	kernelFindMinSAD << <resolution, searchRange >> >(d_sad_data, d_disparity, cols, searchRange);
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

	kernelRemap << <rows, cols >> >(d_left, d_left_gpu_data, d_mapx1, d_mapy1, rows, cols);
	kernelRemap << <rows, cols >> >(d_right, d_right_gpu_data, d_mapx2, d_mapy2, rows, cols);

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
		kernelCvtColor << <rows, cols >> >(d_src, d_dst, rows, cols);
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
	kernelCvtColor << <rows, cols >> >(d_left, d_left_cvted, rows, cols);
	kernelCvtColor << <rows, cols >> >(d_right, d_right_cvted, rows, cols);

	// remap
	kernelRemap << <rows, cols >> >(d_left_cvted, d_left_remapped, d_x1, d_y1, rows, cols);
	kernelRemap << <rows, cols >> >(d_right_cvted, d_right_remapped, d_x2, d_y2, rows, cols);

	// stereo matching
	dim3 block = dim3(24, 32, 1);
	dim3 grid = dim3(10, 10, numDisparity);
	kernelPreCal_V2 << <grid, block >> >(d_left_remapped, d_right_remapped, d_difference, cols, rows, totalPixel);
	kernelFindCorr << <rows, cols >> >(d_difference, d_disparity, cols, rows, windowArea, numDisparity, totalPixel, windowLength, windowSize);

	// download data(no filter)
	//cudaMemcpy(h_disparity, d_disparity, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	//imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));

	// guided filter 
	filter.filter(d_disparity, d_disparity, d_filtered_disp);
	cudaMemcpy(h_disparity, d_filtered_disp, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));

	// test
	// gpu gilter
	//filter.filter(d_disparity, d_disparity, d_filtered_disp);
	//// cpu filter 
	//cudaMemcpy(h_disparity, d_disparity, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	//Mat cpu_result = Mat(rows, cols, CV_8UC1, h_disparity);
	//GuidedFilter gf = GuidedFilter(cpu_result, 2, 0.02 * 0.02 * 255 * 255);
	//Mat dd = gf.filter(cpu_result);
	//imshow("CPU", dd);
	//// compare the result 
	//cudaMemcpy(h_fresult, filter.result_float, totalPixel * sizeof(float), cudaMemcpyDeviceToHost);
	//imshow("GPU float", Mat(rows, cols, CV_32FC1, h_fresult));

	//cudaMemcpy(h_uresult, d_filtered_disp, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	//imshow("GPU uchar", Mat(rows, cols, CV_8UC1, h_uresult));
	//for (size_t i = 0; i < rows; i++)
	//{
	//	for (size_t j = 0; j < cols; j++)
	//	{
	//		int index = i * cols + j;
	//		if (h_fresult[index] != gf.impl_->getMat("result").ptr<float>(i)[j])
	//		{
	//			cout << "Diff : " << '[' << i << ',' << j << "]\t" << h_fresult[index] << '\t' << (float)gf.impl_->getMat("result").ptr<float>(i)[j] << endl;
	//		}
	//		else
	//		{
	//			cout << "Same : " << '[' << i << ',' << j << "]\t" << h_fresult[index] << '\t' << (float)gf.impl_->getMat("result").ptr<float>(i)[j] << endl;
	//		}
	//	}
	//}
	//imshow("convert", Mat(rows, cols, CV_32FC1, h_fresult));
	//kernelBoxFilter<<<rows, cols>>>(temp1, result, filter.r, filter.c, rows, cols);
	//cudaMemcpy(h_result, result, totalPixel * sizeof(float), cudaMemcpyDeviceToHost);
	//imshow("BoxFilter", Mat(rows, cols, CV_32FC1, h_result));
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
	kernelCvtColor << <rows, cols >> >(d_left, d_left_cvted, rows, cols);
	kernelCvtColor << <rows, cols >> >(d_right, d_right_cvted, rows, cols);

	// remap
	kernelRemap << <rows, cols >> >(d_left_cvted, d_left_remapped, d_x1, d_y1, rows, cols);
	kernelRemap << <rows, cols >> >(d_right_cvted, d_right_remapped, d_x2, d_y2, rows, cols);

	// stereo matching
	//dim3 block = dim3(24, 32, 1);
	//dim3 grid = dim3(10, 10, numDisparity);
	//kernelPreCal_V2 << <grid, block >> >(d_left_remapped, d_right_remapped, d_difference, cols, rows, totalPixel);
	kernelFindCorrNonPreCal << <rows, cols >> >(d_left_remapped, d_right_remapped, d_disparity, cols, rows, windowArea, numDisparity, totalPixel, windowLength, windowSize);

	// download data
	cudaMemcpy(h_disparity, d_disparity, totalPixel * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("Disp", Mat(rows, cols, CV_8UC1, h_disparity));
}
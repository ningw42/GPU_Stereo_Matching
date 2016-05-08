//
//  utility.cpp
//  OpenCV_Test
//
//  Created by Ning Wang on 4/25/16.
//  Copyright © 2016 Ning Wang. All rights reserved.
//

#include "Utility.h"
#include "guidedfilter.h"
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

Mat LoadData(const string& filename, const string& varName)
{
	cv::Mat result;
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs[varName] >> result;
	fs.release();
	return result;
}

void LoadDataBatch(const string& filename, OutputArray camMat1, OutputArray camMat2, OutputArray distCoe1, OutputArray distCoe2, OutputArray R, OutputArray T)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	Mat temp;
	fs["LeftMat"] >> temp;
	temp.convertTo(camMat1, CV_64F);
	fs["RightMat"] >> temp;
	temp.convertTo(camMat2, CV_64F);
	fs["LeftDist"] >> temp;
	temp.convertTo(distCoe1, CV_64F);
	fs["RightDist"] >> temp;
	temp.convertTo(distCoe2, CV_64F);
	fs["RotationVec"] >> temp;
	temp.convertTo(R, CV_64F);
	fs["TranslationVec"] >> temp;
	temp.convertTo(T, CV_64F);
	fs.release();
}

string MatType(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void BM_SBM(InputArray img1, InputArray img2)
{
	Mat g1, g2;
	Mat disp, disp8;

	imshow("left", img1);
	imshow("right", img2);


	cvtColor(img1, g1, CV_BGR2GRAY);
	cvtColor(img2, g2, CV_BGR2GRAY);

	StereoBM sbm;
	sbm.state->SADWindowSize = 9;
	sbm.state->numberOfDisparities = 112;
	sbm.state->preFilterSize = 5;
	sbm.state->preFilterCap = 61;
	sbm.state->minDisparity = -39;
	sbm.state->textureThreshold = 507;
	sbm.state->uniquenessRatio = 0;
	sbm.state->speckleWindowSize = 0;
	sbm.state->speckleRange = 8;
	sbm.state->disp12MaxDiff = 1;

	sbm(g1, g2, disp);
	normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

	imshow("disp", disp8);
}

int StereoCalib()
{
	int numBoards, board_w, board_h;
	cout << "Number of Positions = ";
	cin >> numBoards;
	cout << "Number of Coners_h= ";
	cin >> board_h;
	cout << "Number of Coners_w = ";
	cin >> board_w;

	Size board_sz = Size(board_w, board_h);
	Size targetSize = Size(320, 240);
	int board_n = board_w * board_h;

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > imagePoints_left, imagePoints_right;
	vector<Point2f> corners_left, corners_right;

	vector<Point3f> obj;
	for (int j = 0; j < board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j % board_w, 0.0f));
	}

	Mat img_left, img_right, gray_left, gray_right;
	VideoCapture cap_left = VideoCapture(0);
	VideoCapture cap_right = VideoCapture(1);

	// Switch the two cameras
	cout << "Check the Camera's Relative Position\n";
	int sw;
	while (true)
	{
		cap_left >> img_left;
		cap_right >> img_right;
		imshow("Temp_Left", img_left);
		imshow("Temp_Right", img_right);
		sw = waitKey(1);
		if (sw == ' ')
		{
			break;
		}
		else if (sw == 'p')
		{
			cap_left.release();
			cap_right.release();
			cap_left.open(1);
			cap_right.open(0);
		}
	}
	destroyAllWindows();

	// find corner points
	int success = 0, k = 0;
	bool found_left = false, found_right = false;
	while (success < numBoards)
	{
		cap_left >> img_left;
		cap_right >> img_right;
		// resize
		resize(img_left, img_left, targetSize);
		resize(img_right, img_right, targetSize);
		cvtColor(img_left, gray_left, CV_BGR2GRAY);
		cvtColor(img_right, gray_right, CV_BGR2GRAY);

		found_left = findChessboardCorners(img_left, board_sz, corners_left, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		found_right = findChessboardCorners(img_right, board_sz, corners_right, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found_left)
		{
			cornerSubPix(gray_left, corners_left, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray_left, board_sz, corners_left, found_left);
		}

		if (found_right)
		{
			cornerSubPix(gray_right, corners_right, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray_right, board_sz, corners_right, found_right);
		}

		imshow("Left", gray_left);
		imshow("Right", gray_right);

		k = waitKey(10);
		if (found_left && found_right)
		{
			k = waitKey(0);
		}
		if (k == 27)
		{
			break;
		}
		if (k == ' ' && found_left != 0 && found_right != 0)
		{
			imagePoints_left.push_back(corners_left);
			imagePoints_right.push_back(corners_right);
			object_points.push_back(obj);
			printf("Corners No.%d stored\n", ++success);

			if (success > numBoards)
			{
				break;
			}
		}
	}

	destroyAllWindows();
	printf("Starting Calibration\n");
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat D1, D2;
	Mat R, T, E, F;

	stereoCalibrate(object_points, imagePoints_left, imagePoints_right,
		CM1, D1, CM2, D2, targetSize, R, T, E, F,
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);

	FileStorage fs1("./../stereocalib.yml", FileStorage::WRITE);
	fs1 << "LeftMat" << CM1;
	fs1 << "RightMat" << CM2;
	fs1 << "LeftDist" << D1;
	fs1 << "RightDist" << D2;
	fs1 << "RotationVec" << R;
	fs1 << "TranslationVec" << T;
	fs1 << "E" << E;
	fs1 << "F" << F;

	printf("Done Calibration\n");

	printf("Starting Rectification\n");

	Mat R1, R2, P1, P2, Q;
	stereoRectify(CM1, D1, CM2, D2, targetSize, R, T, R1, R2, P1, P2, Q);
	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;
	fs1.release();

	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;
	initUndistortRectifyMap(CM1, D1, R1, P1, targetSize, CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(CM2, D2, R2, P2, targetSize, CV_32FC1, map2x, map2y);

	printf("Undistort complete\n");

	while (1)
	{
		cap_left >> img_left;
		cap_right >> img_right;

		resize(img_left, img_left, targetSize);
		resize(img_right, img_right, targetSize);

		remap(img_left, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(img_right, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

		imshow("Left", imgU1);
		imshow("Right", imgU2);

		k = waitKey(5);

		if (k == 27)
		{
			break;
		}
	}

	cap_left.release();
	cap_right.release();

	return(0);
}

void photo()
{
	VideoCapture capLeft(1);
	VideoCapture capRight(2);
	Mat leftFrame, rightFrame, originalLeft, originalRight;
	int key;
	int counter = 0;

	while (true) {
		capLeft >> originalLeft;
		capRight >> originalRight;

		resize(originalLeft, leftFrame, Size(originalLeft.cols / 3, originalLeft.rows / 3));
		resize(originalRight, rightFrame, Size(originalRight.cols / 3, originalRight.rows / 3));
		imshow("LeftCam", leftFrame);
		imshow("RightCam", rightFrame);

		key = waitKey(30);
		if (key == ' ') {
			imwrite("./Chess/Left_" + to_string(counter) + ".jpg", originalLeft);
			imwrite("./Chess/Right_" + to_string(counter) + ".jpg", originalRight);
			cout << "Pairs " + to_string(counter) + " Taken" << endl;
			counter++;
		}
		else if (key == 'q') {
			exit(0);
		}
	}
}

void Rectify(InputArray camMat1, InputArray camMat2, InputArray distCoe1, InputArray distCoe2, InputArray R, InputArray T, Size imageSize, OutputArray mapX1, OutputArray mapY1, OutputArray mapX2, OutputArray mapY2)
{
	Mat R1, R2, P1, P2, Q;
	stereoRectify(camMat1, distCoe1, camMat2, distCoe2, imageSize, R, T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY);
	initUndistortRectifyMap(camMat1, distCoe1, R1, P1, imageSize, CV_32FC1, mapX1, mapY1);
	initUndistortRectifyMap(camMat2, distCoe2, R2, P2, imageSize, CV_32FC1, mapX2, mapY2);
}

void cvtColor_impl(Mat &src, Mat &dst)
{
	int rows = src.rows, cols = src.cols;
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			uchar3 rgb = src.ptr<uchar3>(r)[c];
			float channelSum = .299f * rgb.x + .587f * rgb.y + .114f * rgb.z;
			dst.ptr<uchar>(r)[c] = (uchar)channelSum;
		}
	}
}

void remap_impl(Mat &src, Mat &dst, Mat &mapx, Mat &mapy)
{
	// dst = new uchar[src.rows * src.cols];
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			const float xcoo = mapx.ptr<float>(row)[col];
			const float ycoo = mapy.ptr<float>(row)[col];
			//dst[row * src.cols + col] = saturate_cast<uchar>(CPU_BilinearInterpolation(src, ycoo, xcoo));
			dst.ptr<uchar>(row)[col] = saturate_cast<uchar>(CPU_BilinearInterpolation(src, ycoo, xcoo));
		}
	}
}

float CPU_BilinearInterpolation(Mat &src, float x, float y)
{
	int rows = src.rows, cols = src.cols;
	int x1 = floor(x), y1 = floor(y), x2 = x1 + 1, y2 = y1 + 1;
	if (x1 < 0 || x2 >= rows || y1 < 0 || y2 >= cols) {
		return 0;
	}
	uchar *data = src.data;

	uchar Q11 = data[x1 * cols + y1], Q12 = data[x1 * cols + y2], Q21 = data[x2 * cols + y1], Q22 = data[x2 * cols + y2];

	float left = (x2 - x) * Q11 + (x - x1) * Q21;
	float right = (x2 - x) * Q12 + (x - x1) * Q22;

	float result = (y2 - y) * left + (y - y1) * right;
	return result;
}

//void remap_cpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int total, uchar *result)
//{
//	clock_t start, end;
//	uchar *left_cpu_data, *right_cpu_data;
//	Mat left_ref, right_ref;
//	left_cpu_data = new uchar[total];
//	right_cpu_data = new uchar[total];
//
//	start = clock();
//	// CPU_Remap(left, result, mapX1, mapY1);
//	CPU_Remap(left, left_cpu_data, mapX1, mapY1);
//	CPU_Remap(right, right_cpu_data, mapX2, mapY2);
//	end = clock();
//	cout << "CPU Remap : " << double(end - start) / CLOCKS_PER_SEC << endl;
//
//	//start = clock();
//	//remap(left, left_ref, mapX1, mapY1, INTER_LINEAR);
//	//remap(right, right_ref, mapX2, mapY2, INTER_LINEAR);
//	//end = clock();
//	
//	//cout << "CPU Remap : " << double(end - start) / CLOCKS_PER_SEC << endl;
//}

void getCalibResult(Size targetSize, Mat &x1, Mat &y1, Mat &x2, Mat &y2)
{
	Mat camMat1, camMat2, distCoe1, distCoe2, R, T;
	Mat mapx1, mapy1, mapx2, mapy2;

	// load calib data
	LoadDataBatch("./../stereocalib.yml", camMat1, camMat2, distCoe1, distCoe2, R, T);
	// calib
	Rectify(camMat1, camMat2, distCoe1, distCoe2, R, T, targetSize, x1, y1, x2, y2);
}

Host::Host()
{
}

Host::~Host()
{
}

Host::Host(Size size, int numDisp, int wsz, Mat &mx1, Mat &my1, Mat &mx2, Mat &my2)
{
	sz = size;
	windowSize = wsz;
	windowLength = 2 * wsz + 1;
	windowArea = windowLength * windowLength;
	rows = size.height;
	cols = size.width;
	totalPixel = rows * cols;
	numDisparity = numDisp;

	x1 = mx1;
	y1 = my1;
	x2 = mx2;
	y2 = my2;

	left_color_cvted = Mat(rows, cols, CV_8UC1);
	right_color_cvted = Mat(rows, cols, CV_8UC1);
	left_remapped = Mat(rows, cols, CV_8UC1);
	right_remapped = Mat(rows, cols, CV_8UC1);
	disparity = Mat(rows, cols, CV_8UC1);
}

void Host::blockMatching(Mat &left, Mat &right, Mat &disparity)
{
	int total = cols * rows;
	const uchar * data_left = left.ptr<uchar>(0);
	const uchar * data_right = right.ptr<uchar>(0);
	uchar * data_dm = new uchar[total];
	int windowLength = 2 * windowSize + 1;
	int windowArea = windowLength * windowLength;

	memset(data_dm, 0, total * sizeof(uchar));
	// x is col index in the windowLength * windowLength window
	// y is row index in this window
	// z is (x + y * cols).
	// I compute them in advance for avoid computing repeatedly.
	Point3i * dLocDif = new Point3i[windowArea];

	for (int i = 0; i < windowArea; i++) {
		dLocDif[i] = Point3i(i % windowLength - windowSize, i / windowLength - windowSize, 0);
		dLocDif[i].z = dLocDif[i].x + dLocDif[i].y * cols;
	}

	// I compute disparity difference for each search range to avoid
	// computing repeatedly.
	uchar * dif_ = new uchar[total * numDisparity];
	memset(dif_, 0, total * numDisparity * sizeof(uchar));
	for (int _search = 0; _search < numDisparity; _search++) {
		int th = _search * total;
		for (int i = 0; i < total; i++) {
			int c = i % cols - _search;
			if (c < 0) continue;
			dif_[i + th] = (uchar)std::abs(data_left[i] - data_right[i - _search]);
		}
	}

	for (int p = 0; p < total; p++) {
		int min = 50 * windowArea;
		int dm = -256;
		int _col = p % cols;
		int _row = p / cols;
		int th = 0;

		// I search for the smallest difference between left and right image
		// using def_.
		for (int _search = 0; _search < numDisparity; _search++, th += total) {
			if (_col + _search > cols) break;
			int temp = 0;
			for (int i = 0; i < windowArea; i++) {
				int _c = _col + dLocDif[i].x;
				if (_c >= cols || _c < 0) continue;
				int _r = _row + dLocDif[i].y;
				if (_r >= rows || _r < 0) continue;
				temp += dif_[th + p + dLocDif[i].z];
				if (temp > min) {
					break;
				}
			}
			if (temp < min) {
				dm = _search;
				min = temp;
			}
		}

		data_dm[p] = dm;
	}

	disparity = Mat(rows, cols, CV_8UC1, data_dm);
}

void Host::calculateFrame(Mat &left, Mat &right)
{
	// resize 
	resize(left, left, sz);
	resize(right, right, sz);
	//imshow("Left", left);
	//imshow("Right", right);

	Mat left_normed, right_normed;

	// convert color
	cvtColor_impl(left, left_color_cvted);
	cvtColor_impl(right, right_color_cvted);

	//normalize(left_color_cvted, left_normed, 0, 255, CV_MINMAX, CV_8UC1);
	//normalize(right_color_cvted, right_normed, 0, 255, CV_MINMAX, CV_8UC1);
	//imshow("gray_left", left_color_cvted);
	//imshow("gray_right", right_color_cvted);
	//imshow("norm_left", left_normed);
	//imshow("norm_right", right_normed);

	// remap
	remap_impl(left_color_cvted, left_remapped, x1, y1);
	remap_impl(right_color_cvted, right_remapped, x2, y2);

	// stereo matching
	blockMatching(left_remapped, right_remapped, disparity);

	int r = 2;
	double eps = 0.02 * 0.02;
	eps *= 255 * 255;
	imshow("Disp", disparity);
	clock_t start, end;
	//Mat filtered_left = guidedFilter(left_color_cvted, disparity, r, eps);
	//Mat filtered_result;
	//disparity.convertTo(filtered_result, CV_32FC1, 1.0 / 255);
	//blur(left_remapped, filtered_result, Size(r, r));
	Mat filtered_right = guidedFilter(right_color_cvted, disparity, r, eps);
	//cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
	imshow("Filtered_result", filtered_right);
	//imshow("Filtered_right", filtered_right);
	//imshow("self", guidedFilter(disparity, disparity, r, eps));
}
//
//  utility.cpp
//  OpenCV_Test
//
//  Created by Ning Wang on 4/25/16.
//  Copyright © 2016 Ning Wang. All rights reserved.
//

#include "Utility.h"
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

int CalibrationTest(char* argv[])
{
	int numBoards = atoi(argv[1]);
	int board_w = atoi(argv[2]);
	int board_h = atoi(argv[3]);

	cout << numBoards << board_h << board_w;

	Size board_sz = Size(board_w, board_h);
	int board_n = board_w*board_h;

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > image_points;
	vector<Point2f> corners;

	vector<Point3f> obj;
	for (int j = 0; j<board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j%board_w, 0.0f));
	}

	Mat img, gray;
	VideoCapture cap = VideoCapture(1);

	int success = 0;
	int k = 0;
	bool found = false;
	while (success < numBoards)
	{
		cap >> img;
		cvtColor(img, gray, CV_BGR2GRAY);
		found = findChessboardCorners(gray, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray, board_sz, corners, found);
		}

		//        imshow("image", img);
		imshow("corners", gray);

		k = waitKey(1);
		if (found)
		{
			k = waitKey(0);
		}
		if (k == 27)
		{
			break;
		}
		if (k == ' ' && found != 0)
		{
			image_points.push_back(corners);
			object_points.push_back(obj);
			printf("Corners stored\n");
			success++;

			if (success >= numBoards)
			{
				break;
			}
		}
		cout << found << ':' << success << endl;
	}
	destroyAllWindows();
	printf("Starting calibration\n");
	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distcoeffs;
	vector<Mat> rvecs, tvecs;

	intrinsic.at<float>(0, 0) = 1;
	intrinsic.at<float>(1, 1) = 1;

	calibrateCamera(object_points, image_points, img.size(), intrinsic, distcoeffs, rvecs, tvecs);

	FileStorage fs1("mycalib.yml", FileStorage::WRITE);
	fs1 << "CM1" << intrinsic;
	fs1 << "D1" << distcoeffs;

	printf("calibration done\n");

	Mat imgU;
	while (1)
	{
		cap >> img;
		undistort(img, imgU, intrinsic, distcoeffs);

		imshow("image", img);
		imshow("undistort", imgU);

		k = waitKey(5);
		if (k == 27)
		{
			break;
		}
	}
	cap.release();
	return 0;
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

void CPU_Remap(Mat &src, uchar *dst, Mat &mapx, Mat &mapy)
{
	// dst = new uchar[src.rows * src.cols];
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			const float xcoo = mapx.ptr<float>(row)[col];
			const float ycoo = mapy.ptr<float>(row)[col];
			dst[row * src.cols + col] = saturate_cast<uchar>(CPU_BilinearInterpolation(src, ycoo, xcoo));
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

void remap_cpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int total, uchar *result)
{
	clock_t start, end;
	uchar *left_cpu_data, *right_cpu_data;
	Mat left_ref, right_ref;
	left_cpu_data = new uchar[total];
	right_cpu_data = new uchar[total];

	start = clock();
	// CPU_Remap(left, result, mapX1, mapY1);
	CPU_Remap(left, left_cpu_data, mapX1, mapY1);
	CPU_Remap(right, right_cpu_data, mapX2, mapY2);
	end = clock();
	cout << "CPU Remap : " << double(end - start) / CLOCKS_PER_SEC << endl;

	//start = clock();
	//remap(left, left_ref, mapX1, mapY1, INTER_LINEAR);
	//remap(right, right_ref, mapX2, mapY2, INTER_LINEAR);
	//end = clock();
	
	//cout << "CPU Remap : " << double(end - start) / CLOCKS_PER_SEC << endl;
}

void cvtColor_cpu(uchar3 *src, uchar *dst, int rows, int cols)
{
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			uchar3 rgb = src[r * cols + c];
			float channelSum = .299f * rgb.x + .587f * rgb.y + .114f * rgb.z;
			dst[r * cols + c] = (uchar)channelSum;
		}
	}
}

void getCalibResult(Size targetSize, Mat &x1, Mat &y1, Mat &x2, Mat &y2)
{
	Mat camMat1, camMat2, distCoe1, distCoe2, R, T;
	Mat mapx1, mapy1, mapx2, mapy2;

	// load calib data
	LoadDataBatch("./../Calib_Data_OpenCV.yml", camMat1, camMat2, distCoe1, distCoe2, R, T);
	// calib
	Rectify(camMat1, camMat2, distCoe1, distCoe2, R, T, targetSize, x1, y1, x2, y2);
}
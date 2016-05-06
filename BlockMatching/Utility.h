//
//  utility.hpp
//  OpenCV_Test
//
//  Created by Ning Wang on 4/25/16.
//  Copyright © 2016 Ning Wang. All rights reserved.
//

#ifndef utility_h
#define utility_h

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <vector_types.h>


using namespace cv;

// functions for loading camera data
Mat LoadData(const string& filename, const string& varName);
void LoadDataBatch(const string& filename, OutputArray camMat1, OutputArray camMat2, OutputArray distCoe1, OutputArray distCoe2, OutputArray R, OutputArray T);

// get the data type inside a Mat
string MatType(int type);

// Block Matching wrapper
void BM_SBM(InputArray img1, InputArray img2);

// calib
int StereoCalib();

// take photo
void photo();

// rectify camera
void Rectify(InputArray camMat1, InputArray camMat2, InputArray distCoe1, InputArray distCoe2, InputArray R, InputArray T, Size imageSize, OutputArray mapX1, OutputArray mapY1, OutputArray mapX2, OutputArray mapY2);

// self implemented remap function
void remap_impl(Mat &src, Mat &dst, Mat &mapx, Mat &mapy);

// self implemented Bilinear Interpolation
float CPU_BilinearInterpolation(Mat &src, float x, float y);

// self implemented convert color
void cvtColor_impl(Mat &src, Mat &dst);

// proxy function for remap on CPU
//void remap_cpu(Mat &left, Mat &right, Mat &mapX1, Mat &mapY1, Mat &mapX2, Mat &mapY2, int total, uchar *result);

void getCalibResult(Size targetSize, Mat &x1, Mat &y1, Mat &x2, Mat &y2);


class Host
{
public:
	Host();
	Host(Size size, int numDisp, int wsz, Mat &mx1, Mat &my1, Mat &mx2, Mat &my2);
	~Host();
	//void remap(Mat &src, Mat &dst);
	//void cvtColor(Mat &src, Mat &dst);
	void blockMatching(Mat &left, Mat &right, Mat &disparity);
	void calculateFrame(Mat &left, Mat &right);
private:
	int windowSize;
	int windowLength;
	int windowArea;
	int totalPixel;
	int numDisparity;
	Size sz;
	int rows;
	int cols;
	Mat x1;
	Mat y1;
	Mat x2;
	Mat y2;

	Mat left_color_cvted;
	Mat right_color_cvted;
	Mat left_remapped;
	Mat right_remapped;
	Mat disparity;
};

#endif /* utility_h */

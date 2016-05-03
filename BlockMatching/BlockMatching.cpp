#include "BlockMatching.h"
#include <iostream>

using namespace cv;
using namespace std;

void testBM(const Mat &left0, const Mat &right0, int SAD, int searchRange)
{
	Mat disparity;
	int cols = left0.cols;
	int rows = left0.rows;
	int total = cols * rows;
	const uchar * data_left = left0.ptr<uchar>(0);
	const uchar * data_right = right0.ptr<uchar>(0);
	uchar * data_dm = new uchar[total];
	int dbNum = 2 * SAD + 1;
	int dNum = dbNum * dbNum;

	memset(data_dm, 0, total * sizeof(uchar));
	// x is col index in the dbNum * dbNum window
	// y is row index in this window
	// z is (x + y * cols).
	// I compute them in advance for avoid computing repeatedly.
	Point3i * dLocDif = new Point3i[dNum];

	clock_t start = clock();
	for (int i = 0; i < dNum; i++) {
		dLocDif[i] = Point3i(i % dbNum - SAD, i / dbNum - SAD, 0);
		dLocDif[i].z = dLocDif[i].x + dLocDif[i].y * cols;
	}
	clock_t end = clock();

	cout << "prep location " << double(end - start) / CLOCKS_PER_SEC << endl;

	// I compute disparity difference for each search range to avoid
	// computing repeatedly.

	start = clock();
	uchar * dif_ = new uchar[total * searchRange];
	memset(dif_, 0, total * searchRange * sizeof(uchar));
	for (int _search = 0; _search < searchRange; _search++) {
		int th = _search * total;
		for (int i = 0; i < total; i++) {
			int c = i % cols - _search;
			if (c < 0) continue;
			dif_[i + th] = (uchar)std::abs(data_left[i] - data_right[i - _search]);
		}
	}
	end = clock();
	cout << "precalculate diff " << double(end - start) / CLOCKS_PER_SEC << endl;

	start = clock();
	for (int p = 0; p < total; p++) {
		int min = 50 * dNum;
		int dm = -256;
		int _col = p % cols;
		int _row = p / cols;
		int th = 0;

		// I search for the smallest difference between left and right image
		// using def_.
		for (int _search = 0; _search < searchRange; _search++, th += total) {
			if (_col + _search > cols) break;
			int temp = 0;
			for (int i = 0; i < dNum; i++) {
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
	end = clock();

	cout << "main loop " << double(end - start) / CLOCKS_PER_SEC << endl;

	disparity = Mat(rows, cols, CV_8UC1, data_dm);
	imshow("CPU", disparity);
}

void PreCal(const Mat &left0, const Mat &right0, uchar *dif_, int SAD, int searchRange)
{
	int cols = left0.cols;
	int rows = left0.rows;
	int total = cols * rows;
	const uchar * data_left = left0.ptr<uchar>(0);
	const uchar * data_right = right0.ptr<uchar>(0);
	// uchar * data_dm = new uchar[total];

	// I compute disparity difference for each search range to avoid
	// computing repeatedly.
	// dif_ = new uchar[total * searchRange];
	for (int _search = 0; _search < searchRange; _search++) {
		int th = _search * total;
		for (int i = 0; i < total; i++) {
			int c = i % cols - _search;
			if (c < 0) continue;
			dif_[i + th] = (uchar)std::abs(data_left[i] - data_right[i - _search]);
		}
	}
}

void getDisp(const Mat &left0, const Mat &right0, uchar *data_dm, int SAD, int searchRange)
{
	int cols = left0.cols;
	int rows = left0.rows;
	int total = cols * rows;
	const uchar * data_left = left0.ptr<uchar>(0);
	const uchar * data_right = right0.ptr<uchar>(0);
	// uchar * data_dm = new uchar[total];
	int dbNum = 2 * SAD + 1;
	int dNum = dbNum * dbNum;

	memset(data_dm, 0, total * sizeof(uchar));
	// x is col index in the dbNum * dbNum window
	// y is row index in this window
	// z is (x + y * cols).
	// I compute them in advance for avoid computing repeatedly.
	Point3i * dLocDif = new Point3i[dNum];

	clock_t start = clock();
	for (int i = 0; i < dNum; i++) {
		dLocDif[i] = Point3i(i % dbNum - SAD, i / dbNum - SAD, 0);
		dLocDif[i].z = dLocDif[i].x + dLocDif[i].y * cols;
	}
	clock_t end = clock();

	cout << "prep location " << double(end - start) / CLOCKS_PER_SEC << endl;

	// I compute disparity difference for each search range to avoid
	// computing repeatedly.

	start = clock();
	uchar * dif_ = new uchar[total * searchRange];
	memset(dif_, 0, total * searchRange * sizeof(uchar));
	for (int _search = 0; _search < searchRange; _search++) {
		int th = _search * total;
		for (int i = 0; i < total; i++) {
			int c = i % cols - _search;
			if (c < 0) continue;
			dif_[i + th] = (uchar)std::abs(data_left[i] - data_right[i - _search]);
		}
	}
	end = clock();
	cout << "precalculate diff " << double(end - start) / CLOCKS_PER_SEC << endl;

	start = clock();
	for (int p = 0; p < total; p++) {
		int min = 50 * dNum;
		int dm = -256;
		int _col = p % cols;
		int _row = p / cols;
		int th = 0;

		// I search for the smallest difference between left and right image
		// using def_.
		for (int _search = 0; _search < searchRange; _search++, th += total) {
			if (_col + _search > cols) break;
			int temp = 0;
			for (int i = 0; i < dNum; i++) {
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
	end = clock();

	cout << "main loop " << double(end - start) / CLOCKS_PER_SEC << endl;
} // testBM

void getAllSAD(const Mat &left0, const Mat &right0, uchar *data_dm, int SAD, int searchRange)
{
	int cols = left0.cols;
	int rows = left0.rows;
	int total = cols * rows;
	const uchar * data_left = left0.ptr<uchar>(0);
	const uchar * data_right = right0.ptr<uchar>(0);
	int dbNum = 2 * SAD + 1;
	int dNum = dbNum * dbNum;

	memset(data_dm, 0, total * sizeof(uchar));
	// x is col index in the dbNum * dbNum window
	// y is row index in this window
	// z is (x + y * cols).
	// I compute them in advance for avoid computing repeatedly.
	Point3i * dLocDif = new Point3i[dNum];

	clock_t start = clock();
	for (int i = 0; i < dNum; i++) {
		dLocDif[i] = Point3i(i % dbNum - SAD, i / dbNum - SAD, 0);
		dLocDif[i].z = dLocDif[i].x + dLocDif[i].y * cols;
	}
	clock_t end = clock();

	cout << "prep location " << double(end - start) / CLOCKS_PER_SEC << endl;

	// I compute disparity difference for each search range to avoid
	// computing repeatedly.

	start = clock();
	uchar * dif_ = new uchar[total * searchRange];
	memset(dif_, 0, total * searchRange * sizeof(uchar));
	for (int _search = 0; _search < searchRange; _search++) {
		int th = _search * total;
		for (int i = 0; i < total; i++) {
			int c = i % cols - _search;
			if (c < 0) continue;
			dif_[i + th] = (uchar)std::abs(data_left[i] - data_right[i - _search]);
		}
	}
	end = clock();
	cout << "precalculate diff " << double(end - start) / CLOCKS_PER_SEC << endl;

	start = clock();
	for (int p = 0; p < total; p++) {
		int min = 50 * dNum;
		int dm = -256;
		int _col = p % cols;
		int _row = p / cols;
		int th = 0;

		// I search for the smallest difference between left and right image
		// using def_.
		for (int _search = 0; _search < searchRange; _search++, th += total) {
			if (_col + _search > cols)
			{
				data_dm[p * searchRange + _search] = 255;
				continue;
			}
			int temp = 0;
			for (int i = 0; i < dNum; i++) {
				int _c = _col + dLocDif[i].x;
				if (_c >= cols || _c < 0) continue;
				int _r = _row + dLocDif[i].y;
				if (_r >= rows || _r < 0) continue;
				temp += dif_[th + p + dLocDif[i].z];
			}
			data_dm[p * searchRange + _search] = temp;
		}
	}
}

void compareDiff(const cv::Mat &left0, const cv::Mat &right0, uchar *GPUresult, int SADWindowSize, int searchRange, int total)
{
	uchar *ref = new uchar[total * searchRange];
	memset(ref, 0, total * searchRange * sizeof(uchar));
	PreCal(left0, right0, ref, SADWindowSize, searchRange);
	for (size_t i = 0; i < searchRange * total; i++)
	{
		if (ref[i] != GPUresult[i])
		{
			cout << i << endl;
		}
	}
	cout << -1;
}

void compareDisp(const cv::Mat &left, const cv::Mat &right, uchar *GPUresult, int SADWindowSize, int searchRange, int cols, int rows)
{
	uchar *ref = new uchar[rows * cols];
	getDisp(left, right, ref, SADWindowSize, searchRange);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			if (ref[i * cols + j] != GPUresult[i * cols + j])
			{
				cout << "[" << i << ":" << j << "]" << endl;
				cout << "CPU = " << (int)ref[i * cols + j] << ", GPU = " << (int)GPUresult[i * cols + j] << endl;
			}
		}
	}
}

void compareSAD(const cv::Mat &left, const cv::Mat &right, uchar *GPUresult, int SADWindowSize, int searchRange, int cols, int rows)
{
	int total = rows * cols;
	uchar *allSAD = new uchar[searchRange * total];
	memset(allSAD, 255, searchRange * total * sizeof(uchar));
	getAllSAD(left, right, allSAD, SADWindowSize, searchRange);

	for (size_t i = 0; i < searchRange * total; i++)
	{
		if (allSAD[i] != GPUresult[i])
			cout << i << endl;
	}
	cout << -1;
}
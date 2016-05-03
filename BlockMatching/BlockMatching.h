#ifndef BLOCK_MATCHING_H
#define BLOCK_MATCHING_H

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <ctime>

void testBM(const cv::Mat &left0, const cv::Mat &right0, int SAD, int searchRange);
void PreCal(const cv::Mat &left0, const cv::Mat &right0, uchar *diff_, int SAD, int searchRange);
void getDisp(const cv::Mat &left0, const cv::Mat &right0, uchar *disparity, int SAD, int searchRange);
void getAllSAD(const cv::Mat &left0, const cv::Mat &right0, uchar *data_dm, int SAD, int searchRange);

void compareDiff(const cv::Mat &left0, const cv::Mat &right0, uchar *GPUresult, int SADWindowSize, int searchRange, int total);
void compareDisp(const cv::Mat &left, const cv::Mat &right, uchar *GPUresult, int SADWindowSize, int searchRange, int cols, int rows);
void compareSAD(const cv::Mat &left, const cv::Mat &right, uchar *GPUresult, int SADWindowSize, int searchRange, int cols, int rows);
#endif
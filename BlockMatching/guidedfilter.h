#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class GuidedFilterImpl
{
public:
	virtual ~GuidedFilterImpl() {}

	cv::Mat filter(const cv::Mat &p, int depth);

	virtual cv::Mat getMat(const std::string &s) = 0;
protected:
	int Idepth;

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) = 0;
};

class GuidedFilterColor : public GuidedFilterImpl
{
public:
	GuidedFilterColor(const cv::Mat &I, int r, double eps);
	virtual cv::Mat getMat(const std::string &s);
private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p);

private:
	std::vector<cv::Mat> Ichannels;
	int r;
	double eps;
	cv::Mat mean_I_r, mean_I_g, mean_I_b;
	cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
	GuidedFilterMono(const cv::Mat &I, int r, double eps);
	virtual cv::Mat getMat(const std::string &s);
private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p);

public:
	int r;
	double eps;
	cv::Mat I, mean_I, var_I;

	// test
	cv::Mat mean_II, mean_p, mean_Ip, cov_Ip, a, b, mean_a, mean_b, result;
};

class GuidedFilter
{
public:
    GuidedFilter(const cv::Mat &I, int r, double eps);
    ~GuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

    GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);

#endif

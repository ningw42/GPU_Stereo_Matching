#ifndef caller_h
#define caller_h

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;

//void singleFrame();
//void remapTest();
//void cvtColorTest();
void streamDepth(Size targetSize, int SADWindowSize, int numDisp);
void photoDepth(Size targetSize, int SADWindowSize, int numDisp);
int cameraTest();
int Calib();
void OpenCVBM();

// compare the three method
int start_gpu(Size targetSize, int windowSize, int numDisp);
int start_cpu(Size targetSize, int windowSize, int numDisp);
int start_opencv(Size targetSize, int windowSize, int numDisp);
#endif caller_h
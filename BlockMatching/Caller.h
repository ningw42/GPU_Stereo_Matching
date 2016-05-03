#ifndef caller_h
#define caller_h

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;

void singleFrame();
void remapTest();
void cvtColorTest();
void streamDepth(Size targetSize, int SADWindowSize, int minDisp);
void photoDepth(Size targetSize, int SADWindowSize, int minDisp);
int cameraTest();
#endif caller_h
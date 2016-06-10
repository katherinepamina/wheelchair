#ifndef GAZE_TRACKING_H
#define GAZE_TRACKING_H

#include "opencv2/opencv.hpp"
#include <vector>

using namespace std;
using namespace cv;

class gaze_tracker {
	// Detect faces in an image using a Haar cascade
	vector<Rect_ <int> > detectFaces(Mat frame, string cascade_path);
};

#endif
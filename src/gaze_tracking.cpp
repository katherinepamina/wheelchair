#include "../include/wheelchair/gaze_tracking.h"

using namespace std;
using namespace cv;

gaze_tracker::vector<Rect_ <int> > detectFaces(Mat frame, string cascade_path) {
    vector<Rect_<int> > faces = vector<Rect_<int>>();
    
    /*Mat original_copy = frame.clone();  <-- now converting to gray before this function
    Mat gray_copy;
    cvtColor(original_copy, gray_copy, CV_BGR2GRAY);
     */
    face_cascade.detectMultiScale(frame, faces);
    return faces;
}


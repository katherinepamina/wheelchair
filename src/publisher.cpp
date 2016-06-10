#include <iostream>
#include <sstream>
#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "std_msgs/String.h"
#include "geometry_msgs/Twist.h"
//#include "../include/wheelchair/gaze_tracking.h" TODO: clean everything up

using namespace std;
using namespace cv;

// Global variables
string face_cascade_path = "/home/pamina/Desktop/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;

// Constants
const float  kSmoothFaceFactor = 0.005;
const float  kEyeTopFraction = .25;
const float  kEyeSideFraction = .10;
const float  kEyeHeightFraction = .30;
const float  kEyeWidthFraction = .35;
const double kGradientThreshold = 50.0;
const int    kWeightBlurSize = 5;
const float  kWeightDivisor = 1.0;
const int    kScaledDownEyeWidth = 50;
const float  centerLowerThreshold = 0.45; // under this threshold, user is looking left
const float  centerUpperThreshold = 0.65; // above this threshold, user is looking right


// shit ton of things to move elsewhere
// Helper functions (move to other file later)
vector<Rect_ <int> > detectFaces(Mat frame, string cascade_path) {
    vector<Rect_<int> > faces = vector<Rect_<int> >();
    face_cascade.detectMultiScale(frame, faces);
    return faces;
}

Rect getBiggestFace(vector<Rect_ <int> > faces) {
    int maxSize = 0;
    Rect maxFace;
    for (int i=0; i<faces.size(); i++) {
        Rect face_i = faces.at(i);
        int currentSize = face_i.width + face_i.height;
        if (currentSize > maxSize) {
            maxSize = currentSize;
            maxFace = face_i;
        }
    }
    return maxFace;
}

// Some helper functions to move between frames of references i.e. moving out of a smaller frame of reference within a larger one)
Point translatePoint(Point p, Rect currentFrame) {
    p.x = p.x + currentFrame.x;
    p.y = p.y + currentFrame.y;
    return p;
}

Rect translateRect(Rect r, Rect from) {
    r.x = r.x + from.x;
    r.y = r.y + from.y;
    return r;
}

// Make a draw all things function later
void drawFace(Mat &frame, Rect_ <int> face) {
    rectangle(frame, face, CV_RGB(0,255,0), 1);
    return;
}

Mat computeXGradient(Mat mat) {
    Mat gradient(mat.rows, mat.cols, CV_64F);

    for (int j=0; j<mat.rows; j++) {
        const uchar *matRowPtr = mat.ptr<uchar>(j);
        double *gradientRowPtr = gradient.ptr<double>(j);
        
        // Right now, calculating the gradient by taking the difference on x or y side
        
        // Process the first column separately
        gradientRowPtr[0] = matRowPtr[1] - matRowPtr[0];
        
        // Process middle columns
        for (int i=1; i<mat.cols-1; i++) {
            gradientRowPtr[i] = (matRowPtr[i+1] - matRowPtr[i-1])/2.0;
        }
        
        // Also process the last column separately
        gradientRowPtr[mat.cols-1] = matRowPtr[mat.cols-1] - matRowPtr[mat.cols-2];
    }
    
    return gradient;
}

Mat computeYGradient(Mat mat) {
    // Compute the y gradient by taking the gradient of the transpose and then transposing it again
    return computeXGradient(mat.t()).t();
}

Mat computeMagnitudes(Mat mat1, Mat mat2) {
    // check that mat1 and mat2 are the same dimension? will be necessary if using cascade
    Mat mag(mat1.rows, mat1.cols, CV_64F);
    for (int j=0; j<mat1.rows; j++) {
        const double *xPtr = mat1.ptr<double>(j);
        const double *yPtr = mat2.ptr<double>(j);
        double *magPtr = mag.ptr<double>(j);
        for (int i=0; i<mat1.cols; i++) {
            magPtr[i] = sqrt((xPtr[i]*xPtr[i]) + yPtr[i]*yPtr[i]);
        }
    }
    return mag;
}

double calculateGradientThreshold(Mat gradient) {
    Scalar stdDev;
    Scalar mean;
    meanStdDev(gradient, mean, stdDev);
    //cout << "mean: " << mean[0] << endl;
    //cout << "stdDev: " << stdDev[0] << endl;
    double stdDevScaled = stdDev[0] / sqrt(gradient.rows*gradient.cols);
    return mean[0] + stdDevScaled*50; // trying 50 for now based on recommendation from article
    
    //return mean[0] + stdDev[0]/10.0;
}

void normalizeMats(Mat &mat1, Mat &mat2) {
    Mat magnitudes = computeMagnitudes(mat1, mat2);
    
    // Get some sort of threshold so if the gradient is under the threshold, just set it to zero
    double gradThreshold = calculateGradientThreshold(magnitudes);
    //cout << "threshold: " << gradThreshold << endl;
    for (int j=0; j<mat1.rows; j++) {
        double * mat1Ptr = mat1.ptr<double>(j);
        double * mat2Ptr = mat2.ptr<double>(j);
        double * magPtr = magnitudes.ptr<double>(j);
        
        for (int i=0; i<mat1.cols; i++) {
            double mat1Element = mat1Ptr[i];
            double mat2Element = mat2Ptr[i];
            double mag = magPtr[i];
            if (mag > gradThreshold) {
                mat1Ptr[i] = mat1Element / mag;
                mat2Ptr[i] = mat2Element / mag;
            } else {
                mat1Ptr[i] = 0.0;
                mat2Ptr[i] = 0.0;
            }
        }
    }
    return;
}

void normalizeVector(double &x, double &y) {
    double magnitude = sqrt(x*x + y*y);
    x = x / magnitude;
    y = y / magnitude;
    return;
}


Mat getWeightedImage(Mat image) {
    Mat weight;
    GaussianBlur(image, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
    
    //blur(image, weight, Size(kWeightBlurSize, kWeightBlurSize));
    for (int j=0; j<weight.rows; j++) {
        unsigned char * rowPtr = weight.ptr<unsigned char>(j);
        for (int i=0; i<weight.cols; i++) {
            rowPtr[i] = (255 - rowPtr[i]);
        }
    }
    //imshow("weighted", weight);
    return weight;
}

void scaleDownImage(Mat &src, Mat &dst) {
    float ratio = (float)src.rows/src.cols;
    Size smallerSize = Size(kScaledDownEyeWidth, ((float)kScaledDownEyeWidth)*ratio);
    resize(src, dst, smallerSize);
}

Point unscalePoint(Point p, Rect size) {
    float ratio = (((float)kScaledDownEyeWidth)/size.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return Point(x, y);
}

// where x and y is a Mat element, gradX and gradY is gradient vector at that element
void evaluateAllCenters(int x, int y, const Mat &weight, double gradX, double gradY, Mat &result) {
    for (int j=0; j < result.rows; j++) {
        double * resultPtr = result.ptr<double>(j);
        const unsigned char * weightPtr = weight.ptr<unsigned char>(j);
        for (int i=0; i<result.cols; i++) {
            //cout << "(" << i << ", " << j << ") " << endl;
            // if the current location being tested is the same as the one we're evaluating at, continue
            if (x == i && y == j) {
                continue;
            }
            // otherwise, calculate the direction vectors
            double directionX = x - i;
            double directionY = y - j;
            //cout << "before norm: (" << directionX << ", " << directionY << ") " << endl;
            normalizeVector(directionX, directionY);
            //cout << "after norm: (" << directionX << ", " << directionY << ") " << endl;
            
            // Get the dot product
            //cout << "gradient: (" << gradX << ", " << gradY << ") " << endl;
            double dotProduct = directionX*gradX + directionY*gradY;
            dotProduct = (dotProduct > 0) ? dotProduct : -1 * dotProduct;
 
            // Add the weighting
            resultPtr[i] += dotProduct*dotProduct*weightPtr[i]/kWeightDivisor;
            
        }
    }
}

Mat getSubImage(Mat image, Rect imageRect, Rect roi) {
    if (imageRect.width > 0 && imageRect.height > 0) {
        int X = roi.x - imageRect.x;
        int Y = roi.y - imageRect.y;
        Rect translatedROI(X, Y, roi.width, roi.height);
        Mat subimage = image(translatedROI);
        if (!subimage.empty()) {
            //imshow("subimage", subimage);
            return subimage;
        } else {
            return image;
        }
    } else {
        return image; // TODO: is this really desired behavior
    }
}


Point findEyeCenter(Mat eyeImageUnscaled, Rect eyeROI, String window) {
    Mat eyeImage;
    scaleDownImage(eyeImageUnscaled, eyeImage);
    
    // Get the gradients of the eye image
    Mat gradX = computeXGradient(eyeImage);
    Mat gradY = computeYGradient(eyeImage);
    
    normalizeMats(gradX, gradY);
    
    // Get a "weight" Mat, equal to the inverse gray-scale image
    Mat weight = getWeightedImage(eyeImage);
   
    // Set up the result Mat
    Mat result = Mat::zeros(eyeImage.rows, eyeImage.cols, CV_64F);
    
    // For each gradient location, evaluate every possible eye center
    for (int j=0; j<eyeImage.rows; j++) {
        const double * gradXPtr = gradX.ptr<double>(j);
        const double * gradYPtr = gradY.ptr<double>(j);
        for (int i=0; i<eyeImage.cols; i++) {
            double gradX = gradXPtr[i];
            double gradY = gradYPtr[i];
            // if the gradient is 0, ignore the point
            if (gradX == 0.0 && gradY == 0.0) {
                continue;
            }
            // otherwise, test all possible centers against this location/gradient
            evaluateAllCenters(i, j, weight, gradX, gradY, result);
        }
    }
    
    // Look for the maximum dot product (should correspond with the center of the circle)
    double numGradients = eyeImage.rows * eyeImage.cols;
    Mat resultScaled;
    result.convertTo(resultScaled, CV_32F, 1.0/numGradients);
    Point maxCenter;
    double maxDotProduct = 0;
    
    double currentMax = 0;
    Point currentMaxPoint;
    for (int j=0; j<resultScaled.rows; j++) {
        const float * resultPtr = resultScaled.ptr<float>(j);
        for (int i=0; i<resultScaled.cols; i++) {
            if (resultPtr[i] > currentMax) {
                currentMax = resultPtr[i];
                currentMaxPoint.x = i;
                currentMaxPoint.y = j;
            }
        }
    }
    maxCenter = currentMaxPoint;
    maxDotProduct = currentMax;

    //cout << "max dot product " << maxDotProduct << endl;
    //cout << "max center" << maxCenter.x << ", " << maxCenter.y << endl;
    //cout << "best center: (" << maxCenter.x << ", " << maxCenter.y << ")" << endl;
    //cout << "eyeRegion: (" << eyeRegion.x << ", "<< eyeRegion.y << ")" << endl;
    //cout << "faceRegion: (" << faceRegion.x << ", " << faceRegion.y << ")" << endl;
    
    // Need to translate eye point back to original coordinate system (biggestFaceRect)
    Point resultCenter = unscalePoint(maxCenter, eyeROI);
    resultCenter.x += eyeROI.x;
    resultCenter.y += eyeROI.y;
    
    //const double * eyePtr = eyeImage.ptr<double>(maxCenter.y);
    //cout << "color at pupil: " << eyePtr[maxCenter.x] << endl;
    return resultCenter;
}

vector<Point> detectCorner(Mat frame, Rect eyeRect) {
    Mat img = frame(eyeRect);
    vector<Point> corners = vector<Point>();
    goodFeaturesToTrack(img, corners, 50, 0.02, 5); // what number makes sense for the "quality level" parameter?  just using 0.02 for now
    for (int i=0; i<corners.size(); i++) {
        corners.at(i) = translatePoint(corners.at(i), eyeRect);
        //circle(frame, corners.at(i), 2, CV_RGB(0,0,255), -1);
    }
    return corners;
}

vector<Rect> getEyeRegionRect(Rect faceRect) {
    int width = faceRect.width;
    int height = faceRect.height;
    int eyeRegionTop = height * kEyeTopFraction;
    int eyeRegionSide = width * kEyeSideFraction;
    int eyeRegionWidth = width * kEyeWidthFraction;
    int eyeRegionHeight = width * kEyeHeightFraction;
    int leftEyeX = faceRect.x + eyeRegionSide;
    int rightEyeX = faceRect.x + 2*eyeRegionSide + eyeRegionWidth;
    int eyeY =  faceRect.y + eyeRegionTop;
    
    Rect leftEyeRegion(leftEyeX, eyeY, eyeRegionWidth, eyeRegionHeight);
    Rect rightEyeRegion(rightEyeX, eyeY, eyeRegionWidth, eyeRegionHeight);
    
    vector<Rect> eyeRects = vector<Rect>();
    eyeRects.push_back(leftEyeRegion);
    eyeRects.push_back(rightEyeRegion);
    return eyeRects;
}

/* 
 this is the area in which we look for eye corners
*/
Rect getFilterArea(Rect eyeRect, Point pupil) {
    Rect filterRect = Rect();
    // These hard-coded dimensions work for my eyes but still need to be tested on others
    filterRect.x = pupil.x - eyeRect.width/3.5;
    filterRect.y = pupil.y - eyeRect.height/12;
    filterRect.height = eyeRect.height/5;
    filterRect.width = eyeRect.width/1.5;

    return filterRect;
}



double getGazeRatio(Mat &frame, Rect faceRect) {
    Mat face_image = frame(faceRect);
    
    // Get the eye regions using a ratio method
    vector<Rect> eyeRects = getEyeRegionRect(faceRect);
    Rect leftEyeRect = eyeRects.at(0);
    Rect rightEyeRect = eyeRects.at(1); // left and right might be flipped
    
    // Get subimages
    Mat leftEyeImage = getSubImage(face_image, faceRect, leftEyeRect);
    Mat rightEyeImage = getSubImage(face_image, faceRect, rightEyeRect);
    
    // find eye center
    Point leftPupil = findEyeCenter(leftEyeImage, leftEyeRect, "left eye");
    Point rightPupil = findEyeCenter(rightEyeImage, rightEyeRect, "right eye");
    
    // find corners
    Rect leftFilterRect = getFilterArea(leftEyeRect, leftPupil);
    Rect rightFilterRect = getFilterArea(rightEyeRect, rightPupil);
    vector<Point> potentialLeftCorners = detectCorner(frame, leftEyeRect);
    vector<Point> potentialRightCorners = detectCorner(frame, rightEyeRect);
   
    Point leftMaxCorner = Point();
    Point leftMinCorner = Point();

    int maxLeftCol = 0;
    int minLeftCol = 999999999; // haha get the actual constant?  should be fine bc images are small
    for (int i=0; i<potentialLeftCorners.size(); i++) {
        Point potential = potentialLeftCorners.at(i);
        if (potential.x > maxLeftCol &&
            potential.x > leftFilterRect.x && potential.x <= leftFilterRect.x + leftFilterRect.width &&
            potential.y > leftFilterRect.y && potential.y <= leftFilterRect.y + leftFilterRect.height) {
            maxLeftCol = potential.x;
            leftMaxCorner = potential;
        }
        else if (potential.x < minLeftCol &&
            potential.x > leftFilterRect.x && potential.x <= leftFilterRect.x + leftFilterRect.width &&
            potential.y > leftFilterRect.y && potential.y <= leftFilterRect.y + leftFilterRect.height) {
            minLeftCol = potential.x;
            leftMinCorner = potential;
        }
    }
    Point rightMaxCorner = Point();
    Point rightMinCorner = Point();
    int maxRightCol = 0;
    int minRightCol = 999999999;
    for (int i=0; i<potentialRightCorners.size(); i++) {
        Point potential = potentialRightCorners.at(i);
        if (potential.x > maxRightCol &&
            potential.x > rightFilterRect.x && potential.x < rightFilterRect.x + rightFilterRect.width &&
            potential.y > rightFilterRect.y && potential.y < rightFilterRect.y + rightFilterRect.height) {
            maxRightCol = potential.x;
            rightMaxCorner = potential;
        }
        else if (potential.x < minRightCol &&
            potential.x > rightFilterRect.x && potential.x <= rightFilterRect.x + rightFilterRect.width &&
            potential.y > rightFilterRect.y && potential.y <= rightFilterRect.y + rightFilterRect.height) {
            minRightCol = potential.x;
            rightMinCorner = potential;
        }
    }


    float rightEyeLength = abs(rightMaxCorner.x - rightMinCorner.x);
    float leftEyeLength = abs(leftMaxCorner.x - leftMinCorner.x);
    float rightPupilToCorner = abs(rightMaxCorner.x - rightPupil.x);
    float leftPupilToCorner = abs(leftMaxCorner.x - leftPupil.x);
    float rightEyeRatio = rightPupilToCorner / rightEyeLength;
    float leftEyeRatio = leftPupilToCorner / leftEyeLength;

    // draw THINGS ELSEWHERE
    //rectangle(frame, leftEyeRect, CV_RGB(0,0,255), 1);
    //rectangle(frame, rightEyeRect, CV_RGB(0,0,255), 1);
    circle(frame, leftPupil, 3, CV_RGB(0,0,255), -1);
    circle(frame, rightPupil, 3, CV_RGB(0,0,255), -1);
    circle(frame, leftMinCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, rightMinCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, leftMaxCorner, 2, CV_RGB(0,0,255), -1);
    circle(frame, rightMaxCorner, 2, CV_RGB(0,0,255), -1);
    rectangle(frame, leftFilterRect, CV_RGB(0,0,255), 1);
    rectangle(frame, rightFilterRect, CV_RGB(0,0,255), 1);

    //cout << "left pupil: " << leftPupil.x << endl;
    //cout << "right pupil: " << rightPupil.x << endl;
    //cout << "left maxCorner: " << leftMaxCorner.x << endl;
    //cout << "right maxCorner: " << rightMaxCorner.x << endl;
    //cout << "left minCorner: " << leftMinCorner.x << endl;
    //cout << "right minCorner: " << rightMinCorner.x << endl;
    //cout << "left ratio: " << leftEyeRatio << endl;
    //cout << "right ratio: " << rightEyeRatio << endl;
    float avgRatio = (leftEyeRatio + rightEyeRatio)/2.0;
    return avgRatio;
}



int main(int argc, char **argv) {
	// initialize ROS
	ros::init(argc, argv, "gazetracker");

	// This is the main access point to communications with the ROS system.
	// The first NodeHandle constructed will fully initialize this node, and the last NodeHandle destructed
	// will close down the node
	ros::NodeHandle n;
	
	// Test publisher for now
	// advertise function tells ROS you want to advertise on a given topic (param 1)
	// can publish messages on the topic through the publish() function.
	// second param is the size of the message queue
	ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
	// Loop rate in Hz
	ros::Rate loop_rate(500);

	// Load the face cascade
	if (!face_cascade.load(face_cascade_path)) {
		cout << "Error loading face cascade." << endl;
		return -1;
	}

	VideoCapture cap("/home/pamina/Desktop/gaze_sample2.mov");
	if (!cap.isOpened()) {
		cout << "Video file is not open." << endl;
		return -1;
	}

	Mat frame;
	geometry_msgs::Twist prev_cmd = geometry_msgs::Twist();
	geometry_msgs::Twist cmd = geometry_msgs::Twist();
	prev_cmd.linear.x = 1;
	prev_cmd.linear.y = 0;
	prev_cmd.linear.z = 0;
	prev_cmd.angular.z = 0.5;
	cmd.linear.x = 1;
	cmd.angular.z = 0.5;
	//ros::ok() false when Ctrl-C pressed, if ros is shut down, if all node handles have been destroyed, etc
	while (ros::ok()) {
		//cap.grab();
		//cap.retrieve(frame);
		cap.read(frame);
		
		/* do img processing here */
		cvtColor(frame, frame, CV_BGR2GRAY);
		vector<Rect_ <int> > faces = detectFaces(frame, face_cascade_path);
		double gaze_ratio;
		Rect biggestFace = getBiggestFace(faces);
        if (biggestFace.width > 0 && biggestFace.height > 0) {
            //drawFace(frame, biggestFace);
            gaze_ratio = getGazeRatio(frame, biggestFace);
            if (gaze_ratio < centerLowerThreshold) {
            	// looking left
            	cmd.linear.x = 1;//just some number for now
            	cmd.angular.z = 0.5;
            }
            else if (gaze_ratio < centerUpperThreshold) {
            	// looking center
            	cmd.linear.x = 1;//just some number for now
            	cmd.angular.z = 0;
            }
            else {
            	// looking right
            	cmd.linear.x = 1;//just some number for now
            	cmd.angular.z = -0.5;
            }
        }

		// display the frame
		imshow("frame", frame);
		waitKey(1);
	
		// only publish message if it's a DIFFERENT direction from before
		if (prev_cmd.linear.x != cmd.linear.x || prev_cmd.angular.z != cmd.angular.z) {
			cout << gaze_ratio << endl;
			pub.publish(cmd);
			// call all the callbacks
			ros::spinOnce();
			prev_cmd.linear.x = cmd.linear.x; // not sure if it'd be deep or shallow copy.. too tired to check
			prev_cmd.angular.z = cmd.angular.z;
		}
		
		// sleep for the remaining time to his the 10Hz publish rate
		loop_rate.sleep();
		
	}

	return 0;
}
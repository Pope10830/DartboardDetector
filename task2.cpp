// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

struct Bound {
  int id;
	int x;
	int y;
	int width;
	int height;
};

/** Function Headers */
void detectAndDisplay(Mat frame, vector<Rect> &dartboards);
void overlayGroundTruth(Mat frame, vector<Bound> groundTruth);
bool isSuccessfullyDetected(Bound groundTruth, vector<Rect> dartboards);
vector<Bound> getSuccessfullyDetected(vector<Bound> groundTruth, vector<Rect> dartboards);
void overlaySuccessfullyDetected(Mat frame, vector<Bound> successfullyDetected);
Bound createBound(int id, int x, int y, int width, int height);
vector<Bound> getGroundTruth(int id);
float getIOU(Bound groundTruth, Rect dartboard);
float calculateF1Score(vector<Bound> groundTruth, vector<Bound> successfullyDetected, vector<Rect> dartboards);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main(int argc, const char** argv) {
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  string arg = argv[1];
  arg.erase(0, 4);
  arg.erase(arg.length() - 4);

  int id;
  std::istringstream(arg) >> id;

  vector<Bound> groundTruth = getGroundTruth(id);

	// Load the Strong Classifier in a structure called `Cascade'
	if(!cascade.load(cascade_name)) {
    printf("--(!)Error loading\n");
    return -1;
  };

  vector<Rect> dartboards;

	detectAndDisplay(frame, dartboards);
  overlayGroundTruth(frame, groundTruth);
  vector<Bound> successfullyDetected = getSuccessfullyDetected(groundTruth, dartboards);
  overlaySuccessfullyDetected(frame, successfullyDetected);
  float f1Score = calculateF1Score(groundTruth, successfullyDetected, dartboards);

  string filename = argv[1];
  imwrite("task2outputs/task2_" + filename, frame);
	imwrite("detected.jpg", frame);

	return 0;
}

float calculateF1Score(vector<Bound> groundTruth, vector<Bound> successfullyDetected, vector<Rect> dartboards) {
  float tp = successfullyDetected.size();
  float fn = groundTruth.size() - successfullyDetected.size();
  float fp = dartboards.size() - successfullyDetected.size();

  float f1 = tp / (tp + (0.5 * (fp + fn)));

  std::cout << "tp " << tp << std::endl;
  std::cout << "fn " << fn << std::endl;
  std::cout << "fp " << fp << std::endl;
  std::cout << "f1 " << f1 << std::endl;

  return f1;
}

float getIOU(Bound groundTruth, Rect dartboard) {
  // Intersection Rect Coords
  int xa = std::max(groundTruth.x, dartboard.x);
  int ya = std::max(groundTruth.y, dartboard.y);
  int xb = std::min(groundTruth.x + groundTruth.width, dartboard.x + dartboard.width);
  int yb = std::min(groundTruth.y + groundTruth.height, dartboard.y + dartboard.height);

  float intersectionArea = std::max(0, xb - xa + 1) * std::max(0, yb - ya + 1);
  float truthArea = groundTruth.width * groundTruth.height;
  float dartboardArea = dartboard.width * dartboard.height;

  float iou = intersectionArea / (truthArea + dartboardArea - intersectionArea);

  return iou;
}

bool isSuccessfullyDetected(Bound groundTruth, vector<Rect> dartboards) {
  float threshold = 0.7;

  for (int j = 0; j < dartboards.size(); j++) {
    if (getIOU(groundTruth, dartboards[j]) > threshold) {
      return true;
    }
  }

  return false;
}

vector<Bound> getSuccessfullyDetected(vector<Bound> groundTruth, vector<Rect> dartboards) {
  vector<Bound> successfullyDetected;
  for (int i = 0; i < groundTruth.size(); i++) {
    if (isSuccessfullyDetected(groundTruth[i], dartboards)) {
      successfullyDetected.push_back(groundTruth[i]);
    }
  }

  return successfullyDetected;
}

void overlaySuccessfullyDetected(Mat frame, vector<Bound> successfullyDetected) {
  for (int i = 0; i < successfullyDetected.size(); i++) {
    rectangle(frame, Point(successfullyDetected[i].x, successfullyDetected[i].y), Point(successfullyDetected[i].x + successfullyDetected[i].width, successfullyDetected[i].y + successfullyDetected[i].height), Scalar(255, 0, 0), 2);
  }
}

vector<Bound> getGroundTruth(int id) {
  vector<Bound> groundTruth;

  if (id == 0) {
    groundTruth.push_back(createBound(0, 445, 15, 151, 176));
  } else if (id == 1) {
    groundTruth.push_back(createBound(1, 198, 131, 193, 190));
  } else if (id == 2) {
    groundTruth.push_back(createBound(2, 104, 98, 86, 86));
  } else if (id == 3) {
    groundTruth.push_back(createBound(3, 326, 149, 63, 69));
  } else if (id == 4) {
    groundTruth.push_back(createBound(4, 185, 97, 191, 198));
  } else if (id == 5) {
    groundTruth.push_back(createBound(5, 434, 142, 101, 105));
  } else if (id == 6) {
    groundTruth.push_back(createBound(6, 214, 119, 57, 59));
  } else if (id == 7) {
    groundTruth.push_back(createBound(7, 255, 173, 136, 141));
  } else if (id == 8) {
    groundTruth.push_back(createBound(8, 69, 254, 57, 85));
    groundTruth.push_back(createBound(8, 844, 219, 112, 117));
  } else if (id == 9) {
    groundTruth.push_back(createBound(9, 205, 48, 227, 229));
  } else if (id == 10) {
    groundTruth.push_back(createBound(10, 93, 105, 92, 107));
    groundTruth.push_back(createBound(10, 586, 129, 53, 81));
    groundTruth.push_back(createBound(10, 918, 151, 32, 61));
  } else if (id == 11) {
    groundTruth.push_back(createBound(11, 177, 107, 54, 65));
  } else if (id == 12) {
    groundTruth.push_back(createBound(12, 159, 80, 55, 132));
  } else if (id == 13) {
    groundTruth.push_back(createBound(13, 276, 123, 125, 126));
  } else if (id == 14) {
    groundTruth.push_back(createBound(14, 123, 104, 121, 121));
    groundTruth.push_back(createBound(14, 990, 97, 119, 121));
  } else if (id == 15) {
    groundTruth.push_back(createBound(15, 157, 59, 123, 132));
  }

  return groundTruth;
}

Bound createBound(int id, int x, int y, int width, int height) {
  Bound b = {id, x, y, width, height};
  return b;
}

void overlayGroundTruth(Mat frame, vector<Bound> groundTruth) {
  for (int i = 0; i < groundTruth.size(); i++) {
    rectangle(frame, Point(groundTruth[i].x, groundTruth[i].y), Point(groundTruth[i].x + groundTruth[i].width, groundTruth[i].y + groundTruth[i].height), Scalar(0, 0, 255), 2);
  }
}

void detectAndDisplay(Mat frame, vector<Rect> &dartboards) {
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500));

       // 3. Print number of Faces found
	std::cout << dartboards.size() << std::endl;

       // 4. Draw box around faces found
	for(int i = 0; i < dartboards.size(); i++) {
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
}

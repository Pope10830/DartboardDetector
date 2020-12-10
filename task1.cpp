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
void detectAndDisplay(Mat frame, vector<Rect> &faces);
void overlayGroundTruth(Mat frame, vector<Bound> groundTruth);
bool isSuccessfullyDetected(Bound groundTruth, vector<Rect> faces);
vector<Bound> getSuccessfullyDetected(vector<Bound> groundTruth, vector<Rect> faces);
void overlaySuccessfullyDetected(Mat frame, vector<Bound> successfullyDetected);
Bound createBound(int id, int x, int y, int width, int height);
vector<Bound> getGroundTruth(int id);
float getIOU(Bound groundTruth, Rect face);
float calculateF1Score(vector<Bound> groundTruth, vector<Bound> successfullyDetected, vector<Rect> faces);

/** Global variables */
String cascade_name = "frontalface.xml";
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

  vector<Rect> faces;

	detectAndDisplay(frame, faces);
  overlayGroundTruth(frame, groundTruth);
  vector<Bound> successfullyDetected = getSuccessfullyDetected(groundTruth, faces);
  overlaySuccessfullyDetected(frame, successfullyDetected);
  float f1Score = calculateF1Score(groundTruth, successfullyDetected, faces);

	imwrite("detected.jpg", frame);

	return 0;
}

float calculateF1Score(vector<Bound> groundTruth, vector<Bound> successfullyDetected, vector<Rect> faces) {
  float tp = successfullyDetected.size();
  float fn = groundTruth.size() - successfullyDetected.size();
  float fp = faces.size() - successfullyDetected.size();

  float f1 = tp / (tp + (0.5 * (fp + fn)));

  std::cout << "tp " << tp << std::endl;
  std::cout << "fn " << fn << std::endl;
  std::cout << "fp " << fp << std::endl;
  std::cout << "f1 " << f1 << std::endl;

  return f1;
}

float getIOU(Bound groundTruth, Rect face) {
  // Intersection Rect Coords
  int xa = std::max(groundTruth.x, face.x);
  int ya = std::max(groundTruth.y, face.y);
  int xb = std::min(groundTruth.x + groundTruth.width, face.x + face.width);
  int yb = std::min(groundTruth.y + groundTruth.height, face.y + face.height);

  float intersectionArea = std::max(0, xb - xa + 1) * std::max(0, yb - ya + 1);
  float truthArea = groundTruth.width * groundTruth.height;
  float faceArea = face.width * face.height;

  float iou = intersectionArea / (truthArea + faceArea - intersectionArea);

  return iou;
}

bool isSuccessfullyDetected(Bound groundTruth, vector<Rect> faces) {
  float threshold = 0.7;

  for (int j = 0; j < faces.size(); j++) {
    if (getIOU(groundTruth, faces[j]) > threshold) {
      return true;
    }
  }

  return false;
}

vector<Bound> getSuccessfullyDetected(vector<Bound> groundTruth, vector<Rect> faces) {
  vector<Bound> successfullyDetected;
  for (int i = 0; i < groundTruth.size(); i++) {
    if (isSuccessfullyDetected(groundTruth[i], faces)) {
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

  if (id == 4) {
    groundTruth.push_back(createBound(4, 350, 130, 130, 120));
  } else if (id == 5) {
    groundTruth.push_back(createBound(5, 62, 140, 58, 55));
    groundTruth.push_back(createBound(5, 54, 260, 62, 52));
    groundTruth.push_back(createBound(5, 192, 217, 57, 62));
    groundTruth.push_back(createBound(5, 249, 168, 58, 52));
    groundTruth.push_back(createBound(5, 293, 250, 56, 57));
    groundTruth.push_back(createBound(5, 379, 194, 57, 51));
    groundTruth.push_back(createBound(5, 427, 242, 59, 57));
    groundTruth.push_back(createBound(5, 518, 184, 49, 48));
    groundTruth.push_back(createBound(5, 560, 252, 58, 55));
    groundTruth.push_back(createBound(5, 646, 191, 59, 54));
    groundTruth.push_back(createBound(5, 680, 252, 52, 57));
  } else if (id == 13) {
    groundTruth.push_back(createBound(13, 418, 137, 107, 102));
  } else if (id == 14) {
    groundTruth.push_back(createBound(14, 471, 229, 80, 83));
    groundTruth.push_back(createBound(14, 732, 206, 89, 82));
  } else if (id == 15) {
    groundTruth.push_back(createBound(15, 69, 139, 55, 63));
    groundTruth.push_back(createBound(15, 368, 117, 45, 61));
    groundTruth.push_back(createBound(15, 532, 136, 57, 65));
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

void detectAndDisplay(Mat frame, vector<Rect> &faces) {
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500));

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for(int i = 0; i < faces.size(); i++) {
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
}

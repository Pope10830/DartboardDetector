#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <cmath>
#include <iterator>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
String filename = "dart3.jpg";
int scale = 15;

struct Coord {
  int x;
  int y;
  int r;
};

struct AccumulatorStruct {
  int array[80][80][100];
  int largest;
};

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

void convolution(Mat image, Mat newImage, float Kernel[][3]) {
  for (int y = 1; y < image.rows - 1; y++) {
    for (int x = 1; x < image.cols - 1; x++) {
      int sum = 0;
      for (int m = -1; m < 2; m++) {
        for (int n = -1; n < 2; n++) {
          sum += (image.at<uchar>(y-n,x-m) * Kernel[n+1][m+1]);
        }
      }
      newImage.at<uchar>(y,x) = sum/30;
    }
  }
}

void calculateGradientMagnitude(Mat Gx, Mat Gy, Mat magnitude) {
  for (int y = 0; y < magnitude.rows; y++) {
    for (int x = 0; x < magnitude.cols; x++) {
      uchar gx2 = pow(Gx.at<uchar>(y, x), 2);
      uchar gy2 = pow(Gy.at<uchar>(y, x), 2);
      uchar value = sqrt(gx2 + gy2);
      magnitude.at<uchar>(y, x) = value;
    }
  }
}

void calculateGradientDirection(Mat Gx, Mat Gy, Mat direction) {
  for (int y = 0; y < direction.rows; y++) {
    for (int x = 0; x < direction.cols; x++) {
      uchar gxval = Gx.at<uchar>(y, x);
      uchar gyval = Gy.at<uchar>(y, x);
      if (gxval != 0) {
        uchar divided = gyval / gxval;
        uchar value = atan(divided) * 100;
        direction.at<uchar>(y, x) = value;
      } else {
        direction.at<uchar>(y, x) = 0;
      }
    }
  }
}

void applyThreshold(Mat magnitude, int threshold) {
  for (int y = 0; y < magnitude.rows; y++) {
    for (int x = 0; x < magnitude.cols; x++) {
      uchar value = magnitude.at<uchar>(y, x);
      if (value < threshold) {
        value = 0;
      } else {
        value = 255;
      }
      magnitude.at<uchar>(y, x) = value;
    }
  }
}

void sobel(cv::Mat &input, Mat magnitude, Mat direction) {
  Mat image = imread(filename, 1);
  Mat Gx;
  Mat Gy;

  cvtColor(image, Gx, CV_BGR2GRAY);
  cvtColor(image, Gy, CV_BGR2GRAY);

  float dx[3][3] = {
    {-1.0, 0.0,  1.0},
    {-2.0, 0.0,  2.0},
    {-1.0, 0.0,  1.0}
  };

  float dy[3][3] = {
    {-1.0, -2.0, -1.0},
    { 0.0,  0.0,  0.0},
    { 1.0,  2.0,  1.0}
  };

  convolution(input, Gx, dx);
  convolution(input, Gy, dy);

  calculateGradientMagnitude(Gx, Gy, magnitude);
  calculateGradientDirection(Gx, Gy, direction);

  imwrite("gx.jpg", Gx);
  imwrite("gy.jpg", Gy);
  imwrite("magnitude.jpg", magnitude);
  imwrite("direction.jpg", direction);
}

AccumulatorStruct getAccumulator(Mat magnitude, Mat direction, int minRadius, int maxRadius) {
  int accumulatorASize = 80;
  int accumulatorBSize = 80;
  AccumulatorStruct acc;

  for (int a = 0; a < accumulatorASize; a++) {
    for (int b = 0; b < accumulatorBSize; b++) {
      for (int r = 0; r < maxRadius - minRadius; r++) {
        acc.array[a][b][r] = 0;
      }
    }
  }

  int largestValue = 0;

  for (int x = 0; x < magnitude.cols; x++) {
    std::cout << x << std::endl;
    for (int y = 0; y < magnitude.rows; y++) {
      if (magnitude.at<uchar>(y, x) != 0) {
        // Draw circle in hough space for each radius
        for (int r = minRadius; r < maxRadius; r++) {
          for (int i = 0; i < 360; i++) {
            int xa = x + (r * cos(i * M_PI / 180));
            int ya = y + (r * sin(i * M_PI / 180));

            int xx = int(round(xa / scale));
            int yy = int(round(ya / scale));

            if (xx < accumulatorASize && yy < accumulatorBSize) {
              if (acc.array[xx][yy][r - minRadius] < 2147483647) {
                acc.array[xx][yy][r - minRadius] += 1;
                if (acc.array[xx][yy][r - minRadius] > largestValue) {
                  largestValue = acc.array[xx][yy][r - minRadius];
                }
              } else {
                std::cout << "overflow accumulator" << std::endl;
              }
            }
          }
        }
      }
    }
  }

  acc.largest = largestValue;

  return acc;
}

void drawHough(AccumulatorStruct &acc, int minRadius, int maxRadius) {
  Mat image;
  Mat hough;
  image = imread(filename, 1);
  cvtColor(image, hough, CV_BGR2GRAY);

  int largestValue = 0;
  int values[80][80];
  for (int x = 0; x < hough.cols; x++) {
    for (int y = 0; y < hough.rows; y++) {
      int value = 0;
      for (int r = minRadius; r < maxRadius; r++) {
        if (value < 2147483647 - acc.array[int(round(x / scale))][int(round(y / scale))][r-minRadius]) {
          value += acc.array[int(round(x / scale))][int(round(y / scale))][r-minRadius];
        } else {
          std::cout << "overflow value" << std::endl;
          value = 2147483647;
        }
      }
      values[int(round(x / scale))][int(round(y / scale))] = value;
      if (value > largestValue) {
        largestValue = value;
      }
    }
  }

  int divideBy = largestValue / 255;
  for (int x = 0; x < hough.cols; x++) {
    for (int y = 0; y < hough.rows; y++) {
      hough.at<uchar>(y, x) = values[int(round(x / scale))][int(round(y / scale))] / divideBy;
    }
  }

  string arg = filename;
  arg.erase(arg.length() - 4);

  imwrite("hough.jpg", hough);
  imwrite("task3outputs/task3_" + arg + "_hough.jpg", hough);
}

bool notTooClose(int x, int y, vector<Coord> maximums, int minDistance) {
  for (int i = 0; i < maximums.size(); i++) {
    int xDistance = (x * scale) - maximums[i].x;
    int yDistance = (y * scale) - maximums[i].y;
    int distance = sqrt((xDistance*xDistance) + (yDistance*yDistance));
    if (distance < minDistance) {
      return false;
    }
  }

  return true;
}

vector<Coord> findMaximums(AccumulatorStruct acc, int width, int height, int minRadius, int maxRadius) {
  vector<Coord> maximums;
  int minDistance = 100;
  int threshold = acc.largest * 0.7;
  std::cout << threshold << std::endl;

  while (true) {
    int maxX = -1;
    int maxY = -1;
    int maxR = -1;
    int maxVal = 0;

    for (int x = 0; x < 80; x++) {
      for (int y = 0; y < 80; y++) {
        if ((x * scale) < width && (y * scale) < height) {
          for (int r = minRadius; r < maxRadius; r++) {
            if (acc.array[x][y][r - minRadius] > maxVal) {
              if (notTooClose(x, y, maximums, minDistance)) {
                maxVal = acc.array[x][y][r - minRadius];
                maxX = x;
                maxY = y;
                maxR = r;
              }
            }
          }
        }
      }
    }

    std::cout << maxVal << std::endl;

    if (maxVal >= threshold) {
      std::cout << "Maximum: " << maxX << ", " << maxY << ", " << maxR << std::endl;
      Coord c;
      c.x = maxX * scale;
      c.y = maxY * scale;
      c.r = maxR;
      maximums.push_back(c);
    } else {
      break;
    }
  }

  return maximums;
}

void setRed(int x, int y, Mat &image) {
  if (x > 0 && y > 0 && x < image.cols && y < image.rows) {
    image.at<Vec3b>(y, x)[0] = 0;
    image.at<Vec3b>(y, x)[1] = 0;
    image.at<Vec3b>(y, x)[2] = 255;
  }
}

void drawCircle(int xc, int yc, int x, int y, Mat &image) {
  setRed(xc+x, yc+y, image);
  setRed(xc-x, yc+y, image);
  setRed(xc+x, yc-y, image);
  setRed(xc-x, yc-y, image);
  setRed(xc+y, yc+x, image);
  setRed(xc-y, yc+x, image);
  setRed(xc+y, yc-x, image);
  setRed(xc-y, yc-x, image);
}

void circleBresenham(int xc, int yc, int r, Mat &image) {
  int x = 0;
  int y = r;
  int d = 3 - 2 * r;

  drawCircle(xc, yc, x, y, image);

  while (y >= x) {
    x++;

    if (d > 0) {
      y--;
      d = d + 4 * (x - y) + 10;
    } else {
      d = d + 4 * x + 6;
    }

    drawCircle(xc, yc, x, y, image);
  }
}

void outputCircles(vector<Coord> maximums) {
  Mat newImage;
  newImage = imread(filename, 1);

  for (int i = 0; i < maximums.size(); i++) {
    circleBresenham(maximums[i].x, maximums[i].y, maximums[i].r, newImage);
  }

  imwrite("circled.jpg", newImage);
}

void saveAccumulator(AccumulatorStruct accumulator) {
  FILE *fout = fopen("accumulator.dat", "w");
  fwrite(&accumulator, sizeof(AccumulatorStruct), 1, fout);
  fclose(fout);
}

AccumulatorStruct loadAccumulator() {
  AccumulatorStruct accumulator;

  FILE *fin = fopen("accumulator.dat", "r");
  fread(&accumulator, sizeof(AccumulatorStruct), 1, fin);
  fclose(fin);

  return accumulator;
}

int calculateScale(Mat image) {
  if (image.cols > image.rows) {
    return int((image.cols * 1.2) / 80);
  } else {
    return int((image.rows * 1.2) / 80);
  }
}

vector<Coord> getHoughCircles(Mat image) {
  Mat gray_image;
  Mat magnitude;
  Mat direction;
  cvtColor(image, gray_image, CV_BGR2GRAY);
  cvtColor(image, magnitude, CV_BGR2GRAY);
  cvtColor(image, direction, CV_BGR2GRAY);

  sobel(gray_image, magnitude, direction);

  string arg = filename;
  arg.erase(arg.length() - 4);

  applyThreshold(magnitude, 6);
  imwrite("magnitude.jpg", magnitude);
  imwrite("task3outputs/task3_" + arg + "_magnitude.jpg", magnitude);

  scale = calculateScale(image);

  int minRadius = 60;
  int maxRadius = 160;
  AccumulatorStruct accumulator;
  accumulator = getAccumulator(magnitude, direction, minRadius, maxRadius);
  drawHough(accumulator, minRadius, maxRadius);
  std::cout << "Hough Drawn" << std::endl;

  vector<Coord> maximums = findMaximums(accumulator, image.cols, image.rows, minRadius, maxRadius);
  std::cout << "Maximums Calculated" << std::endl;
  outputCircles(maximums);
  std::cout << "Circles Drawn" << std::endl;

  return maximums;
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

  if (dartboards.size() == 0) {
    std::cout << "NO DARTBOARDS" << std::endl;
    return false;
  }

  for (int j = 0; j < dartboards.size(); j++) {
    std::cout << getIOU(groundTruth, dartboards[j]) << std::endl;
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

bool containsHoughCircle(Rect dartboard, vector<Coord> &circles) {
  for (int i = 0; i < circles.size(); i++) {
    if (circles[i].x >= dartboard.x && circles[i].x <= dartboard.x + dartboard.width) {
      if (circles[i].y >= dartboard.y && circles[i].y <= dartboard.y + dartboard.height) {
        return true;
      }
    }
  }

  return false;
}

bool intersectsHoughCircle(Rect dartboard, vector<Coord> &circles) {
  for (int i = 0; i < circles.size(); i++) {
    int x = circles[i].x - max(dartboard.x, min(circles[i].x, dartboard.x + dartboard.width));
    int y = circles[i].y - max(dartboard.y, min(circles[i].y, dartboard.y + dartboard.height));
    if ((x * x + y * y) < (circles[i].r * circles[i].r)) {
      return true;
    }
  }

  return false;
}

void detectAndDisplay(Mat frame, vector<Rect> &dartboards, vector<Coord> &circles) {
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	cascade.detectMultiScale(frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500));

  vector<Rect> houghDartboards;
  for (int i = 0; i < dartboards.size(); i++) {
    if (intersectsHoughCircle(dartboards[i], circles)) {
      houghDartboards.push_back(dartboards[i]);
    }
  }
  dartboards = houghDartboards;

	std::cout << dartboards.size() << std::endl;

	for(int i = 0; i < dartboards.size(); i++) {
    rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar(0, 255, 0), 2);
	}
}

int getIDFromFilename(string arg) {
  arg.erase(0, 4);
  arg.erase(arg.length() - 4);

  int id;
  std::istringstream(arg) >> id;

  return id;
}

int main(int argc, const char** argv) {
  filename = argv[1];
  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_COLOR);

  vector<Coord> circles = getHoughCircles(image);

  int id = getIDFromFilename(filename);

  vector<Bound> groundTruth = getGroundTruth(id);

  // Load the Strong Classifier in a structure called `Cascade'
  if(!cascade.load(cascade_name)) {
    printf("--(!)Error loading\n");
    return -1;
  };

  vector<Rect> dartboards;

  detectAndDisplay(image, dartboards, circles);
  std::cout << "Displayed Detected Dartboards" << std::endl;
  overlayGroundTruth(image, groundTruth);
  std::cout << "Overlayed Ground Truth" << std::endl;
  vector<Bound> successfullyDetected = getSuccessfullyDetected(groundTruth, dartboards);
  std::cout << "Found Successfully Detected" << std::endl;
  overlaySuccessfullyDetected(image, successfullyDetected);
  float f1Score = calculateF1Score(groundTruth, successfullyDetected, dartboards);
  std::cout << "Calculated F1 Score" << std::endl;

  imwrite("task3outputs/task3_" + filename, image);
  imwrite("detected.jpg", image);

  return 0;
}

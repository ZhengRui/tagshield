//
//  visualprivacy.hpp
//  SURFDescriptor
//
//  Created by Rui Zheng, Jiayu Su on 15/9/28.
//  Copyright (c) 2015 Rui Zheng, Jiayu Su. All rights reserved.
//

#ifndef SURFDescriptor_visualprivacy_hpp
#define SURFDescriptor_visualprivacy_hpp

#endif

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <termios.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "dirent.h"

bool do_facedet = false, do_fistdet = false, do_markerdet = false;
int mode, portno, objtype;
std::string objpth;
std::vector<std::string> markerspths;
cv::Mat img_scene, img_scene_show, img_marker;
std::vector<cv::KeyPoint> keypoints_scene, keypoints_marker;
std::vector<std::vector<cv::KeyPoint> > keypoints_marker_s;
cv::Mat descriptors_scene, descriptors_marker;
std::vector<cv::Mat> img_marker_s, descriptors_marker_s;
int minHessian = 400;
cv::SurfFeatureDetector detector( minHessian );
cv::SurfDescriptorExtractor extractor;

cv::CascadeClassifier facecascade;
cv::CascadeClassifier fistcascade;
std::vector<cv::Rect> fcobj, fsobj, fcobj_truth;

std::vector<cv::Rect> mkrpps;
cv::RNG rng( 0xFFFFFFFF );
std::string wndname;

static struct termios old, neww;

/* Initialize new terminal i/o settings */
void initTermios(int echo)
{
    tcgetattr(0, &old); /* grab old terminal i/o settings */
    neww = old; /* make new settings same as old settings */
    neww.c_lflag &= ~ICANON; /* disable buffered i/o */
    neww.c_lflag &= echo ? ECHO : ~ECHO; /* set echo mode */
    tcsetattr(0, TCSANOW, &neww); /* use these new terminal i/o settings now */
}

/* Restore old terminal i/o settings */
void resetTermios(void)
{
    tcsetattr(0, TCSANOW, &old);
}

/* Read 1 character - echo defines echo mode */
char getch_(int echo)
{
    char ch;
    initTermios(echo);
    ch = getchar();
    resetTermios();
    return ch;
}

/* Read 1 character without echo */
char getch(void)
{
    return getch_(0);
}

/* Read 1 character with echo */
char getche(void)
{
    return getch_(1);
}

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool has_prefix(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() &&
    str.compare(0, prefix.size(), prefix) == 0;
}

void readme(std::string exename) {
    std::cout << "Usage: " << exename << " config_file \n" << std::endl;
    std::cout << "config_file help: \n" << std::endl;
    std::cout << "* [task]\n*   xxx\n*   |||\n*   |||\n*   ||-   face detection, 0 no, 1 yes\n*   |--   fist detection, 0 no, 1 yes\n*   |-- marker detection, 0 no, 1 yes\n\n* [markerpaths] (ignored if marker detection is disabled)\n\n* [mode]\n*   0 serve as server, 1 run locally\n\n* [objtype] (ignored if serve as server)\n*   0 image, 1 folder, 2 video, 3 cam\n\n* [objpath] (ignored if serve as server)\n" << std::endl;
    std::cout << "\nconfig_file example: \n" << std::endl;
    std::cout << "task = 111\n\nmarkerpaths = [\"/path/to/marker1.jpg\", \"/path/to/marker2.jpg\"]\n\nmode = 1\n\nobjtype = 2\n\nobjpath = \"/path/to/object\"\n" << std::endl;
}

void readConfig(std::string config_pth);
void prepare();
void loadmarkers();
void runServer();
void runLocal();
void domosaic(cv::Mat m, cv::Rect r);
void facedet();
void fistdet();
void markerdet();
void imgproc();
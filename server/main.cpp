//
//  Created by Rui Zheng, Jiayu Su on 15/9/28.
//  Copyright (c) 2015 Rui Zheng, Jiayu Su. All rights reserved.
//

#include "visualprivacy.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    if (argc <= 1) {
        readme(argv[0]);
        return -1;
    }

    readConfig(argv[1]);

    if (mode) {
        prepare();
        runLocal();
    } else {
        runServer();
    }

    destroyAllWindows();
    return 0;
}

void readConfig(string pth) {
    ifstream cfgfile(pth);
    string line;
    while(getline(cfgfile, line)) {
        boost::algorithm::trim(line);
        size_t p_m = line.find("=");
        if (!has_prefix(line, "*") && p_m != string::npos) {
            string key = line.substr(0, p_m);
            boost::algorithm::trim_right(key);
            string val = line.substr(p_m+1, line.size()-p_m-1);
            boost::algorithm::trim_left(val);
            if (key == "task") {
                if (val[0] - '0') do_facedet = true;
                if (val[1] - '0') do_fistdet = true;
                if (val[2] - '0') do_markerdet = true;
                continue;
            } else if (key == "markerpaths") {
                string tmp = val.substr(1, val.size()-2);
                boost::replace_all(tmp, "\"", "");
                boost::split(markerspths, tmp, boost::is_any_of(","));
                for (size_t i=0; i<markerspths.size(); i++) {
                    boost::algorithm::trim(markerspths[i]);
                }
                continue;
            } else if (key == "mode") {
                mode = stoi(val);
                continue;
            } else if (key == "port") {
                portno = stoi(val);
                continue;
            } else if (key == "objtype") {
                objtype = stoi(val);
                continue;
            } else if (key == "objpath") {
                boost::replace_all(val, "\"", "");
                objpth = val;
                continue;
            } else {}
        }
    }
}

void prepare() {
    if (do_facedet) facecascade.load( "/Users/zerry/Work/CVFun/CVHandDet/haarcascade_frontalface_alt2.xml");
    if (do_fistdet) fistcascade.load( "/Users/zerry/Work/CVFun/CVHandDet/aGest.xml");
    if (do_markerdet) loadmarkers();
    if (mode) {
        switch (objtype) {
            case 0:
                wndname = "Image";
                break;
            case 1:
                wndname = "Images";
                break;
            case 2:
                wndname = "Video";
                break;
            default:
                wndname = "Cam";
                break;
        }
        namedWindow(wndname, WINDOW_AUTOSIZE);
    } else {
        wndname = "RemoteStream";
    }
}

void loadmarkers() {
    for (int i=0; i < markerspths.size(); i++) {
        img_marker = imread( markerspths[i], CV_LOAD_IMAGE_GRAYSCALE );

        int maxL = max(img_marker.rows, img_marker.cols);
        if (maxL > 400) {
            int scale = maxL / 300;
            resize(img_marker, img_marker, Size(img_marker.cols / scale, img_marker.rows / scale), INTER_AREA);
        }
        //cout << img_marker.rows << "x" << img_marker.cols << endl;
        detector.detect( img_marker, keypoints_marker );
        extractor.compute( img_marker, keypoints_marker, descriptors_marker );

        img_marker_s.push_back(img_marker);
        keypoints_marker_s.push_back(keypoints_marker);
        descriptors_marker_s.push_back(descriptors_marker);
    }
}

void runServer() {
    int sockfd, newsockfd, pid;
    socklen_t clilen;
    int bufsize = 100000;
    int frmbufsize = 6220800 / 4;
    int frmrow = 1080 / 2;
    int frmcol = 1920 / 2;
    struct sockaddr_in serv_addr, cli_addr;
    int n, size_recv;
    char buffer[bufsize], frmbuffer[frmbufsize];

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (::bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR on binding");
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    signal(SIGCHLD, SIG_IGN);  // Way 1 non-blocking: ignore SIGCHLD signal sent from child death

    while (1) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0)
            error("ERROR on accept");
        pid = fork();
        //printf("PID: %d, CPId: %d\n", getpid(), pid);
        if (pid < 0)
            error("ERROR on fork");
        if (pid == 0) {
            close(sockfd);
            prepare();              // *** must put here in sub process, the resize(), detect() and compute() in loadmarkers() will make the following namedWindow() blocking if prepare() is put in main thread ***
            namedWindow(wndname, WINDOW_AUTOSIZE);
            while (1) {
                size_recv = 0;
                bzero(frmbuffer, frmbufsize);
                while (1) {
                    bzero(buffer, bufsize);
                    n = recv(newsockfd, buffer, min(frmbufsize - size_recv, bufsize), 0);
                    if (n < 0) error("ERROR reading from socket");
                    else {
                        if (size_recv + n < frmbufsize) {
                            memcpy(&frmbuffer[size_recv], &buffer, n);
                            //printf("Chunk size: %d , total size : %d \n", n, size_recv);
                            size_recv += n;
                        } else {
                            memcpy(&frmbuffer[size_recv], &buffer, frmbufsize - size_recv);
                            assert(n == frmbufsize - size_recv);
                            //printf("For last chunk, remains: %d, actually received: %d\n", frmbufsize - size_recv, n);
                            //printf("One frame received\n");
                            break;
                        }
                    }

                    if (waitKey(30) == 27) {
                        destroyAllWindows();
                        printf("Kick off a client :-) \n");
                        n = write(newsockfd,"Bye from server",15);
                        if (n < 0) error("ERROR writing to socket");
                        exit(0);
                    }
                }

                img_scene = Mat(frmrow, frmcol, CV_8UC3, frmbuffer);
                img_scene_show = img_scene;

                // do frame processing
                imgproc();
                imshow(wndname, img_scene_show);
            }
        }
        else {
            //  Way 2: using wait() or waitpid()
            //  wait(NULL); // wait() is blocking
            //  waitpid(pid, NULL, WNOHANG);  // using WNOHANG option to make waitpid() non-blocking
            //  printf("Be careful of Zombie.\n"); // without any treatment (wait / waitpid / signal), child process becomes zombie process after exit(0)

            close(newsockfd);
        }
    }
    //close(sockfd);
}

void runLocal() {
    if (objtype == 0) {
        img_scene = imread(objpth, CV_LOAD_IMAGE_COLOR);
        img_scene_show = img_scene;
        // do image processing
        imgproc();
        if (max(img_scene_show.rows, img_scene_show.cols) > 1000) {
            int scale = max(img_scene_show.rows, img_scene_show.cols) / 800;
            resize(img_scene_show, img_scene_show, Size(img_scene_show.cols / scale, img_scene_show.rows / scale), INTER_AREA);
        }
        imshow(wndname, img_scene_show);
        waitKey(0);
    } else if (objtype == 1) {
        DIR *dir;
        struct dirent *ent;
        vector<string> imgpths;

        if ((dir = opendir (objpth.c_str())) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                if (has_suffix(ent->d_name, ".jpg") || has_suffix(ent->d_name, ".JPG") || has_suffix(ent->d_name, ".png") || has_suffix(ent->d_name, ".PNG")) {
                    //printf ("%s\n", ent->d_name);
                    imgpths.push_back(ent->d_name);
                }
            }
            closedir (dir);
        } else {
            /* could not open directory */
            perror ("could not open directory");
        }

        int skip = 0;

        for ( int k=0; k<imgpths.size(); ) {
            if (skip) {
                cout << "skip " << imgpths[k] << endl;
            } else {
                img_scene = imread( objpth + "/" + imgpths[k], CV_LOAD_IMAGE_COLOR );
                if (!img_scene.data)
                    break;
                img_scene_show = img_scene;
                // do image processing
                imgproc();
                if (max(img_scene_show.rows, img_scene_show.cols) > 1000) {
                    int scale = max(img_scene_show.rows, img_scene_show.cols) / 800;
                    resize(img_scene_show, img_scene_show, Size(img_scene_show.cols / scale, img_scene_show.rows / scale), INTER_AREA);
                }
                imshow(wndname, img_scene_show);
                waitKey(1);
            }

            char move = getch();
            if (move == 44) {
                k--;
                skip = 0;
                continue;
            } else if (move == 46) {
                cout << imgpths[k] << " -- " << img_scene.cols << "x" << img_scene.rows << endl;
                k++;
                skip = 0;
                continue;
            } else if (move == 32) {
                k++;
                skip = 1;
                continue;
            } else {
                break;
            }
        }
    } else if (objtype == 2) {
        VideoCapture cap(objpth);
        if (!cap.isOpened())
            error("Failed to read video file!");
        for ( ; ; ) {
            cap >> img_scene;
            if (!img_scene.data)
                break;
            img_scene_show = img_scene;
            // do image processing
            imgproc();
            if (max(img_scene_show.rows, img_scene_show.cols) > 1000) {
                int scale = max(img_scene_show.rows, img_scene_show.cols) / 800;
                resize(img_scene_show, img_scene_show, Size(img_scene_show.cols / scale, img_scene_show.rows / scale), INTER_AREA);
            }
            imshow(wndname, img_scene_show);
            if (waitKey(20) >= 0)
                break;
        }
    } else {
        VideoCapture cap(CV_CAP_ANY);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640 );
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480 );
        if (!cap.isOpened())
            error("Failed to open camera!");
        for ( ; ; ) {
            cap >> img_scene;
            if (!img_scene.data)
                break;
            img_scene_show = img_scene;
            // do image processing
            imgproc();
            imshow(wndname, img_scene_show);
            if (waitKey(20) >= 0)
                break;
        }
    }
}


void domosaic(Mat m, Rect r) {
    Mat roi = m(r);
    int W_mosaic = 18;
    int H_mosaic = 18;
    for (int i = W_mosaic; i < roi.cols; i += W_mosaic) {
        for (int j = H_mosaic; j < roi.rows; j += H_mosaic) {
            uchar s = roi.at<uchar>(j - H_mosaic / 2, (i - W_mosaic / 2) * 3);
            uchar s1 = roi.at<uchar>(j - H_mosaic / 2, (i - W_mosaic / 2) * 3 + 1);
            uchar s2 = roi.at<uchar>(j - H_mosaic / 2, (i - W_mosaic / 2) * 3 + 2);
            for (int ii = i - W_mosaic; ii <= i; ii++) {
                for (int jj = j - H_mosaic; jj <= j; jj++) {
                    roi.at<uchar>(jj, ii * 3 + 0) = s;
                    roi.at<uchar>(jj, ii * 3 + 1) = s1;
                    roi.at<uchar>(jj, ii * 3 + 2) = s2;
                }
            }
        }
    }
}

void facedet() {
    int fdscale = 4;
    Mat img_scene_fd_color, img_scene_fd;
    resize(img_scene, img_scene_fd_color, Size(img_scene.cols / fdscale, img_scene.rows / fdscale), INTER_AREA);
    cvtColor(img_scene_fd_color, img_scene_fd, CV_BGR2GRAY);
    facecascade.detectMultiScale( img_scene_fd, fcobj, 1.1, 5, CV_HAAR_SCALE_IMAGE, cvSize(20, 20) );
    int facenum = 0;
    for( vector<Rect>::const_iterator r = fcobj.begin(); r != fcobj.end(); r++ ) {
        int tlx = r->x * fdscale;
        int tly = r->y * fdscale;
        int brx = (r->x + r->width) * fdscale;
        int bry = (r->y + r->height) * fdscale;
        Mat facepp_hsv;
        cvtColor(img_scene_fd_color(*r), facepp_hsv, CV_BGR2HSV);
        Mat skinmsk;
        inRange(facepp_hsv, Scalar(0, 10, 60), Scalar(30, 150, 255), skinmsk);
        if (sum(skinmsk)[0] / (skinmsk.rows * skinmsk.cols * 255.0) > 0.2) {
            rectangle( img_scene_show, cvPoint( tlx, tly ), cvPoint( brx, bry ), CV_RGB( 0, 0, 255 ), 3, 8, 0 );
            facenum++;
            if (do_markerdet) {
                Rect mkrpp = Rect(Point(max(tlx-25, 0), bry + 20), Point(min(brx+25, img_scene_show.cols), img_scene_show.rows - 10));
                //Rect mkrpp = Rect(Point(max(tlx-25, 0), bry + 20), Point(min(brx+25, img_scene_show.cols), img_scene_show.rows - 200));
                //if (mkrpp.area() > 50000)
                mkrpps.push_back(mkrpp);
                fcobj_truth.push_back(Rect(cvPoint( tlx, tly ), cvPoint( brx, bry )));
            }
        }
    }
}

void fistdet() {
    int fsscale = 3;
    Mat img_scene_fs_color, img_scene_fs;
    resize(img_scene, img_scene_fs_color, Size(img_scene.cols / fsscale, img_scene.rows / fsscale), INTER_AREA);
    cvtColor(img_scene_fs_color, img_scene_fs, CV_BGR2GRAY);
    fistcascade.detectMultiScale( img_scene_fs, fsobj, 1.1, 3, CV_HAAR_SCALE_IMAGE, cvSize(20, 20) );
    int fistnum = 0;
    for( vector<Rect>::const_iterator r = fsobj.begin(); r != fsobj.end(); r++ ) {
        int tlx = r->x * fsscale;
        int tly = r->y * fsscale;
        int brx = (r->x + r->width) * fsscale;
        int bry = (r->y + r->height) * fsscale;
        Mat fistpp_hsv;
        cvtColor(img_scene_fs_color(*r), fistpp_hsv, CV_BGR2HSV);
        Mat skinmsk;
        inRange(fistpp_hsv, Scalar(0, 10, 60), Scalar(30, 150, 255), skinmsk);
        if (sum(skinmsk)[0] / (skinmsk.rows * skinmsk.cols * 255.0) > 0.2 /*&& r->width * 1.0 / img_scene_fs.cols < 0.08*/) {
            rectangle( img_scene_show, cvPoint( tlx, tly ), cvPoint( brx, bry ), CV_RGB( 0, 255, 0 ), 3, 8, 0 );
            fistnum++;
        }
    }
}

void markerdet() {
    mkrpps.push_back(Rect(0, 0, img_scene.cols, img_scene.rows));
    int marknum = 0;
    for (int r = 0; r < mkrpps.size()-1; r++) {
        rectangle(img_scene_show, mkrpps[r].tl(), mkrpps[r].br(), Scalar(255,255,255), 3, 8, 0);
        int mkscale;
        Mat img_scene_mkr_color, img_scene_mkr;
        if (r == mkrpps.size() - 1) {
            mkscale = 1;
            resize(img_scene, img_scene_mkr_color, Size(img_scene.cols / mkscale, img_scene.rows / mkscale), INTER_AREA);
        } else {
            mkscale = 1;
            resize(img_scene(mkrpps[r]), img_scene_mkr_color, Size(mkrpps[r].width / mkscale, mkrpps[r].height / mkscale), INTER_AREA);
        }
        cvtColor(img_scene_mkr_color, img_scene_mkr, CV_BGR2GRAY);

        detector.detect( img_scene_mkr, keypoints_scene );
        extractor.compute( img_scene_mkr, keypoints_scene, descriptors_scene );

        if (keypoints_scene.size() < 4) continue;
        FlannBasedMatcher matcher;
        vector< DMatch > matches;

        for (int m=0; m < descriptors_marker_s.size(); m++) {
            descriptors_marker = descriptors_marker_s[m];
            keypoints_marker = keypoints_marker_s[m];
            img_marker = img_marker_s[m];

            matcher.match( descriptors_marker, descriptors_scene, matches );
            double max_dist = 0; double min_dist = 100;

            //-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < descriptors_marker.rows; i++ )
            { double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }

            //printf("-- Max dist : %f \n", max_dist );
            //printf("-- Min dist : %f \n", min_dist );

            //-- Pick only "good" matches (i.e. whose distance is less than 3*min_dist )
            vector< DMatch > good_matches;
            vector< DMatch > good_matches_nxt;

            for( int i = 0; i < descriptors_marker.rows; i++ ) {
                if( matches[i].distance < 3*min_dist ) {
                    good_matches.push_back( matches[i]);
                }
            }

            //cout << "good matches size: " << good_matches.size() << endl;
            if (good_matches.size() < 4) continue;

            //-- Localize the object
            vector<Point2f> obj;
            vector<Point2f> scene;
            vector<Point2f> scene_proj;

            //while (good_matches.size() > 5) {

                for( int i = 0; i < good_matches.size(); i++ ) {
                    //-- Get the keypoints from the good matches
                    obj.push_back( keypoints_marker[ good_matches[i].queryIdx ].pt );
                    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
                }

                Mat inliers;
                Mat H = findHomography( obj, scene, CV_RANSAC, 3, inliers );


                vector<Point2f> obj_corners(4);
                obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_marker.cols, 0 );
                obj_corners[2] = cvPoint( img_marker.cols, img_marker.rows ); obj_corners[3] = cvPoint( 0, img_marker.rows );
                vector<Point2f> scene_corners(4);

                perspectiveTransform( obj_corners, scene_corners, H);

                // line( img_scene_show, mkscale * scene_corners[0] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[1] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 255, 255, 255), 4 );
                // line( img_scene_show, mkscale * scene_corners[1] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[2] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 255, 255, 255), 4 );
                // line( img_scene_show, mkscale * scene_corners[2] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[3] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 255, 255, 255), 4 );
                // line( img_scene_show, mkscale * scene_corners[3] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[0] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 255, 255, 255), 4 );

                // cout << "Area: " << contourArea(scene_corners) << endl;
                if (contourArea(scene_corners) < 5000 || !isContourConvex(scene_corners) /*|| contourArea(scene_corners) > 40000*/) {
                    // break;   // if in while for multiple marker detection of same type in each proposal
                    continue;   // if in for with while commented for single marker detection of same type in each proposal
                }

                marknum++;
                line( img_scene_show, mkscale * scene_corners[0] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[1] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 0, 0, 255), 4 );
                line( img_scene_show, mkscale * scene_corners[1] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[2] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 0, 0, 255), 4 );
                line( img_scene_show, mkscale * scene_corners[2] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[3] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 0, 0, 255), 4 );
                line( img_scene_show, mkscale * scene_corners[3] + Point2f(mkrpps[r].x, mkrpps[r].y), mkscale * scene_corners[0] + Point2f(mkrpps[r].x, mkrpps[r].y), Scalar( 0, 0, 255), 4 );

                // put mosaic on face
                if (r < mkrpps.size()-1)
                    domosaic(img_scene_show, fcobj_truth[r]);

                // good_matches_nxt.clear();

                // use mask returned from findHomography() function
                // for (int i = 0; i < good_matches.size(); i++ ) {
                //     if ( inliers.at<uchar>(i,0) == 0) good_matches_nxt.push_back(good_matches[i]);
                // }

                // find inliers by comparing distances between matched keypoint on the scene and its projected point on the scene using H
                // perspectiveTransform(obj, scene_proj, H);
                // for (int i = 0; i < good_matches.size(); i++ ) {
                //     if (norm(scene[i] - scene_proj[i]) > 5.0) good_matches_nxt.push_back(good_matches[i]);
                // }
                //
                // good_matches = good_matches_nxt;
                // obj.clear();
                // scene.clear();
            //}
            matches.clear();
        }
    }
    mkrpps.clear();
    fcobj_truth.clear();
}

void imgproc() {
    if (do_facedet) facedet();
    if (do_fistdet) fistdet();
    if (do_markerdet) markerdet();
}

package com.rzheng.visualprivacy;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class MainActivity extends AppCompatActivity {

    private SurfaceView mPreview;
    private SurfaceHolder mPreviewHolder;
    private Camera mCamera;
    private boolean mInPreview = false;
    private boolean mCameraConfigured = false;
    private Size size;

    private static final String TAG = "Visual Privacy";
    private byte[] callbackBuffer;
    private Queue<Integer> tskQueue = new LinkedList<Integer>();

    private Socket socket;
    private OutputStream oStream;

    private CascadeClassifier mFaceDetector;
    private CascadeClassifier mFistDetector;

    private List<Rect> mDrawFaces = new ArrayList<Rect>();
    private List<Rect> mDrawHiddenFaces = new ArrayList<Rect>();
    private Rect[] mDrawFists;
    private List<Point[]> mDrawMarks = new ArrayList<Point[]>();
    private DrawOnTop mDraw;

    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 0, 255, 255);
    private static final Scalar HAND_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/VisualPrivacy/";
    private String[] mkrNames = new String[]{"card.jpg", "privacy2.jpg", "hkust.jpg", "sunflower.jpg", "starsky.jpg"};

    private frmTransmissionThread fTTh;
    private boolean isReady;
    private boolean fstFrame;
    private boolean shouldContinue;
    private boolean useThread = true;
    private int maxAsyncTsks = 1;
    private int sktScale = 6;
    private int front1back0 = 1;

    private Mat img_marker, descriptors_marker;
    private MatOfKeyPoint keypoints_marker;
    private List<Mat> img_marker_s = new ArrayList<Mat>();
    private List<Mat> descriptors_marker_s = new ArrayList<Mat>();
    private List<MatOfKeyPoint> keyPoints_marker_s = new ArrayList<MatOfKeyPoint>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initializeDependencies();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeDependencies() {

        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            mFaceDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (mFaceDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mFaceDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Face detector initialization failed: " + e);
        }

        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_fist2);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_fist2.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            mFistDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (mFistDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mFistDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Fist detector initialization failed: " + e);
        }
    }

    static {
        System.loadLibrary("opencv_java");
        System.loadLibrary("nonfree");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, " onCreate() called.");

        File dir = new File(DATA_PATH);
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                Log.v(TAG, "ERROR: Creation of directory " + DATA_PATH + " on sdcard failed");
                return;
            } else {
                Log.v(TAG, "Created directory " + DATA_PATH + " on sdcard");
            }
        }

        for (int i=0; i<mkrNames.length; i++) {
            String mkrname = mkrNames[i];
            if (!(new File(DATA_PATH + mkrname)).exists()) {
                try {
                    AssetManager assetManager = getAssets();
                    InputStream in = assetManager.open(mkrname);
                    OutputStream out = new FileOutputStream(DATA_PATH
                            + mkrname);
                    byte[] buf = new byte[1024];
                    int len;
                    while ((len = in.read(buf)) > 0) {
                        out.write(buf, 0, len);
                    }
                    in.close();
                    out.close();
                    Log.v(TAG, "Copied " + mkrname);
                } catch (IOException e) {
                    Log.e(TAG, "Was unable to copy " + mkrname + e.toString());
                }
            }
        }

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        mPreview = (SurfaceView) findViewById(R.id.preview);
        mPreviewHolder = mPreview.getHolder();
        mPreviewHolder.addCallback(surfaceCallback);

        mDraw = new DrawOnTop(this);
        addContentView(mDraw, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        if (Build.VERSION.SDK_INT < 16) { //ye olde method
            getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                    WindowManager.LayoutParams.FLAG_FULLSCREEN);
        } else { // Jellybean and up, new hotness
            View decorView = getWindow().getDecorView();
            // Hide the status bar.
            int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
            decorView.setSystemUiVisibility(uiOptions);
        }

        // opencv version number should <= inner version from the output (here is 2.4.9),
        // otherwise it reminds you to install opencv manager (always failed unless first uninstall
        // then install a higher version)
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);

        isReady = false;
        fstFrame = true;
        shouldContinue = true;
        new socketCreationTask("10.89.151.229", 51717).execute();
    }

    SurfaceHolder.Callback surfaceCallback = new SurfaceHolder.Callback() {
        public void surfaceCreated(SurfaceHolder holder) {
            Log.i(TAG, " surfaceCreated() called.");
        }

        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            Log.i(TAG, " surfaceChanged() called.");
            initPreview(width, height);
            startPreview();
        }

        public void surfaceDestroyed(SurfaceHolder holder) {
            Log.i(TAG, " surfaceDestroyed() called.");
        }
    };

    Camera.PreviewCallback frameIMGProcCallback = new Camera.PreviewCallback() {

        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            isReady = true;

            if (useThread) {
                if (fTTh != null) {
                    fTTh.setFrm(data);
                }
            } else {
                // use asynctask, too many asynctasks will be very resources consuming
                // if (tskQueue.peek() != null && oStream != null) {
                //     new frmTransmissionTask(tskQueue.poll()).execute(data);   // send data to server throught wifi using AsyncTask
                // }
            }

            mDraw.invalidate();
            mCamera.addCallbackBuffer(callbackBuffer);
        }
    };

    private class socketCreationTask extends AsyncTask<Void, Void, Void> {
        String desAddress;
        int dstPort;

        socketCreationTask(String addr, int port) {
            this.desAddress = addr;
            this.dstPort = port;
        }

        @Override
        protected Void doInBackground(Void... arg) {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (isReady) {

                for (int i = 1; i <= maxAsyncTsks; i++) tskQueue.add(i);
                fTTh = new frmTransmissionThread();
                Log.i(TAG, "frmTransmissionThread created");
                fTTh.start();

                try {
                    socket = new Socket(desAddress, dstPort);
                    Log.i(TAG, "Socket established");
                    oStream = socket.getOutputStream();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return null;
        }
    }

    private class frmTransmissionTask extends AsyncTask<byte[], Void, Void> {
        private int tskId;

        public frmTransmissionTask(int tskId) {
            this.tskId = tskId;
            //Log.i(TAG, "Asynctask - " + tskId + " started.");
        }

        @Override
        protected Void doInBackground(byte[]... frmdata) {
            byte[] res = frameProc(frmdata[0]);
            if (oStream != null) {   // oStream maybe set to null by previous failed asynctask
                try {
                    // 0. test socket
                    // oStream.write("Hello from VisualPrivacy".getBytes());

                    // 1. do yuv2rgb in android
                    oStream.write(res);

                    // 2. send raw data, do everything on server
                    // oStream.write(frmdata[0]);


                    //Log.i(TAG, "Asynctask - " + tskId + " succeeded.");
                    // try {
                    //     Thread.sleep(5000);
                    // } catch (InterruptedException e) {
                    //     e.printStackTrace();
                    // }
                } catch (IOException e) {
                    e.printStackTrace();
                    if (socket != null) {
                        Log.i(TAG, "Connection lost.");
                        try {
                            oStream.close();
                            socket.close();
                            oStream = null;
                            socket = null;
                        } catch (IOException e1) {
                            e1.printStackTrace();
                        }
                    }
                    //Log.i(TAG, "Asynctask - " + tskId + " failed.");
                }
            } else {
                //Log.i(TAG, "Asynctask - " + tskId + " skipped.");
            }
            return null;
        }

        @Override
        public void onPostExecute(Void result) {
            //Log.i(TAG, "Asynctask - " + tskId + " ended.");
            tskQueue.add(tskId);
        }
    }

    private class frmTransmissionThread extends Thread {
        private byte[] frm;

        public void run() {
            // Log.i(TAG, "frmTransmissionThread run() started");
            while (shouldContinue) {
                if (frm != null) {
                    byte[] res = frameProc(frm);
                }

                if (oStream != null) {
                    try {
                        oStream.write(frameProc(frm));
                        //    Log.i(TAG, "One frame sent");
                    } catch (IOException e) {
                        e.printStackTrace();
                        if (socket != null) {
                            Log.i(TAG, "Connection lost in frmTransmissionThread.");
                            try {
                                oStream.close();
                                socket.close();
                                oStream = null;
                                socket = null;
                            } catch (IOException e1) {
                                e1.printStackTrace();
                            }
                        }
                    }
                }
            }
            // Log.i(TAG, "frmTransmissionThread run() ended");
        }

        public void setFrm(byte[] bytes) {
            frm = bytes;
        }
    }



    private byte[] frameProc(byte[] framedata) {

        Mat YUVMat = new Mat(size.height + size.height / 2, size.width, CvType.CV_8UC1);
        YUVMat.put(0, 0, framedata);

        Mat BGRMat = new Mat(size.height, size.width, CvType.CV_8UC3);
        Imgproc.cvtColor(YUVMat, BGRMat, Imgproc.COLOR_YUV420sp2BGR);
        Mat BGRMat_orig = BGRMat.clone();

        // 1. faces detection
        int fdscale = 2;
        Mat BGRMat4fd = new Mat(size.height/fdscale, size.width/fdscale, CvType.CV_8UC3);
        Imgproc.resize(BGRMat_orig, BGRMat4fd, BGRMat4fd.size(), 0, 0, Imgproc.INTER_AREA);
        Mat GRAYMat4fd = new Mat(BGRMat4fd.rows(), BGRMat4fd.cols(), CvType.CV_8UC1);
        Imgproc.cvtColor(BGRMat4fd, GRAYMat4fd, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.equalizeHist(GRAYMat4fd, GRAYMat4fd);

        MatOfRect faces = new MatOfRect();
        if (mFaceDetector != null) {
            mFaceDetector.detectMultiScale(GRAYMat4fd, faces, 1.1, 5, Objdetect.CASCADE_SCALE_IMAGE,
                    new org.opencv.core.Size(20, 20), new org.opencv.core.Size());
        }

        Rect[] faceList = faces.toArray();
        List<Rect> mDrawFacesCache = new ArrayList<Rect>();
        List<Rect> mkrpps = new ArrayList<Rect>();
        for (int i=0; i<faceList.length; i++) {
            Rect faceROI = faceList[i];
            Mat facepp_hsv = new Mat(faceROI.height, faceROI.width, CvType.CV_8UC3);
            Imgproc.cvtColor(BGRMat4fd.submat(faceROI), facepp_hsv, Imgproc.COLOR_BGR2HSV);
            Mat skinmsk = new Mat(facepp_hsv.rows(), facepp_hsv.cols(), CvType.CV_8U);
            Core.inRange(facepp_hsv, new Scalar(0, 10, 60), new Scalar(30, 150, 255), skinmsk);
            if (Core.sumElems(skinmsk).val[0] / (skinmsk.total() * 255.0) > 0.2) {
                Rect faceROI_truth = new Rect(new Point(faceROI.tl().x * fdscale, faceROI.tl().y * fdscale),
                        new Point(faceROI.br().x * fdscale, faceROI.br().y * fdscale));
                mDrawFacesCache.add(faceROI_truth);
                Rect mkrpp = new Rect(new Point(Math.max(faceROI_truth.tl().x-25, 0), faceROI_truth.br().y+20),
                        new Point(Math.min(faceROI_truth.br().x+25, size.width), size.height-10));
                mkrpps.add(mkrpp);
            }
        }

        mDrawFaces = mDrawFacesCache;
        //Log.i(TAG, "Detected " + mDrawFaces.size() + " faces" );
        for (int i = 0; i < mDrawFaces.size(); i++)
            Core.rectangle(BGRMat, mDrawFaces.get(i).tl(), mDrawFaces.get(i).br(), FACE_RECT_COLOR, 3);


        /*
        // 2. fists detection
        Mat GRAYMat4fs = new Mat(size.height, size.width, CvType.CV_8UC1);
        Imgproc.cvtColor(BGRMat_orig, GRAYMat4fs, Imgproc.COLOR_BGR2GRAY);
        MatOfRect fists = new MatOfRect();
        if (mFistDetector != null) {
            mFistDetector.detectMultiScale(GRAYMat4fs, fists, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE,
                    new org.opencv.core.Size(80, 80), new org.opencv.core.Size());
        }

        mDrawFists = fists.toArray();
        //Log.i(TAG, "Detected " + mDrawFists.length + "hands");
        for (int i = 0; i < mDrawFists.length; i++)
            Core.rectangle(BGRMat, mDrawFists[i].tl(), mDrawFists[i].br(), HAND_RECT_COLOR, 3);

        */

        // 3. markers detection

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        List<Point[]> mDrawMarksCache = new ArrayList<Point[]>();
        List<Rect> mDrawHiddenFacesCache = new ArrayList<Rect>();

        if (fstFrame) {
            for (int i=0; i<mkrNames.length; i++) {
                img_marker = Highgui.imread(DATA_PATH + mkrNames[i], Highgui.CV_LOAD_IMAGE_GRAYSCALE);

                int maxL = Math.max(img_marker.rows(), img_marker.cols());
                if (maxL > 400) {
                    int scale = maxL / 300;
                    Imgproc.resize(img_marker, img_marker, new org.opencv.core.Size(img_marker.cols()/scale, img_marker.rows()/scale), 0, 0, Imgproc.INTER_AREA);
                }

                keypoints_marker = new MatOfKeyPoint();
                detector.detect(img_marker, keypoints_marker);
                descriptors_marker = new Mat();
                extractor.compute(img_marker, keypoints_marker, descriptors_marker);

                img_marker_s.add(img_marker);
                keyPoints_marker_s.add(keypoints_marker);
                descriptors_marker_s.add(descriptors_marker);
            }
        }

        mkrpps.add(new Rect(0,0,size.width,size.height));
        Log.i(TAG, "--- marker proposals number: " + mkrpps.size() + " ---");

        int mkscale;
        Mat BGRMat4mkd;
        for (int r=0; r<mkrpps.size()-1; r++) {
            Log.i(TAG, "+++ proposal " + (r+1) + " +++");
            if (r == mkrpps.size() - 1) {
                mkscale = 2;
                BGRMat4mkd = new Mat(size.height / mkscale, size.width / mkscale, CvType.CV_8UC3);
                Imgproc.resize(BGRMat_orig, BGRMat4mkd, BGRMat4mkd.size(), 0, 0, Imgproc.INTER_AREA);
            } else {
                mkscale = 1;
                BGRMat4mkd = new Mat(mkrpps.get(r).height / mkscale, mkrpps.get(r).width / mkscale, CvType.CV_8UC3);
                Imgproc.resize(BGRMat_orig.submat(mkrpps.get(r)), BGRMat4mkd, BGRMat4mkd.size(), 0, 0, Imgproc.INTER_AREA);
            }

            Mat GRAYMat4mkd = new Mat(BGRMat4mkd.rows(), BGRMat4mkd.cols(), CvType.CV_8UC1);
            Imgproc.cvtColor(BGRMat4mkd, GRAYMat4mkd, Imgproc.COLOR_BGR2GRAY);

            // Step 1: Detect the keypoints using SURF Detector
            MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
            detector.detect(GRAYMat4mkd, keypoints_scene);

            if (keypoints_scene.rows() < 4) continue;

            // Step 2: Calculate descriptors (feature vectors)
            Mat descriptors_scene = new Mat();
            extractor.compute(GRAYMat4mkd, keypoints_scene, descriptors_scene);

            // Step 3: Matching descriptor vectors using FLANN matcher
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
            MatOfDMatch matches = new MatOfDMatch();

            int mkrnum = 0;
            for (int m=0; m<descriptors_marker_s.size(); m++) {
                Log.i(TAG, "*** marker " + (m+1) + " ***");
                descriptors_marker = descriptors_marker_s.get(m);
                keypoints_marker = keyPoints_marker_s.get(m);
                img_marker = img_marker_s.get(m);

                matcher.match(descriptors_marker, descriptors_scene, matches);

                // Quick calculation of max and min distances between keypoints
                DMatch[] tmp1 = matches.toArray();
                double max_dist = 0;
                double min_dist = 100;
                for (int i = 0; i < descriptors_marker.rows(); i++) {
                    double dist = tmp1[i].distance;
                    if (dist < min_dist) min_dist = dist;
                    if (dist > max_dist) max_dist = dist;
                }
                //Log.i(TAG, "Max dist: " + max_dist + "; Min dist: " + min_dist);

                // Draw only good matches (i.e. whose distance is less than 3*min_dist)
                List<DMatch> good_matches = new ArrayList<DMatch>();
                List<DMatch> good_matches_nxt = new ArrayList<DMatch>();

                for (int i = 0; i < descriptors_marker.rows(); i++) {
                    if (tmp1[i].distance < 3 * min_dist) {
                        good_matches.add(tmp1[i]);
                    }
                }

                Log.i(TAG, "matches size: " + tmp1.length);
                Log.i(TAG, "good matches size: " + good_matches.size());
                if (good_matches.size() < 4) continue;

                LinkedList<Point> objList = new LinkedList<Point>();
                LinkedList<Point> sceneList = new LinkedList<Point>();
                Point[] scene_projArr;

                List<KeyPoint> keypoints_objectList = keypoints_marker.toList();
                List<KeyPoint> keypoints_sceneList = keypoints_scene.toList();

                // Get the corner from the img_object
                Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);

                obj_corners.put(0, 0, new double[]{0, 0});
                obj_corners.put(1, 0, new double[]{img_marker.cols(), 0});
                obj_corners.put(2, 0, new double[]{img_marker.cols(), img_marker.rows()});
                obj_corners.put(3, 0, new double[]{0, img_marker.rows()});

                //while (good_matches.size() > 5) {

                    for (int i = 0; i < good_matches.size(); i++) {
                        objList.addLast(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
                        sceneList.addLast(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
                    }

                    MatOfPoint2f obj = new MatOfPoint2f();
                    obj.fromList(objList);

                    MatOfPoint2f scene = new MatOfPoint2f();
                    scene.fromList(sceneList);

                    Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC, 3);

                    Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);
                    Core.perspectiveTransform(obj_corners, scene_corners, H);

                    Point[] scene_corners_truth = new Point[]{
                            new Point(scene_corners.get(0, 0)[0] * mkscale + mkrpps.get(r).x, scene_corners.get(0, 0)[1] * mkscale + mkrpps.get(r).y),
                            new Point(scene_corners.get(1, 0)[0] * mkscale + mkrpps.get(r).x, scene_corners.get(1, 0)[1] * mkscale + mkrpps.get(r).y),
                            new Point(scene_corners.get(2, 0)[0] * mkscale + mkrpps.get(r).x, scene_corners.get(2, 0)[1] * mkscale + mkrpps.get(r).y),
                            new Point(scene_corners.get(3, 0)[0] * mkscale + mkrpps.get(r).x, scene_corners.get(3, 0)[1] * mkscale + mkrpps.get(r).y)
                    };

                    MatOfPoint scene_corners_truth_mop = new MatOfPoint();
                    scene_corners_truth_mop.fromArray(scene_corners_truth);
                    if (Imgproc.contourArea(scene_corners) < 5000 || !Imgproc.isContourConvex(scene_corners_truth_mop)) {
                        //break;    // if in while for multiple marker detection of same type in each proposal
                        continue;   // if in for with while commented for single marker detection of same type in each proposal
                    }

                    Log.i(TAG, ":) :) :)");
                    mkrnum++;
                    Core.line(BGRMat, scene_corners_truth[0], scene_corners_truth[1], new Scalar(255, 0, 0), 2);
                    Core.line(BGRMat, scene_corners_truth[1], scene_corners_truth[2], new Scalar(255, 0, 0), 2);
                    Core.line(BGRMat, scene_corners_truth[2], scene_corners_truth[3], new Scalar(255, 0, 0), 2);
                    Core.line(BGRMat, scene_corners_truth[3], scene_corners_truth[0], new Scalar(255, 0, 0), 2);

                    mDrawMarksCache.add(scene_corners_truth);
                    if (r != mkrpps.size()-1) {
                        Mat zero = new Mat(mDrawFaces.get(r).height, mDrawFaces.get(r).width, CvType.CV_8UC3);
                        zero.copyTo(BGRMat.submat(mDrawFaces.get(r)));
                        mDrawHiddenFacesCache.add(mDrawFaces.get(r));
                    }

                    // good_matches_nxt.clear();
                    // MatOfPoint2f scene_proj = new MatOfPoint2f();
                    // Core.perspectiveTransform(obj, scene_proj, H);
                    // for (int i=0; i<good_matches.size(); i++) {
                    //     if (Core.norm(scene.row(i), scene_proj.row(i)) > 5.0)   good_matches_nxt.add(good_matches.get(i));
                    // }
                    // good_matches = good_matches_nxt;
                    // objList.clear();
                    // sceneList.clear();
                    // Log.i(TAG, "next good matches size: " + good_matches.size());
                //} // for each appearance , end of while()

                if (mkrnum > 0) break;
            } // for each marker , end of  for(m)
        } // for each proposal , end of for(r)
        mDrawMarks = mDrawMarksCache;
        mDrawHiddenFaces = mDrawHiddenFacesCache;


        fstFrame = false;
        Mat BGRMatScaled = new Mat(size.height / sktScale, size.width / sktScale, CvType.CV_8UC3);
        Imgproc.resize(BGRMat, BGRMatScaled, BGRMatScaled.size(), 0, 0, Imgproc.INTER_AREA);
        byte[] frmdataToSend = new byte[(int) (BGRMatScaled.total() * BGRMatScaled.channels())];
        BGRMatScaled.get(0, 0, frmdataToSend);

        return frmdataToSend;
    }

    // public double[] multiply(double[] p, int s) {
    //     double[] res = new double[p.length];
    //     for (int i=0; i<p.length; i++) res[i] = p[i] * s;
    //     return res;
    // }

    private void initPreview(int width, int height) {
        Log.i(TAG, "initPreview() called");
        if (mCamera != null && mPreviewHolder.getSurface() != null) {
            if (!mCameraConfigured) {
                Camera.Parameters params = mCamera.getParameters();
                params.setPreviewSize(1920, 1080);
                params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                mCamera.setParameters(params);
                size = mCamera.getParameters().getPreviewSize();
                Log.i(TAG, "Preview size: " + size.width + ", " + size.height);
                callbackBuffer = new byte[(size.height + size.height / 2) * size.width];
                mCameraConfigured = true;
            }

            try {
                mCamera.setPreviewDisplay(mPreviewHolder);
                mCamera.addCallbackBuffer(callbackBuffer);
                mCamera.setPreviewCallbackWithBuffer(frameIMGProcCallback);
            } catch (Throwable t) {
                Log.e(TAG, "Exception in initPreview()", t);
            }

        }
    }

    private void startPreview() {
        Log.i(TAG, "startPreview() called");
        if (mCameraConfigured && mCamera != null) {
            mCamera.startPreview();
            mInPreview = true;
        }
    }

    @Override
    public void onResume() {
        Log.i(TAG, " onResume() called.");
        super.onResume();
        mCamera = Camera.open(front1back0);   // 0 for back, 1 for frontal
        mCamera.setDisplayOrientation(90);
        startPreview();
    }

    @Override
    public void onPause() {
        Log.i(TAG, " onPause() called.");
        if (mInPreview)
            mCamera.stopPreview();

        mCamera.setPreviewCallbackWithBuffer(null);

        mCamera.release();
        mCamera = null;
        mInPreview = false;
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        Log.i(TAG, " onDestroy() called.");
        tskQueue.clear();
        shouldContinue = false;
        fTTh = null;
        if (socket != null) try {
            oStream.close();
            socket.close();
            oStream = null;
            socket = null;
        } catch (IOException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }

    /*
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
    */


    /**
     * Inner class for drawing
     */
    class DrawOnTop extends View {

        Paint paintFace;
        Paint paintHiddenFace;
        Paint paintFist;
        Paint paintMark;

        public DrawOnTop(Context context) {
            super(context);

            paintFace = new Paint();
            paintFace.setStyle(Paint.Style.STROKE);
            paintFace.setStrokeWidth(3);
            paintFace.setColor(Color.RED);

            paintHiddenFace = new Paint();
            paintHiddenFace.setStyle(Paint.Style.FILL);
            paintHiddenFace.setColor(Color.MAGENTA);
            paintHiddenFace.setAlpha(100);

            paintFist = new Paint();
            paintFist.setStyle(Paint.Style.STROKE);
            paintFist.setStrokeWidth(3);
            paintFist.setColor(Color.GREEN);

            paintMark = new Paint();
            paintMark.setStyle(Paint.Style.STROKE);
            paintMark.setStrokeWidth(3);
            paintMark.setColor(Color.BLUE);
        }


        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);

            if (mDrawFaces != null) {
                for (int i = 0; i < mDrawFaces.size(); i++) {
                    int r = size.height - mDrawFaces.get(i).y;
                    int t = mDrawFaces.get(i).x;
                    int l = r - mDrawFaces.get(i).height;
                    int b = t + mDrawFaces.get(i).width;
                    if (front1back0 == 1) {
                        int tt = size.width - b;
                        int bb = size.width - t;
                        t = tt;
                        b = bb;
                    }
                    canvas.drawRect(l, t, r, b, paintFace);
                }
            }

            if (mDrawHiddenFaces != null) {
                for (int i = 0; i < mDrawHiddenFaces.size(); i++) {
                    int r = size.height - mDrawHiddenFaces.get(i).y;
                    int t = mDrawHiddenFaces.get(i).x;
                    int l = r - mDrawHiddenFaces.get(i).height;
                    int b = t + mDrawHiddenFaces.get(i).width;
                    if (front1back0 == 1) {
                        int tt = size.width - b;
                        int bb = size.width - t;
                        t = tt;
                        b = bb;
                    }
                    canvas.drawRect(l, t, r, b, paintHiddenFace);
                }
            }

            if (mDrawFists != null) {
                for (int i = 0; i < mDrawFists.length; i++) {
                    int r = size.height - mDrawFists[i].y;
                    int t = mDrawFists[i].x;
                    int l = r - mDrawFists[i].height;
                    int b = t + mDrawFists[i].width;
                    if (front1back0 == 1) {
                        int tt = size.width - b;
                        int bb = size.width - t;
                        t = tt;
                        b = bb;
                    }
                    canvas.drawRect(l, t, r, b, paintFist);
                }
            }

            if (mDrawMarks != null) {
                for (int i = 0; i < mDrawMarks.size(); i++) {
                    Point[] cvpth = mDrawMarks.get(i);
                    float startx = size.height - (float) cvpth[0].y;
                    float starty = (float) cvpth[0].x;
                    if (front1back0 == 1) starty = size.width - starty;
                    float x0 = startx;
                    float y0 = starty;
                    for (int j = 1; j < cvpth.length; j++) {
                        float endx = size.height - (float) cvpth[j].y;
                        float endy = (float) cvpth[j].x;
                        if (front1back0 == 1) endy = size.width - endy;
                        canvas.drawLine(startx, starty, endx, endy, paintMark);
                        startx = endx;
                        starty = endy;
                    }
                    canvas.drawLine(startx, starty, x0, y0, paintMark);
                }
            }

        }
    }
}

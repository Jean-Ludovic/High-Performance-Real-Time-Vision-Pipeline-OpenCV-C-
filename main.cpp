#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <numeric>
#include <string>

using namespace cv;
using namespace std;

struct Args {
    int camera_id = 0;
    string video_path = "";
    int width = 960;
    int height = 540;

    // Thresholds (peuvent être recalibrés)
    int dark_thresh = 60;           // pour "tache sombre" (0..255)
    int min_blob_area = 300;        // aire min blob sombre
    int canny1 = 60, canny2 = 150;  // rayures
    int hough_min_len = 35;
    int hough_max_gap = 8;

    // Stabilisation
    int stable_k = 4;               // NOK si >= k frames NOK consécutives
    int decision_every_n = 1;        // mode simulateur: décision toutes les N frames

    // ROI
    bool use_roi = true;
    Rect roi = Rect(0,0,0,0);

    // Screenshot
    int shot_id = 0;
};

static double nowSec() {
    return (double)getTickCount() / getTickFrequency();
}

static void putHUD(Mat &frame, const string &status, double fps, const Rect &roi) {
    string fpsText = "FPS: " + to_string((int)fps);
    putText(frame, fpsText, Point(15, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 2, LINE_AA);

    Scalar col = (status == "OK") ? Scalar(0,255,0) : Scalar(0,0,255);
    putText(frame, "STATUS: " + status, Point(15, 65), FONT_HERSHEY_SIMPLEX, 0.9, col, 2, LINE_AA);

    if (roi.area() > 0) {
        rectangle(frame, roi, Scalar(255, 200, 0), 2);
        putText(frame, "ROI", Point(roi.x + 5, roi.y - 8), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,200,0), 2);
    }

    putText(frame, "Keys: c=calibrate | s=screenshot | q=quit", Point(15, frame.rows - 15),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(220,220,220), 1, LINE_AA);
}

static Rect defaultCenterROI(const Mat& frame) {
    int w = (int)(frame.cols * 0.55);
    int h = (int)(frame.rows * 0.55);
    int x = (frame.cols - w) / 2;
    int y = (frame.rows - h) / 2;
    return Rect(x, y, w, h);
}

static void parseArgs(int argc, char** argv, Args &a) {
    for (int i=1; i<argc; i++) {
        string k = argv[i];
        auto next = [&](int& i)->string { return (i+1<argc)? string(argv[++i]) : ""; };

        if (k == "--camera_id") a.camera_id = stoi(next(i));
        else if (k == "--video") a.video_path = next(i);
        else if (k == "--dark_thresh") a.dark_thresh = stoi(next(i));
        else if (k == "--min_blob_area") a.min_blob_area = stoi(next(i));
        else if (k == "--canny1") a.canny1 = stoi(next(i));
        else if (k == "--canny2") a.canny2 = stoi(next(i));
        else if (k == "--hough_min_len") a.hough_min_len = stoi(next(i));
        else if (k == "--hough_max_gap") a.hough_max_gap = stoi(next(i));
        else if (k == "--stable_k") a.stable_k = stoi(next(i));
        else if (k == "--every_n") a.decision_every_n = max(1, stoi(next(i)));
        else if (k == "--no_roi") a.use_roi = false;
        else if (k == "--size") {
            string s = next(i); // ex: 960x540
            auto p = s.find('x');
            if (p != string::npos) { a.width = stoi(s.substr(0,p)); a.height = stoi(s.substr(p+1)); }
        }
    }
}

struct Calib {
    bool ready = false;
    double meanL = 0.0;
    double stdL = 0.0;
};

// Calibrage simple : sur 30 frames, calcule moyenne + std de luminance (L de Lab)
static Calib calibrateDarkThreshold(VideoCapture &cap, Args &a) {
    Calib c;
    const int N = 30;
    vector<double> means; means.reserve(N);

    cout << "[Calib] Hold camera steady... capturing " << N << " frames" << endl;

    for (int i=0; i<N; i++) {
        Mat frame; cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(a.width, a.height));
        if (a.use_roi && a.roi.area() > 0) frame = frame(a.roi).clone();

        Mat lab; cvtColor(frame, lab, COLOR_BGR2Lab);
        vector<Mat> ch; split(lab, ch);

        Scalar m = mean(ch[0]);
        means.push_back(m[0]);

        imshow("Calibration", frame);
        if (waitKey(10) == 'q') break;
    }
    destroyWindow("Calibration");

    if ((int)means.size() < 10) return c;

    double avg = accumulate(means.begin(), means.end(), 0.0) / means.size();
    double var = 0.0;
    for (double v : means) var += (v - avg)*(v - avg);
    var /= means.size();
    double st = sqrt(var);

    c.meanL = avg;
    c.stdL = st;
    c.ready = true;

    // règle simple: dark_thresh = mean - 2*std (borné)
    int th = (int)round(avg - 2.0*st);
    a.dark_thresh = std::clamp(th, 10, 200);

    cout << "[Calib] meanL=" << avg << " stdL=" << st << " => dark_thresh=" << a.dark_thresh << endl;
    return c;
}

static bool detectDarkBlob(const Mat &roiBgr, const Args &a, int &maxBlobArea, Rect &blobBox) {
    Mat gray; cvtColor(roiBgr, gray, COLOR_BGR2GRAY);
    Mat blurImg; GaussianBlur(gray, blurImg, Size(5,5), 0);

    Mat mask;
    threshold(blurImg, mask, a.dark_thresh, 255, THRESH_BINARY_INV);

    // nettoie bruit
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_OPEN, k);
    morphologyEx(mask, mask, MORPH_CLOSE, k);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    maxBlobArea = 0;
    blobBox = Rect();

    for (auto &c : contours) {
        int area = (int)contourArea(c);
        if (area > maxBlobArea) {
            maxBlobArea = area;
            blobBox = boundingRect(c);
        }
    }
    return maxBlobArea >= a.min_blob_area;
}

static bool detectScratchLines(const Mat &roiBgr, const Args &a, int &lineCount) {
    Mat gray; cvtColor(roiBgr, gray, COLOR_BGR2GRAY);
    Mat blurImg; GaussianBlur(gray, blurImg, Size(3,3), 0);

    Mat edges;
    Canny(blurImg, edges, a.canny1, a.canny2);

    // HoughP pour segments de rayures
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 60, a.hough_min_len, a.hough_max_gap);

    // Filtre simple: garder lignes assez “fines” déjà captées par Canny + longueur
    lineCount = (int)lines.size();
    return lineCount >= 2; // règle basique, ajuste selon ton cas
}

int main(int argc, char** argv) {
    Args a;
    parseArgs(argc, argv, a);

    VideoCapture cap;
    if (!a.video_path.empty()) cap.open(a.video_path);
    else cap.open(a.camera_id);

    if (!cap.isOpened()) {
        cerr << "Error: cannot open camera/video." << endl;
        return 1;
    }

    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: empty first frame." << endl;
        return 1;
    }

    resize(frame, frame, Size(a.width, a.height));
    if (a.use_roi) a.roi = defaultCenterROI(frame);

    deque<int> nokHistory; // 1=NOK, 0=OK
    const int histMax = max(3, a.stable_k);

    double t0 = nowSec();
    int frames = 0;
    double fps = 0.0;
    int frameIdx = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(a.width, a.height));
        Mat roiView = (a.use_roi && a.roi.area() > 0) ? frame(a.roi).clone() : frame.clone();

        bool doDecision = (frameIdx % a.decision_every_n == 0);
        bool isNok = false;

        int blobArea = 0; Rect blobBox;
        int lineCount = 0;

        if (doDecision) {
            bool dark = detectDarkBlob(roiView, a, blobArea, blobBox);
            bool scratch = detectScratchLines(roiView, a, lineCount);
            isNok = (dark || scratch);

            nokHistory.push_back(isNok ? 1 : 0);
            while ((int)nokHistory.size() > histMax) nokHistory.pop_front();
        }

        // Stabilisation: NOK si au moins stable_k derniers sont NOK
        int sumNok = 0;
        for (int v : nokHistory) sumNok += v;
        bool stableNok = (sumNok >= a.stable_k);

        string status = stableNok ? "NOK" : "OK";

        // Dessins debug
        if (a.use_roi && a.roi.area() > 0) {
            // draw inside ROI: convert local blobBox -> global
            if (blobBox.area() > 0) {
                Rect g = blobBox + Point(a.roi.x, a.roi.y);
                rectangle(frame, g, Scalar(0,0,255), 2);
                putText(frame, "dark_blob area=" + to_string(blobArea),
                        Point(g.x, max(15, g.y - 8)),
                        FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,255), 2);
            }
            // show lineCount
            putText(frame, "scratch_lines=" + to_string(lineCount),
                    Point(a.roi.x + 10, a.roi.y + 25),
                    FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255,255,0), 2);
        }

        // FPS calc
        frames++;
        double t1 = nowSec();
        if (t1 - t0 >= 0.5) {
            fps = frames / (t1 - t0);
            t0 = t1; frames = 0;
        }

        putHUD(frame, status, fps, (a.use_roi ? a.roi : Rect()));
        imshow("OpenCV C++ | Real-time Anomaly + FPS", frame);

        int key = waitKey(1);
        if (key == 'q' || key == 27) break;

        if (key == 's') {
            string name = "screenshot_" + to_string(a.shot_id++) + ".png";
            imwrite(name, frame);
            cout << "[Shot] saved " << name << endl;
        }

        if (key == 'c') {
            // recalibrer seuil de tache sombre
            calibrateDarkThreshold(cap, a);
        }

        frameIdx++;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

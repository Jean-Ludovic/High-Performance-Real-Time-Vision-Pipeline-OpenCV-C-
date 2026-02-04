#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <numeric>
#include <string>

using namespace cv;
using namespace std;

struct Args {
    int camera_id = 0;
    string vedeo_path = "";
    int width =960;
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
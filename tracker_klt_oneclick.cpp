#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;

struct Args {
    string source;
    string initStr;
    string outCsv = "";
    bool   vis = true;
};

static void usage() {
    printf(
      "Usage:\n"
      "  tracker_klt_oneclick --source <video|camera:IDX@WxH@FPS> [--init \"x,y,w,h\"] [--out tracks.csv] [--vis 1]\n"
      "\nHotkeys / Mouse:\n"
      "  Left-click: ONE-CLICK select (GrabCut → auto box → start tracking)\n"
      "  S         : Select ROI (drag a box) and start tracking\n"
      "  Q / Esc   : Quit\n"
    );
}

static Rect2d parseInit(const string& s) {
    int x=0,y=0,w=0,h=0; char c;
    stringstream ss(s);
    ss >> x >> c >> y >> c >> w >> c >> h;
    return Rect2d(x,y,w,h);
}

static bool openSource(VideoCapture& cap, const string& source) {
    if (source.rfind("camera:", 0) == 0) {
        int idx=0, w=1280, h=720, fps=30;
        if (sscanf(source.c_str(), "camera:%d@%dx%d@%d", &idx,&w,&h,&fps) < 1) {
            idx=0; w=1280; h=720; fps=30;
        }
#if defined(CAP_V4L2)
        cap.open(idx, CAP_V4L2);
#else
        cap.open(idx);
#endif
        if (!cap.isOpened()) return false;
        cap.set(CAP_PROP_FRAME_WIDTH,  w);
        cap.set(CAP_PROP_FRAME_HEIGHT, h);
        cap.set(CAP_PROP_FPS,          fps);
        return true;
    } else {
        return cap.open(source);
    }
}

static vector<Point2f> rectToQuad(const Rect2d& r){
    return {Point2f((float)r.x,(float)r.y),
            Point2f((float)(r.x+r.width),(float)r.y),
            Point2f((float)(r.x+r.width),(float)(r.y+r.height)),
            Point2f((float)r.x,(float)(r.y+r.height))};
}

static Rect2d quadToRect(const vector<Point2f>& q){
    float minx=1e9f,miny=1e9f,maxx=-1e9f,maxy=-1e9f;
    for(auto&p:q){ minx=min(minx,p.x); miny=min(miny,p.y); maxx=max(maxx,p.x); maxy=max(maxy,p.y); }
    return Rect2d(minx,miny,max(1.f,maxx-minx),max(1.f,maxy-miny));
}

static Rect2d grabcutFromClick(const Mat& bgr, Point click, float relRect=0.25f, int iters=3){
    int W=bgr.cols, H=bgr.rows;
    int side = (int)std::round(relRect * std::min(W,H));
    side = std::max(40, std::min(side, std::min(W,H)));
    int x = std::clamp(click.x - side/2, 0, W-1);
    int y = std::clamp(click.y - side/2, 0, H-1);
    Rect rect(x, y, std::min(side, W-x), std::min(side, H-y));

    Mat mask(H, W, CV_8U, Scalar(GC_BGD));
    mask(rect) = Scalar(GC_PR_FGD);

    Mat bgdModel, fgdModel;
    grabCut(bgr, mask, rect, bgdModel, fgdModel, iters, GC_INIT_WITH_RECT);

    Mat bin = (mask == GC_FGD) | (mask == GC_PR_FGD);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    morphologyEx(bin, bin, MORPH_OPEN, kernel);
    morphologyEx(bin, bin, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if(contours.empty()) return Rect2d(rect);
    size_t best=0; double bestA=0;
    for(size_t i=0;i<contours.size();++i){
        double a = contourArea(contours[i]);
        if(a>bestA){ bestA=a; best=i; }
    }
    Rect bb = boundingRect(contours[best]);
    int pad = (int)std::round(0.04 * std::min(bb.width, bb.height));
    Rect bbp = Rect(max(0, bb.x - pad), max(0, bb.y - pad),
                    min(W - max(0, bb.x - pad), bb.width + 2*pad),
                    min(H - max(0, bb.y - pad), bb.height + 2*pad));
    return Rect2d(bbp);
}

struct ClickState { bool hasClick=false; Point pt; };

int main(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;i++){
        string s=argv[i];
        if (s=="--source" && i+1<argc) a.source=argv[++i];
        else if (s=="--init" && i+1<argc) a.initStr=argv[++i];
        else if (s=="--out" && i+1<argc) a.outCsv=argv[++i];
        else if (s=="--vis" && i+1<argc) a.vis = (string(argv[++i])!="0");
        else if (s=="--help"||s=="-h"){ usage(); return 0; }
    }
    if (a.source.empty()){ usage(); return 1; }

    VideoCapture cap;
    if(!openSource(cap,a.source)){ fprintf(stderr,"Failed to open source\n"); return 2; }

    Mat frame, gray, prevGray;
    if(!cap.read(frame) || frame.empty()){ fprintf(stderr,"No frames\n"); return 3; }

    const int   maxCorners=600; const double qualityLevel=0.01, minDistance=6.0;
    const int   blockSize=3; const bool useHarris=false; const double k=0.04;
    const Size  lkWin(21,21); const int lkMaxLevel=3;
    const TermCriteria lkCrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);

    const double ransacReproj=3.0; const double minInlierFrac=0.4;
    const int    minPointsTrack=30; const int seedEveryN=8;

    bool hasInit=false; Rect2d box; vector<Point2f> quad = rectToQuad(Rect2d(100,100,200,150));
    vector<Point2f> ptsPrev, ptsCurr; vector<uchar> status; vector<float> err;

    FILE* fout=nullptr;
    if(!a.outCsv.empty()){
        fout=fopen(a.outCsv.c_str(),"w");
        if(fout) fprintf(fout,"# frame,x,y,w,h,ms\n");
    }

    const string win="KLT-SRT One-Click Tracker";
    if (a.vis){ namedWindow(win, WINDOW_NORMAL); resizeWindow(win, 1280, 720); }

    ClickState click;
    if (a.vis) {
        setMouseCallback(win, [](int event, int x, int y, int, void* userdata){
            ClickState* cs = reinterpret_cast<ClickState*>(userdata);
            if(event == EVENT_LBUTTONDOWN){ cs->hasClick = true; cs->pt = Point(x,y); }
        }, &click);
    }

    auto initFromBox = [&](const Rect2d& roi){
        Mat mask(frame.rows, frame.cols, CV_8U, Scalar(0));
        rectangle(mask, roi, Scalar(255), FILLED);
        cvtColor(frame, prevGray, COLOR_BGR2GRAY);
        goodFeaturesToTrack(prevGray, ptsPrev, maxCorners, qualityLevel, minDistance, mask,
                            blockSize, useHarris, k);
        if(ptsPrev.size() < 8) return false;
        box = roi; quad = rectToRect(roi); // helper not defined, keep as rectToQuad
        quad = rectToQuad(roi);
        hasInit = true; return true;
    };

    if(!a.initStr.empty()){
        if(!initFromBox(parseInit(a.initStr))){ fprintf(stderr,"Init failed: not enough features\n"); return 4; }
    }

    int frameId=0, sinceSeed=0; double fpsEMA=0.0; const double alpha=0.1;

    while(true){
        if(frameId>0){ if(!cap.read(frame) || frame.empty()) break; }

        if(a.vis && click.hasClick){
            click.hasClick=false;
            Rect2d gcBox = grabcutFromClick(frame, click.pt, 0.25f, 3);
            if(gcBox.area()>100){
                if(!initFromBox(gcBox)){
                    Rect2d infl = gcBox;
                    infl.x = max(0.0, infl.x - infl.width*0.1);
                    infl.y = max(0.0, infl.y - infl.height*0.1);
                    infl.width  = min((double)frame.cols - infl.x, infl.width*1.2);
                    infl.height = min((double)frame.rows - infl.y, infl.height*1.2);
                    initFromBox(infl);
                }
            }
        }

        int key = a.vis ? (waitKey(1)&0xFF) : -1;
        if(key==27 || key=='q' || key=='Q') break;
        if(a.vis && (key=='s'||key=='S')){
            Rect2d roi = selectROI("Select ROI", frame, false, false);
            destroyWindow("Select ROI");
            if(roi.area()>0) initFromBox(roi);
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        bool okBox=false; double ms=0.0;
        auto t0 = chrono::high_resolution_clock::now();

        if(hasInit){
            calcOpticalFlowPyrLK(prevGray, gray, ptsPrev, ptsCurr, status, err, lkWin, lkMaxLevel, lkCrit);
            vector<Point2f> p0, p1; p0.reserve(ptsPrev.size()); p1.reserve(ptsCurr.size());
            for(size_t i=0;i<ptsCurr.size();++i) if(status[i]){ p0.push_back(ptsPrev[i]); p1.push_back(ptsCurr[i]); }

            if(p0.size() >= 6){
                Mat inliers;
                Mat A = estimateAffinePartial2D(p0, p1, inliers, RANSAC, ransacReproj, 2000, 0.995);
                if(!A.empty()){
                    int inl=0; for(int i=0;i<inliers.rows;i++) if(inliers.at<uchar>(i)) inl++;
                    if(inl >= (int)(minInlierFrac * (int)p0.size())){
                        vector<Point2f> qNew(4);
                        for(int i=0;i<4;i++){
                            Point2f p = quad[i];
                            float nx = (float)(A.at<double>(0,0)*p.x + A.at<double>(0,1)*p.y + A.at<double>(0,2));
                            float ny = (float)(A.at<double>(1,0)*p.x + A.at<double>(1,1)*p.y + A.at<double>(1,2));
                            qNew[i] = Point2f(nx,ny);
                        }
                        quad.swap(qNew);
                        box = quadToRect(quad);
                        okBox = true;
                    }
                }
            }

            sinceSeed++;
            bool needReseed = (!okBox) || ((int)p1.size() < minPointsTrack) || (sinceSeed >= seedEveryN);
            if(needReseed){
                Rect2d r = okBox ? box : quadToRect(quad);
                r &= Rect2d(0,0, frame.cols, frame.rows);
                if(r.width>5 && r.height>5){
                    Mat mask(gray.rows, gray.cols, CV_8U, Scalar(0));
                    rectangle(mask, r, Scalar(255), FILLED);
                    goodFeaturesToTrack(gray, ptsCurr, maxCorners, qualityLevel, minDistance, mask,
                                        blockSize, useHarris, k);
                }
                sinceSeed = 0;
            } else {
                ptsCurr.swap(p1);
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        ms = chrono::duration<double, std::milli>(t1 - t0).count();
        double instFPS = ms>0.0 ? 1000.0/ms : 0.0;
        fpsEMA = (1.0-alpha)*fpsEMA + alpha*instFPS;

        if(a.vis){
            if(hasInit){
                polylines(frame, quad, true, Scalar(0,255,0), 2);
                for(size_t i=0;i<ptsCurr.size() && i<150;i++) circle(frame, ptsCurr[i], 2, Scalar(0,255,255), -1, LINE_AA);
                putText(frame, format("FPS: %.1f", fpsEMA), {10,30}, FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
                putText(frame, "Left-click=One-click | S=ROI", {10,frame.rows-12}, FONT_HERSHEY_SIMPLEX, 0.6, {255,255,0}, 2);
            } else {
                putText(frame, "Left-click to one-click select, or press S to drag ROI", {10,30},
                        FONT_HERSHEY_SIMPLEX, 0.7, {0,255,255}, 2);
            }
            imshow("KLT-SRT One-Click Tracker", frame);
        }

        if(fout){
            if(hasInit) fprintf(fout,"%d,%.1f,%.1f,%.1f,%.1f,%.2f\n", frameId, box.x, box.y, box.width, box.height, ms);
            else        fprintf(fout,"%d,-1,-1,-1,-1,%.2f\n", frameId, ms);
        }

        prevGray = gray.clone(); ptsPrev = ptsCurr; frameId++;
    }

    return 0;
}

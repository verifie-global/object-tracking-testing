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
    string outCsv = "";
    bool   vis = true;
    int    redetectEvery = 8;
    int    nfeatures = 1200;
    double objW = 0.10; // meters
    double objH = 0.06; // meters
};

static void usage(){
    printf(
      "Usage:\n"
      "  tracker_orb_klt_pnp --source <video|camera:IDX@WxH@FPS> [--size_m WxH] [--redetect_every N] [--nfeatures N]\n"
      "                      [--out tracks.csv] [--vis 1]\n"
      "\nControls:\n"
      "  S : Select planar face (drag ROI) to capture template\n"
      "  Q/Esc : Quit\n"
    );
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

static vector<Point2f> orderQuad(const vector<Point2f>& pts){
    vector<Point2f> p = pts;
    // tl, tr, br, bl by sums/diffs
    auto sum = [](Point2f a){ return a.x + a.y; };
    auto dif = [](Point2f a){ return a.x - a.y; };
    Point2f tl=p[0], tr=p[0], br=p[0], bl=p[0];
    double sMin=1e9, sMax=-1e9, dMin=1e9, dMax=-1e9;
    for(auto& q: p){
        double s=sum(q), d=dif(q);
        if(s<sMin){ sMin=s; tl=q; }
        if(s>sMax){ sMax=s; br=q; }
        if(d<dMin){ dMin=d; tr=q; }
        if(d>dMax){ dMax=d; bl=q; }
    }
    return {tl,tr,br,bl};
}

static Matx33f approxK(int w, int h, float fov_deg=60.f){
    float fx = w / (2.f * std::tan((fov_deg*CV_PI/180.f)/2.f));
    float fy = h / (2.f * std::tan((fov_deg*CV_PI/180.f)/2.f));
    return Matx33f(fx,0,w*0.5f, 0,fy,h*0.5f, 0,0,1);
}

int main(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;i++){
        string s=argv[i];
        if(s=="--source" && i+1<argc) a.source=argv[++i];
        else if(s=="--out" && i+1<argc) a.outCsv=argv[++i];
        else if(s=="--vis" && i+1<argc) a.vis = (string(argv[++i])!="0");
        else if(s=="--redetect_every" && i+1<argc) a.redetectEvery = atoi(argv[++i]);
        else if(s=="--nfeatures" && i+1<argc) a.nfeatures = atoi(argv[++i]);
        else if(s=="--size_m" && i+1<argc){
            double W=0.1,H=0.06; sscanf(argv[++i], "%lfx%lf", &W, &H); a.objW=W; a.objH=H;
        }
        else if(s=="--help"||s=="-h"){ usage(); return 0; }
    }
    if(a.source.empty()){ usage(); return 1; }

    VideoCapture cap;
    if(!openSource(cap,a.source)){ fprintf(stderr,"Failed to open source\n"); return 2; }

    Mat frame, gray, prevGray;
    if(!cap.read(frame) || frame.empty()){ fprintf(stderr,"No frames\n"); return 3; }

    // Camera intrinsics (approx; replace with calibrated values for best pose)
    Matx33f K = approxK(frame.cols, frame.rows, 60.f);
    Mat dist  = Mat::zeros(1,5,CV_32F);

    // ORB + BFMatcher
    Ptr<ORB> orb = ORB::create(a.nfeatures, 1.2f, 8, 15, 0, 2, ORB::HARRIS_SCORE, 31, 12);
    BFMatcher matcher(NORM_HAMMING, false);

    // Template (from ROI on 'S')
    Mat tmpl, desT;
    vector<KeyPoint> kpT;
    Size tmplSize;
    bool haveTemplate=false;

    // Tracking state
    vector<Point2f> quad(4);         // tl,tr,br,bl in image
    vector<Point2f> ptsPrev, ptsCurr;
    vector<uchar> status; vector<float> err;
    int frameId=0;

    // Pose
    Mat rvec, tvec;

    // Logging
    FILE* fout=nullptr;
    if(!a.outCsv.empty()){
        fout=fopen(a.outCsv.c_str(),"w");
        if(fout) fprintf(fout,"# frame,x,y,w,h,ms\n");
    }

    const string win="ORB re-detect + KLT + PnP";
    if(a.vis){ namedWindow(win, WINDOW_NORMAL); resizeWindow(win, 1280, 720); }

    auto quadToRect = [](const vector<Point2f>& q)->Rect2d{
        float minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
        for(auto&p:q){ minx=min(minx,p.x); miny=min(miny,p.y); maxx=max(maxx,p.x); maxy=max(maxy,p.y); }
        return Rect2d(minx,miny,max(1.f,maxx-minx),max(1.f,maxy-miny));
    };

    auto detectTemplate = [&](const Mat& imgGray){
        vector<KeyPoint> kpF; Mat desF;
        orb->detectAndCompute(imgGray, noArray(), kpF, desF);
        if(desT.empty() || desF.empty()) return false;

        vector<vector<DMatch>> knn; matcher.knnMatch(desT, desF, knn, 2);
        vector<DMatch> good; good.reserve(knn.size());
        for(auto& v: knn) if(v.size()==2 && v[0].distance < 0.75f*v[1].distance) good.push_back(v[0]);
        if(good.size() < 6) return false;

        vector<Point2f> src, dst;
        src.reserve(good.size()); dst.reserve(good.size());
        for(auto&m:good){ src.push_back(kpT[m.queryIdx].pt); dst.push_back(kpF[m.trainIdx].pt); }
        Mat H = findHomography(src, dst, RANSAC, 3.0);
        if(H.empty()) return false;

        vector<Point2f> tquad = { {0,0}, {(float)tmplSize.width,0},
                                  {(float)tmplSize.width,(float)tmplSize.height}, {0,(float)tmplSize.height} };
        perspectiveTransform(tquad, tquad, H);
        quad = orderQuad(tquad);
        return true;
    };

    auto seedPointsInQuad = [&](const Mat& grayImg){
        Rect2d r = quadToRect(quad) & Rect2d(0,0, grayImg.cols, grayImg.rows);
        if(r.width<5 || r.height<5) return;
        Mat mask(grayImg.rows, grayImg.cols, CV_8U, Scalar(0));
        rectangle(mask, r, Scalar(255), FILLED);
        goodFeaturesToTrack(grayImg, ptsCurr, 600, 0.01, 6.0, mask, 3, false, 0.04);
    };

    auto drawAxes = [&](Mat& img){
        drawFrameAxes(img, K, dist, rvec, tvec, (float)min(a.objW, a.objH)*0.6f);
    };

    double fpsEMA=0.0; const double alpha=0.1; int sinceRedetect=0;

    while(true){
        if(frameId>0){ if(!cap.read(frame) || frame.empty()) break; }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        int key = a.vis ? (waitKey(1)&0xFF) : -1;
        if(key==27 || key=='q' || key=='Q') break;

        if(a.vis && key=='s'){
            Rect2d roi = selectROI("Select planar face", frame, false, false);
            destroyWindow("Select planar face");
            if(roi.area()>0){
                tmpl = gray(roi).clone();
                vector<KeyPoint> kpt; Mat dst;
                orb->detectAndCompute(tmpl, noArray(), kpt, dst);
                kpT.clear(); kpT.reserve(kpt.size());
                for(auto& k: kpt){
                    KeyPoint kk = k; kk.pt.x += (float)roi.x; kk.pt.y += (float)roi.y;
                    kpT.push_back(kk);
                }
                desT = dst.clone();
                tmplSize = roi.size();
                haveTemplate = !desT.empty() && kpT.size()>=12;
                if(haveTemplate){
                    if(detectTemplate(gray)){
                        seedPointsInQuad(gray);
                        prevGray = gray.clone();
                    }
                }
            }
        }

        bool ok=false; double ms=0.0;
        auto t0 = chrono::high_resolution_clock::now();

        if(haveTemplate){
            bool needRedetect = (frameId % a.redetectEvery == 0) || ptsPrev.empty() || (ptsPrev.size()<8);
            if(needRedetect){
                if(detectTemplate(gray)){
                    seedPointsInQuad(gray);
                }
                prevGray = gray.clone();
            } else {
                calcOpticalFlowPyrLK(prevGray, gray, ptsPrev, ptsCurr, noArray(), noArray(),
                                     Size(21,21), 3,
                                     TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03));
                if(ptsCurr.size()>=4){
                    vector<Point2f> p0 = ptsPrev, p1 = ptsCurr;
                    Mat inliers;
                    Mat A = estimateAffinePartial2D(p0, p1, inliers, RANSAC, 3.0, 2000, 0.995);
                    if(!A.empty()){
                        vector<Point2f> qNew(4);
                        for(int i=0;i<4;i++){
                            Point2f p = quad[i];
                            float nx = (float)(A.at<double>(0,0)*p.x + A.at<double>(0,1)*p.y + A.at<double>(0,2));
                            float ny = (float)(A.at<double>(1,0)*p.x + A.at<double>(1,1)*p.y + A.at<double>(1,2));
                            qNew[i] = Point2f(nx,ny);
                        }
                        quad.swap(qNew);
                        ok=true;
                    }
                }
                prevGray = gray.clone();
                ptsPrev = ptsCurr;
            }

            if(!quad.empty()){
                // Pose from quad corners (planar IPPE)
                vector<Point3f> obj = { {0,0,0}, {(float)a.objW,0,0}, {(float)a.objW,(float)a.objH,0}, {0,(float)a.objH,0} };
                vector<Point2f> img = quad;
                if(img.size()==4){
                    solvePnP(obj, img, K, dist, rvec, tvec, false, SOLVEPNP_IPPE_SQUARE);
                }
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        ms = chrono::duration<double, std::milli>(t1 - t0).count();
        double instFPS = ms>0.0 ? 1000.0/ms : 0.0;
        fpsEMA = (1.0-alpha)*fpsEMA + alpha*instFPS;

        if(a.vis){
            if(haveTemplate && !quad.empty()){
                polylines(frame, quad, true, Scalar(0,255,0), 2);
                putText(frame, format("FPS: %.1f", fpsEMA), {10,30}, FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
                if(!rvec.empty() && !tvec.empty()){
                    drawAxes(frame);
                    putText(frame, "PnP OK", {10,60}, FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
                } else {
                    putText(frame, "Init with S (planar ROI)", {10,60}, FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2);
                }
            } else {
                putText(frame, "Press S: select planar face (known size)", {10,30}, FONT_HERSHEY_SIMPLEX, 0.7, {0,255,255}, 2);
            }
            imshow("ORB re-detect + KLT + PnP", frame);
        }

        if(fout){
            Rect2d r = (!quad.empty()) ? Rect2d( min(min(quad[0].x,quad[1].x), min(quad[2].x,quad[3].x)),
                                                 min(min(quad[0].y,quad[1].y), min(quad[2].y,quad[3].y)),
                                                 0,0) : Rect2d(0,0,0,0);
            if(!quad.empty()){
                double minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
                for(auto&p:quad){ minx=min(minx,p.x); miny=min(miny,p.y); maxx=max(maxx,p.x); maxy=max(maxy,p.y); }
                r = Rect2d(minx,miny,max(1.0,maxx-minx),max(1.0,maxy-miny));
                fprintf(fout,"%d,%.1f,%.1f,%.1f,%.1f,%.2f\n", frameId, r.x, r.y, r.width, r.height, ms);
            } else {
                fprintf(fout,"%d,-1,-1,-1,-1,%.2f\n", frameId, ms);
            }
        }

        frameId++;
    }

    if(fout) fclose(fout);
    return 0;
}

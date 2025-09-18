# Tracking Suite (C++): Implementations and Guide

This package contains two **from-scratch** C++ trackers and a methodology guide:

## Trackers
1) **One-click: GrabCut → KLT + Similarity (RANSAC)**
   - File: `tracker_klt_oneclick.cpp`
   - Init: one left-click; GrabCut extracts a mask → tight box → KLT tracks Shi–Tomasi features
   - Update: `estimateAffinePartial2D` (RANSAC) on tracked points to update a 2D quad / box
   - Fastest (no 3D pose)

2) **ORB re-detect + KLT + PnP (planar 3D face)**
   - File: `src/tracker_orb_klt_pnp.cpp`
   - Init: press `S` to box a planar face (known real size in meters)
   - Redetect: ORB match + Homography (RANSAC) every N frames (or on confidence drop)
   - Pose: `solvePnP` (`IPPE_SQUARE`) each frame for 6-DoF of the plane
   - KLT tracks points between redetects for speed

## Build (Windows, Linux, macOS)
- Requires OpenCV 4.x. Contrib is recommended for ORB on older builds, but many distros include it in main `libopencv-dev` now.
- Each tracker has its own `CMakeLists.txt`. Example (Linux):
```bash
cd oneclick && mkdir build && cd build
cmake ..
make -j$(nproc)
./tracker_klt_oneclick --source camera:0@1280x720@30 --vis 1
```
```bash
cd ../../orb_klt_pnp && mkdir build && cd build
cmake ..
make -j$(nproc)
./tracker_orb_klt_pnp --source camera:0@1280x720@30 --size_m 0.10x0.06 --vis 1
```

## CLI
Both trackers support:
- `--source <video|camera:IDX@WxH@FPS>`
- `--out tracks.csv` (optional CSV log)
- `--vis 0/1` (visualization on/off)

Additional (ORB+PnP):
- `--size_m WxH` (e.g., `0.10x0.06`) real planar face size (meters)
- `--redetect_every N` (default 8)
- `--nfeatures N` (default 1200)

## Notes
- Use **performance governor** on Pi for stable FPS.
- Lock camera exposure/white balance.
- For best pose accuracy, replace approximate intrinsics with calibrated K, dist.

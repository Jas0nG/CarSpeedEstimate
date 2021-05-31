#!/usr/bin/env python


'''
GitHub https://github.com/Jas0nG/CarSpeedEstimate/blob/master/lk_track.py
Car Speed Estimate Based On Lucas-Kanade sparse optical flow Demo
====================

Uses SIFT for track initialization
and back-tracking for match verification
between frames.

Usage
-----
lk_track.py


Keys
----
ESC - exit
'''
# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
from cv2 import cv2
import os

control_point = [(46, 236), (5, 231), (50, 227), (11, 223), (61, 220), (21, 217), (69, 214), (31, 211), (77, 207), (42, 205), (84, 201),
                 (50, 200), (92, 196), (61, 195), (98, 193), (65, 190), (105, 187), (76, 186), (112, 184), (82, 183), (118, 180), (90, 178)]
images_path = R"C:\Users\12093\Pictures\1"
files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=500,
                      qualityLevel=0.5,
                      minDistance=7,
                      blockSize=7)


def euclidean(t1, t2):  # 欧氏距离计算
    return np.sqrt(np.square(t1[0]-t2[0])+np.square(t1[1]-t2[1]))


class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks_sift = []
        self.tracks = []
        self.frame_idx = 0

    def run(self):
        i = 0
        print(i)
        while i < (len(files)):
            frame = cv2.imread(files[i])
            j = 0
            for j in range(0, len(control_point)-1, 2):
                cv2.line(
                    frame, control_point[j], control_point[j+1], (60, j*70+50, 0), 1, cv2.LINE_AA)
            frame = frame[105:288, 0:233]
            i = i+1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            if len(self.tracks_sift) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1]
                                for tr in self.tracks_sift]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(
                    img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
                    img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks_sift = []
                for tr, (x, y), (x0, y0), good_flag in zip(self.tracks_sift, p1.reshape(-1, 2), p0r.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    if abs(tr[0][0] - x) < 2 or abs(tr[0][1] - y) < 2:
                        continue
                    j = 0
                    last_speed = 103
                    for j in range(0, len(control_point)-2, 2):
                        if control_point[j][1] > y+105 and control_point[j+3][1] < y+105 and control_point[j][0] > x and control_point[j+3][0] < x:
                            #cv2.circle(vis, (x0,y0), 2, (0, 255, 0), -1)
                            cv2.line(vis, (x0, y0), (x, y), (0, 200, 255), 3)
                            speed = 90*(euclidean((x0, y0), (x, y))/((euclidean(
                                control_point[j], control_point[j+2])+euclidean(control_point[j+1], control_point[j+3]))/2))
                            if abs(speed - last_speed) < 5:
                                print("Speed:", speed, "Km/H")
                            last_speed = speed
                            #cv2.putText(vis,"{:.2f}".format(tspeed)+"km/h",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0), 3, cv2.LINE_AA)
                            break
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks_sift.append(tr)
                    cv2.circle(vis, (x, y), 1, (200, 0, 0), -1)
                self.tracks_sift = new_tracks_sift
                cv2.polylines(vis, [np.int32(
                    tr) for tr in self.tracks_sift], False, (255, 90, 100), 1, cv2.LINE_AA)

            if self.frame_idx % self.detect_interval == 0:
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(frame_gray, None)
                pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp])
                if pts is not None:
                    for x, y in np.float32(pts).reshape(-1, 2):
                        self.tracks_sift.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('Speed Estimate', vis)
            cv2.waitKey(40)


def main():
    App().run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()

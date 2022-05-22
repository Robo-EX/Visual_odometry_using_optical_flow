#!/usr/bin/env python3

import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D


class Camera():
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def camera_matrix(self):
        matrix = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fx, self.cy],
                           [0.0, 0.0, 1.0]])
        return matrix


feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=7, blockSize=7)

lk_params = dict(winSize=(21, 21),
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 30, 0.03))


def Rotation(pts, angle, degrees=False):
    if degrees == True:
        theta = np.radians(angle)
    else:
        theta = angle

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    rot_pts = []
    for v in pts:
        v = np.array(v).transpose()
        v = R.dot(v)
        v = v.transpose()
        rot_pts.append(v)

    return rot_pts


class Visual_Odom():
    def __init__(self, camera, path=None, see_features=False):
        self.camera = camera
        self.path = path
        self.see_features = see_features
        self.detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True)
        self.gt = self.get_ground_truth()

    def get_ground_truth(self):
        po_file = []
        if self.path is not None:
            with open(os.path.join(self.path, 'poses/00.txt')) as f:
                self.ground_truth = f.readlines()
                for line in self.ground_truth:
                    position = np.array(line.split()).reshape(
                        (3, 4)).astype(np.float32)
                    po_file.append(position)

        return po_file

    def Detect_from_video(self, position_axes, cap, see_tracking=False):
        prev_image = None
        current_pos = np.zeros((3, 1))
        current_rot = np.eye(3)
        img_idx = 0
        # Import only if not previously imported
        if cap.isOpened() == False:
            print("Error in opening video stream or file")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # Press esc to exit
                if cv2.waitKey(20) & 0xFF == 27:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoint = self.detector.detect(image, None)

                if prev_image is None:
                    prev_image = image
                    prev_keypoint = keypoint
                    continue

                points = cv2.goodFeaturesToTrack(image, mask=None, **feature_params) if see_tracking else np.array([
                    x.pt for x in prev_keypoint], dtype=np.float32)

                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_image, image, points, None, **lk_params)

                if see_tracking:
                    good_new = p1[st == 1]
                    good_old = points[st == 1]
                    img_copy = frame
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        c, d = np.int32(old.ravel())

                        v1 = tuple(np.int32((new - old) * 2.5 + old))
                        d_v = [new - old][0] * 0.5

                        arrow_t1 = Rotation([d_v], 0.3)
                        arrow_t2 = Rotation([d_v], -0.3)
                        tip1 = np.int32(
                            tuple(np.float32(np.array([c, d]) + arrow_t1)[0]))
                        tip2 = np.int32(
                            tuple(np.float32(np.array([c, d]) + arrow_t2)[0]))

                        cv2.line(img_copy, v1, (c, d), (0, 255, 0), 2)
                        cv2.line(img_copy, (c, d), tip1, (28, 24, 178), 2)
                        cv2.line(img_copy, (c, d), tip2, (28, 24, 178), 2)
                        cv2.circle(img_copy, v1, 1, (0, 255, 0), -1)

                    cv2.imshow('Show track', img_copy)

                E, mask = cv2.findEssentialMat(
                    p1, points, self.camera.camera_matrix(), cv2.RANSAC, 0.999, 1.0, None)
                points, R, t, mask = cv2.recoverPose(
                    E, p1, points, self.camera.camera_matrix())

                scale = 1.0

                current_pos += current_rot.dot(t) * scale
                current_rot = R.dot(current_rot)

                # position_axes.scatter(
                #     current_pos[0][0], current_pos[2][0], c='b')
                # plt.pause(0.0001)
                print(current_rot)
                if self.see_features:
                    feature = cv2.drawKeypoints(image, keypoint, None)
                    cv2.imshow('Image', feature)
                # else:
                #     cv2.imshow('Image', image)

                prev_image = image
                prev_keypoint = keypoint

            else:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = Camera(fx=718.8560, fy=718.8560, cx=607.1928, cy=185.2157)
    vid_odom = Visual_Odom(cam, path='dataset/', see_features=False)
    position_figure = plt.figure()
    position_axes = position_figure.add_subplot(1, 1, 1)
    position_axes.set_aspect('equal', adjustable='box')
    legend_elements = [Line2D([0], [0], marker='o', color='b', label='Position',
                              markerfacecolor='b', markersize=7)]
    # if True:
    #     legend_elements.append(Line2D([0], [0], marker='o', color='r', label='Ground truth',
    #                                   markerfacecolor='r', markersize=7))
    position_axes.legend(handles=legend_elements,
                         facecolor='white', framealpha=1)

    cap = cv2.VideoCapture(0)
    vid_odom.Detect_from_video(position_axes, cap, see_tracking=True)

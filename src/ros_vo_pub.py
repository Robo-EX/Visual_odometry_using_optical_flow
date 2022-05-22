#!/usr/bin/env python3

import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D
import rospy
from tf.transformations import euler_matrix, quaternion_from_euler
import math
from geometry_msgs.msg import Pose
# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).


def rotationMatrixToEulerAngles(R):

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


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
        rospy.init_node("VO_node")
        rospy.loginfo("Starting RosNode.")
        self.camera = camera
        self.path = path
        self.see_features = see_features
        self.detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True)
        # self.gt = self.get_ground_truth()
        self.message_pub = rospy.Publisher(
            "optical_Flow/pose", Pose, queue_size=10)

    # def get_ground_truth(self):
    #     po_file = []
    #     if self.path is not None:
    #         with open(os.path.join(self.path, 'poses/00.txt')) as f:
    #             self.ground_truth = f.readlines()
    #             for line in self.ground_truth:
    #                 position = np.array(line.split()).reshape(
    #                     (3, 4)).astype(np.float32)
    #                 po_file.append(position)

    #     return po_file

    def Detect_from_video(self, position_axes, cap, see_tracking=False):
        msg = Pose()
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

                points = cv2.goodFeaturesToTrack(prev_image, mask=None, **feature_params) if see_tracking else np.array([
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

                x, y, z = rotationMatrixToEulerAngles(current_rot)
                print(current_rot, euler_matrix(x, y, z))
                q = quaternion_from_euler(x, y, z, 'sxyz')

                msg.position.x = current_pos[0][0]
                msg.position.y = current_pos[2][0]
                msg.position.z = current_pos[1][0]  # No use of Z
                msg.orientation.x = q[0]
                msg.orientation.y = q[1]
                msg.orientation.z = q[2]
                msg.orientation.w = q[3]
                # position_axes.scatter(
                #     current_pos[0][0], current_pos[2][0], c='b')
                # plt.pause(0.0001)

                if self.see_features:
                    feature = cv2.drawKeypoints(image, keypoint, None)
                    cv2.imshow('Image', feature)
                else:
                    cv2.imshow('Image', image)

                prev_image = image
                prev_keypoint = keypoint
                self.message_pub.publish(msg)

            else:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = Camera(fx=487.402, fy=485.414, cx=363.54, cy=250.89)
    vid_odom = Visual_Odom(cam, see_features=False)
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

    cap = cv2.VideoCapture('dataset/videoplayback.mp4')
    vid_odom.Detect_from_video(position_axes, cap, see_tracking=True)

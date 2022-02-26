#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')
import cv2
import time
import Camera
import threading
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

AK = ArmIK()

range_rgb = {
    'red':   (0, 0, 255),
    'blue':  (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

def get_area_max_contour(contours):
        contour_area_temp = 0
        contour_area = 0
        area_contour = None

        for c in contours :
            contour_area_temp = math.fabs(cv2.contourArea(c))
            if contour_area_temp > contour_area:
                contour_area = contour_area_temp
                if contour_area_temp > 300:
                    area_contour = c

        return area_contour, contour_area

class CubeTracker:
    def __init__(self, color: str = 'red'):
        self.target_color = color
        self.size = (640, 480)
        self.last_x = 0
        self.last_y = 0

    def track(self, frame: np.ndarray):
        img = frame.copy()
        img_h, img_w = img.shape[:2]

        # Draw a line through the center of the screen for calibration (should align
        # with the blue cross on the paper)
        cv2.line(img, (0, int(img_h / 2)), (img_w, int(img_h / 2)), (0, 0, 200), 1)
        cv2.line(img, (int(img_w / 2), 0), (int(img_w / 2), img_h), (0, 0, 200), 1)

        # Blur, image format conversion stuff
        frame_resize = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

        # Find contours of pixels that match our target color range
        c_min, c_max = color_range[self.target_color]
        frame_mask = cv2.inRange(frame_lab, c_min, c_max)
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6,6),np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8))
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        area_contour, area = get_area_max_contour(contours)

        if area_contour is not None and area > 2500:
            # Get a bounding box for the contour
            rect = cv2.minAreaRect(area_contour)
            box = np.int0(cv2.boxPoints(rect))

            # Draw a bounding box around the contour and label it with the
            # converted coordinates
            roi = getROI(box)
            get_roi = True
            img_centerx, img_centery = getCenter(rect, roi, self.size, square_length)
            world_x, world_y = convertCoordinate(img_centerx, img_centery, self.size)
            cv2.drawContours(img, [box], -1, range_rgb[self.target_color], 2)
            cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_rgb[self.target_color], 1)

            cv2.putText(img, "Color: " + self.target_color, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, range_rgb[self.target_color], 2)
        return img


if __name__ == '__main__':
    my_camera = Camera.Camera()
    my_camera.camera_open()
    tracker = CubeTracker()
    while True:
        img = my_camera.frame
        if img is not None:
            frame = img.copy()
            Frame = tracker.track(frame)
            cv2.imshow('Frame', Frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()

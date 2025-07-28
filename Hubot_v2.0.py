#!/usr/bin/python3

### Hubot
# Version 2.0

### Patch note
# 1. First release

import numpy as np
import cv2
import serial


class Hubot:
    def __init__(self):
        self.ser = serial.Serial(port="COM5",
                                 baudrate=57600,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS,
                                 timeout=0)
        
        self.cap = cv2.VideoCapture(1)
        self.w = 0
        self.h = 0

        self.MIN_DETECT_WIDTH = 45
        self.MIN_DETECT_HEIGHT = 10

        self.MIN_LINE_HSV = np.array([20, 125, 50])
        self.MAX_LINE_HSV = np.array([75, 255, 220])
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        self.LOWER_LINE_MASK_LIMIT = 1/3
        self.UPPER_LINE_MASK_LIMIT = 0
        self.LOWER_OBJECT_MASK_LIMIT = 2/4
        self.UPPER_OBJECT_MASK_LIMIT = 1/4


    def serial_write(self, num):
        self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))


    def get_center(self, contours, dst):
        res = False
        center_pos = (0, 0)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.MIN_DETECT_WIDTH and h > self.MIN_DETECT_HEIGHT:
                res = True
                center_x = int(x + w/2)
                center_y = int(y + h/2)
                center_pos = (center_x, center_y)

                cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.circle(dst, (center_x, center_y), 5, (0, 255, 255), -1)
        return res, center_pos

    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))


    def get_offset(self, theta):
        offset = int(theta * 1024/300)
        if offset <= -50:
            return 0
        if offset >= 50:
            return 100
        return offset + 50


    def main(self):
        _, dst = self.cap.read()

        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        self.h, self.w = dst.shape[:2]

        line_mask = cv2.inRange(hsv, self.MIN_LINE_HSV, self.MAX_LINE_HSV)
        object_mask = cv2.inRange(hsv, self.MIN_OBJECT_HSV, self.MAX_OBJECT_HSV)

        line_mask[int(self.h*self.LOWER_LINE_MASK_LIMIT):self.h, :] = 0
        line_mask[0:int(self.h*self.UPPER_LINE_MASK_LIMIT), :] = 0
        object_mask[int(self.h*self.LOWER_OBJECT_MASK_LIMIT):self.h, :] = 0
        object_mask[0:int(self.h*self.UPPER_OBJECT_MASK_LIMIT), :] = 0

        line_res, line_center_pos = self.get_center(cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)
        object_res, _ = self.get_center(cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)

        cv2.imshow("dst", dst)
        cv2.waitKey(1)

        # Object detected
        if object_res == True:
            self.serial_write(200)
            print("Object detected.")
            return
        
        # Line detected
        if line_res == True:
            theta = self.get_theta((self.w/2, self.h*3/4), line_center_pos)
            offset = self.get_offset(theta)

            self.serial_write(offset)
            return
        

        # Line not detected
        else:
            print("Line not detected.")
            return
    

    def run(self):
        try:
            while True:
                self.main()
        except KeyboardInterrupt:
            print("Pressed Ctrl+C")
        self.cap.release()
        cv2.destroyAllWindows()
        self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()
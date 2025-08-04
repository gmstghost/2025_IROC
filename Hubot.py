#!/usr/bin/python3

### Hubot
# Version 2.11.3

### Patch note



import numpy as np
import cv2
import serial
import logging

import tracker as tr



class Hubot:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(lineno)d] %(message)s")

        try:
            self.ser = serial.Serial(port="/dev/ttyUSB0",
                                    baudrate=57600,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    bytesize=serial.EIGHTBITS,
                                    timeout=0)
            self.ser_enabled = True
        except serial.SerialException as e:
            logging.warning(f"SerialError: {str(e)}")
            self.ser_enabled = False

        self.cap = cv2.VideoCapture(0)
        self.cam_w = 0
        self.cam_h = 0

        self.MIN_DETECT_WIDTH = 30
        self.MIN_DETECT_HEIGHT = 10
        self.MIN_DETECT_AREA = 5000

        self.MIN_LINE_HSV = np.array([20, 50, 100])
        self.MAX_LINE_HSV = np.array([30, 230, 230])
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        self.UPPER_LINE_MASK_LIMIT = 0.4
        self.LOWER_LINE_MASK_LIMIT = 1.0
        
        self.CURRENT_LINE_POS = 1.5

        self.UPPER_OBJECT_MASK_LIMIT = 0.50
        self.LOWER_OBJECT_MASK_LIMIT = 1.00
        self.LEFT_OBJECT_MASK_LIMIT = 0.15
        self.RIGHT_OBJECT_MASK_LIMIT = 0.85

        self.TURN_RANGE = 65



    def serial_write(self, num):
        if self.ser_enabled:
            if num < 0 or num > 255:
                logging.error("SerialError: number out of range (0-255)")
                return

            try:
                self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))
            except serial.SerialException as e:
                logging.error(f"SerialError: {str(e)}")


    def vsl_rect(self, dst, rect, *args):
        cv2.rectangle(dst, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), *args)


    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))


    def get_offset(self, theta, offset_limit=75):
        offset = int(theta * 0.75)
        if offset <= -offset_limit:
            return 0
        if offset >= offset_limit:
            return offset_limit*2
        return offset + offset_limit


    def main(self):
        cam_ret, dst = self.cap.read()
        if not cam_ret:
            logging.warning("CameraError: Failed to capture camera")
            return False

        dst = cv2.GaussianBlur(dst, (0, 0), 10)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        self.cam_h, self.cam_w = dst.shape[:2]

        line_mask = cv2.inRange(hsv, self.MIN_LINE_HSV, self.MAX_LINE_HSV)
        object_mask = cv2.inRange(hsv, self.MIN_OBJECT_HSV, self.MAX_OBJECT_HSV)

        line_mask[:int(self.cam_h*self.UPPER_LINE_MASK_LIMIT), :] = 0
        line_mask[int(self.cam_h*self.LOWER_LINE_MASK_LIMIT):, :] = 0
        object_mask[:int(self.cam_h*self.UPPER_OBJECT_MASK_LIMIT), :] = 0
        object_mask[int(self.cam_h*self.LOWER_OBJECT_MASK_LIMIT):, :] = 0
        object_mask[:, :int(self.cam_w*self.LEFT_OBJECT_MASK_LIMIT)] = 0
        object_mask[:, int(self.cam_w*self.RIGHT_OBJECT_MASK_LIMIT):] = 0

        line_hulls = tr.extract_convexHulls(line_mask, min_area=self.MIN_DETECT_AREA)
        line_hulls = tr.sort_contour_area(line_hulls)
        object_rects = tr.extract_rects(object_mask, min_w=self.MIN_DETECT_WIDTH, min_h=self.MIN_DETECT_HEIGHT)
        object_rects = tr.sort_rect_distance_point(object_rects, (self.cam_w/2, None))


        # Object detected
        if object_rects:
            object_rect = object_rects[0]

            self.vsl_rect(dst, object_rect, (255, 127, 0), 3)

            self.serial_write(255)
            logging.info("Object detected    Serial written: 255(PUNCH)")

        # Line detected
        elif line_hulls:
            line_hull = line_hulls[0]
            line_pos = tr.sel_contour_distance_point(line_hull, (self.cam_w/2, self.cam_h*self.CURRENT_LINE_POS), 2)

            cv2.drawContours(dst, line_hulls, -1, (0, 255, 255), 1)
            cv2.drawContours(dst, [line_hull], -1, (0, 255, 255), 3)
            cv2.line(dst, (int(self.cam_w/2), int(self.cam_h*self.CURRENT_LINE_POS)), line_pos, (0, 255, 255), 1)

            theta = self.get_theta((self.cam_w/2, self.cam_h*self.CURRENT_LINE_POS), line_pos)
            if theta < -self.TURN_RANGE:
                self.serial_write(251)
                logging.info(f"Theta: {theta:6.2f}    Serial written: 251(LEFT_TURN)")
            elif theta > self.TURN_RANGE:
                self.serial_write(252)
                logging.info(f"Theta: {theta:6.2f}    Serial written: 252(RIGHT_TURN)")
            else:
                offset = self.get_offset(theta)
                self.serial_write(offset)
                logging.info(f"Theta: {theta:6.2f}    Serial written: {offset:3}")

        # Line not detected
        else:
            logging.info("Line not detected")

        cv2.imshow("dst", dst)
        cv2.imshow("line", line_mask)
        cv2.imshow("object", object_mask)

        key = cv2.waitKey(1)
        if key == ord("q"):
            logging.info("Pressed 'Q'")
            return False

        return True


    def run(self):
        try:
            while True:
                if not self.main():
                    break
        except KeyboardInterrupt:
            logging.info("Pressed 'Ctrl+C'")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ser_enabled:
                self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()
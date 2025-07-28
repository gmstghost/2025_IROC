#!/usr/bin/python3

### Hubot
# Version 2.8.0

### Patch note



import numpy as np
import cv2
import serial 
import logging
from collections import deque

from Selector import Selector


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

        self.MIN_LINE_HSV = np.array([20, 50, 100])
        self.MAX_LINE_HSV = np.array([30, 230, 230])
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        self.UPPER_LINE_MASK_LIMIT = 2/4
        self.LOWER_LINE_MASK_LIMIT = 3/4
        self.CURRENT_LINE_POS = 5/4
        self.UPPER_OBJECT_MASK_LIMIT = 1/4
        self.LOWER_OBJECT_MASK_LIMIT = 2/4

        self.line_stack = deque(maxlen=10)
        self.object_stack = deque(maxlen=10)
        self.line_selector = Selector(self.line_stack)
        self.obejct_selector = Selector(self.object_stack)


    def serial_write(self, num):
        if self.ser_enabled:
            if num < 0 or num > 255:
                logging.error("SerialError: number out of range (0-255)")
                return

            try:
                self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))
            except serial.SerialException as e:
                logging.error(f"SerialError: {str(e)}")


    def get_rects(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
            if rect_w >= self.MIN_DETECT_WIDTH and rect_h >= self.MIN_DETECT_HEIGHT:
                rects.append((rect_x, rect_y, rect_w, rect_h))
        return rects


    def vsl_rect(self, dst, rect, *args):
        cv2.rectangle(dst, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), *args)


    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))


    def get_offset(self, theta, offset_limit=75):
        offset = int(theta * 0.8)
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

        line_rects = self.line_selector.SORT_DISTANCE_POS(self.get_rects(line_mask), (self.cam_w/2, None))
        object_rects = self.obejct_selector.SORT_DISTANCE_POS(self.get_rects(object_mask), (self.cam_w/2, None))


        # Object detected
        if object_rects:
            object_rect = object_rects[0]
            self.object_stack.append(object_rect)
            self.vsl_rect(dst, object_rect, (255, 127, 0), 3)

            self.serial_write(200)
            logging.info("Object detected")

        # Line detected
        elif line_rects:
            line_rect = self.line_selector.AVERAGE_PRE(line_rects[0], weights=(7, 6, 4, 2, 1))
            line_pos = self.line_selector.SEL_DISTANCE_POS_POWER(line_rect, (self.cam_w/2, None), (self.cam_w/2, None))
            self.line_stack.append(line_rect)
            self.vsl_rect(dst, line_rect, (0, 255, 255), 3)
            cv2.line(dst, (int(self.cam_w/2), int(self.cam_h*self.CURRENT_LINE_POS)), line_pos, (0, 255, 255), 1)

            theta = self.get_theta((self.cam_w/2, self.cam_h*self.CURRENT_LINE_POS), line_pos)
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
        except Exception as e:
            logging.error(f"UnexpectedError: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ser_enabled:
                self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()
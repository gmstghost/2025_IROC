#!/usr/bin/python3

### Hubot
# Version 2.2

### Copyright
# Code 손민준, 김동연, 배준형
# Annotate 안제욱
# Help 이승민

### Patch note
# 1. Improved exception handling
#      serial port open failure
#      serial write range error
#      runtime error
#
# 2. Improved detecting center
#      use (-1, -1) to avoid empty list
#
# 3. Improved program shutdown process
#      press 'q' to exit
#      serial port closes only if opened
#
# 4. Improved preprocessing
#      gaussian blur on HSV conversion
#
# 5. Improved debugging output
#      error message
#      value print



import numpy as np
import cv2
import serial


class Hubot:
    def __init__(self):
        try:
            self.ser = serial.Serial(port="/dev/ttyUSB0",
                                    baudrate=57600,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    bytesize=serial.EIGHTBITS,
                                    timeout=0)
        except serial.SerialException as e:
            print("Failed to open serial port: ", str(e))
            self.ser = None
        
        self.cap = cv2.VideoCapture(0)
        self.cam_w = 0
        self.cam_h = 0

        self.MIN_DETECT_WIDTH = 45
        self.MIN_DETECT_HEIGHT = 10

        self.MIN_LINE_HSV = np.array([20, 125, 50])
        self.MAX_LINE_HSV = np.array([75, 255, 220])
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        self.LOWER_LINE_MASK_LIMIT = 1/3
        self.UPPER_LINE_MASK_LIMIT = 0
        self.CURRENT_LINE_POS = 3/4
        self.LOWER_OBJECT_MASK_LIMIT = 2/4
        self.UPPER_OBJECT_MASK_LIMIT = 1/4


    def serial_write(self, num):
        if self.ser == None:
            print("Failed to serial write:", "Serial port is disabled")
            return
        if num < 0 or num > 255:
            print("Failed to serial write:", "number to write is out of range(0-255)")
            return

        try:
            self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))
        except serial.SerialException as e:
            print("SerialException:", str(e))
        except Exception as e:
            print("Unexpected Error:", str(e))


    def get_center(self, contours, dst):
        res = False
        center_pos = [(-1, -1)]
        for contour in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
            if rect_w > self.MIN_DETECT_WIDTH and rect_h > self.MIN_DETECT_HEIGHT:
                res = True
                center_x = int(rect_x + rect_w/2)
                center_y = int(rect_y + rect_h/2)
                center_pos.append((center_x, center_y))

                cv2.rectangle(dst, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0, 255, 255), 2)
                cv2.circle(dst, (center_x, center_y), 5, (0, 255, 255), -1)
        
        center_pos = sorted(center_pos, key=lambda x:abs(x[0] - self.cam_w/2))
        return res, center_pos[0]


    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))


    def get_offset(self, theta, offset_limit=75):
        offset = int(theta * 1024/300)
        if offset <= -offset_limit:
            return 0
        if offset >= offset_limit:
            return offset_limit*2
        return offset + offset_limit


    def main(self):
        cam_res, dst = self.cap.read()
        if cam_res == False:
            print("CameraError: Failed to capture camera")
            return False

        dst = cv2.GaussianBlur(dst, (0, 0), 2)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        self.cam_h, self.cam_w = dst.shape[:2]

        line_mask = cv2.inRange(hsv, self.MIN_LINE_HSV, self.MAX_LINE_HSV)
        object_mask = cv2.inRange(hsv, self.MIN_OBJECT_HSV, self.MAX_OBJECT_HSV)

        line_mask[int(self.cam_h*self.LOWER_LINE_MASK_LIMIT):self.cam_h, :] = 0
        line_mask[0:int(self.cam_h*self.UPPER_LINE_MASK_LIMIT), :] = 0
        object_mask[int(self.cam_h*self.LOWER_OBJECT_MASK_LIMIT):self.cam_h, :] = 0
        object_mask[0:int(self.cam_h*self.UPPER_OBJECT_MASK_LIMIT), :] = 0

        line_res, line_center_pos = self.get_center(cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)
        object_res, _ = self.get_center(cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)

        cv2.imshow("dst", dst)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Pressed Q")
            return False

        # Object detected
        if object_res == True:
            self.serial_write(200)
            print("Object detected.")
            return True

        # Line detected
        if line_res == True:
            theta = self.get_theta((self.cam_w/2, self.cam_h*self.CURRENT_LINE_POS), line_center_pos)
            offset = self.get_offset(theta)

            self.serial_write(offset)
            print("Theta: ", theta)
            print("Written on serial:", offset)
            return True

        # Line not detected
        else:
            print("Line not detected.")
            return True


    def run(self):
        try:
            while True:
                if self.main() == False:
                    break
        except KeyboardInterrupt:
            print("Pressed Ctrl+C")
        except Exception as e:
            print("Unexpected Error:", str(e))
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ser != None:
                self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()
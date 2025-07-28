#!/usr/bin/python3

### Hubot
# Version 2.1

### Patch note
# 1. Improved exception handling
#      camera capture failure
#      serial write failure
#
# 2. Improved detecting center
#      the last detected center -> the center closest to the camera center
#
# 3. Improved program shutdown process
#
# 4. Improved variable naming
#      w -> cam_w
#      h -> cam_h
#
# 5. Improved debuging output

### Copyright
# Code 손민준, 김동연, 배준형
# Annotate 안제욱
# Help 이승민

import numpy as np
import cv2
import serial

class Hubot:
    def __init__(self):
        self.ser = serial.Serial(port="COM7",
                                 baudrate=57600,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS,
                                 timeout=0)
        
        self.cap = cv2.VideoCapture(0)                      # 카메라 불러오기
        self.cam_w = 0                                      # 카메라 너비
        self.cam_h = 0                                      # 카메라 높이

        self.MIN_DETECT_WIDTH = 45                          # 최소 감지 너비
        self.MIN_DETECT_HEIGHT = 10                         # 최소 감지 높이

        # 선 인식 범위
        self.MIN_LINE_HSV = np.array([20, 125, 50])
        self.MAX_LINE_HSV = np.array([75, 255, 220])

        # 장애물 인식 범위
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        self.LOWER_LINE_MASK_LIMIT = 1/3
        self.UPPER_LINE_MASK_LIMIT = 0
        self.CURRENT_LINE_POS = 3/4
        self.LOWER_OBJECT_MASK_LIMIT = 2/4
        self.UPPER_OBJECT_MASK_LIMIT = 1/4


    def serial_write(self, num):
        self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))


    def get_center(self, contours, dst):
        res = False                 # 감지 여부 false
        center_pos = []             # 여러개 감지 시 (center_x, center_y) tuple 저장용
        for contour in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)                  # 전달받은 외곽선의 외접사각형 반환
            if rect_w > self.MIN_DETECT_WIDTH and rect_h > self.MIN_DETECT_HEIGHT:      # 그 사각형의 너비와 높이가 임계값 넘으면
                res = True                                  # 감지 여부 true
                center_x = int(rect_x + rect_w/2)           # 직사각형 가로 중점 계산
                center_y = int(rect_y + rect_h/2)           # 직사각형 세로 중점 계산
                center_pos.append((center_x, center_y))     # tuple 저장

                cv2.rectangle(dst, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0, 255, 255), 2)      # dst (프레임)위에 노란색, 두께 2 외접사각형 그리기
                cv2.circle(dst, (center_x, center_y), 5, (0, 255, 255), -1)                                 # dst에 외접사각형 중심을 중점으로 한 노란색 원 그리기 (내부채움)
        
        if res == False:
            return res, (0, 0)      # 감지 안됐을때 false와 (0, 0) return
        
        center_pos = sorted(center_pos, key=lambda x:abs(x[0] - self.cam_w/2))  # x = center_pos / center_pos 튜플(x,y)들의 첫번째 항목에서 cam_w/2 빼기, 중앙 x좌표 차이가 작은순으로 sort
        return res, center_pos[0]                                               # 인식여부와 중앙에서 가장 가까운 튜플 리턴

    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))    # 두 점 사이의 라디안 각도 계산


    # main에서 호출한 get_theta에서 theta 구함, 기본인자 offset_limit=75
    # PI에서 보내는 값: 0-150, 모터 동작: -75~75
    def get_offset(self, theta, offset_limit=75):
        offset = int(theta * 1024/300)      # 구한 theta값을 모터 제어 인자 단위 (1024)로 변환
        if offset <= -offset_limit:         # 구한 offset이 limit보다 작으면 0리턴
            return 0    
        if offset >= offset_limit:          # 구한 offset이 limit보다 크면 limit*2 (150) 해서 리턴
            return offset_limit*2
        return offset + offset_limit        # 아니면 범위 수정만 해서 리턴


    def main(self): 
        cam_res, dst = self.cap.read()      # cam 프레임 불러오기
        if cam_res == False:                # cam 프레임 없을 때 CameraError 로깅, False 리턴
            print("CameraError: Cannot capture camera")
            return False
        
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)      # 컬러스페이스 변환 (BGR -> HSV)
        self.cam_h, self.cam_w = dst.shape[:2]          # frame의 가로, 세로 구하기 
    
        line_mask = cv2.inRange(hsv, self.MIN_LINE_HSV, self.MAX_LINE_HSV)              # Line HSV 범위에 맞는 색상을 프레임에서 추출                                                                                                                                                                                                                       
        object_mask = cv2.inRange(hsv, self.MIN_OBJECT_HSV, self.MAX_OBJECT_HSV)        # Object HSV 범위에 맞는 색상을 프레임에서 추출

        # 감지 범위 제한
        line_mask[int(self.cam_h*self.LOWER_LINE_MASK_LIMIT):self.cam_h, :] = 0
        line_mask[0:int(self.cam_h*self.UPPER_LINE_MASK_LIMIT), :] = 0
        object_mask[int(self.cam_h*self.LOWER_OBJECT_MASK_LIMIT):self.cam_h, :] = 0
        object_mask[0:int(self.cam_h*self.UPPER_OBJECT_MASK_LIMIT), :] = 0

        # Line과 Object의 윤곽선 추출
        line_res, line_center_pos = self.get_center(cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)
        object_res, _ = self.get_center(cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], dst)

        # 프레임 띄우기
        cv2.imshow("dst", dst)
        cv2.imshow("line", line_mask)
        cv2.imshow("object", object_mask)

        cv2.waitKey(1)      # 키입력 1ms 대기

        # Object detected   
        if object_res == True:
            self.serial_write(200)
            print("Object detected.")
            return True
        
        # Line detected
        if line_res == True:
            theta = self.get_theta((self.cam_w/2, self.cam_h*self.CURRENT_LINE_POS), line_center_pos)
            offset = self.get_offset(theta)

            try:
                self.serial_write(offset)
                print("Theta: ", theta)
                print("Writed on serial:", offset)
                return True
            except Exception as e:
                print("SerialError:", str(e))
                return False
        

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
            print("UnexpectedError:", str(e))
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()     
#!/usr/bin/python3

### Hubot
# Version 2.8.0

### Patch note

### Copyright
# Code 손민준, 김동연, 배준형
# Annotate 안제욱
# Help 이승민

# 각종 Module 호출
import numpy as np
import cv2
import serial 
import logging
from collections import deque

from Selector import Selector       # Selector.py 클래스 호출


class Hubot:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(lineno)d] %(message)s")    # 로그 설정 초기화

        try:                                    # Serial setting
            self.ser = serial.Serial(port="/dev/ttyUSB0",
                                    baudrate=57600,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    bytesize=serial.EIGHTBITS,
                                    timeout=0)
            self.ser_enabled = True
        except serial.SerialException as e:     # Serialerror``
            logging.warning(f"SerialError: {str(e)}")
            self.ser_enabled = False

        self.cap = cv2.VideoCapture(0)     # Camera Open
        self.cam_w = 0                     # 카메라 너비
        self.cam_h = 0                     # 카메라 높이

        self.MIN_DETECT_WIDTH = 30         # 최소 감지 너비
        self.MIN_DETECT_HEIGHT = 10        # 최소 감지 높이

        # 선 인식 범위
        self.MIN_LINE_HSV = np.array([20, 50, 100])
        self.MAX_LINE_HSV = np.array([30, 230, 230])

        # 장애물 인식 범위
        self.MIN_OBJECT_HSV = np.array([100, 100, 50])
        self.MAX_OBJECT_HSV = np.array([160, 255, 150])

        # 프레임 내에서의 라인 감지 범위
        self.UPPER_LINE_MASK_LIMIT = 2/4
        self.LOWER_LINE_MASK_LIMIT = 3/4

        # 임의의 기준점 잡아서 세타 구하기
        self.CURRENT_LINE_POS = 5/4

        # 프레임 내에서의 오브젝트 감지 범위
        self.UPPER_OBJECT_MASK_LIMIT = 1/4
        self.LOWER_OBJECT_MASK_LIMIT = 2/4

        # Line, Object 여러 개 표시하기 위해 stack 사용
        self.line_stack = deque(maxlen=10)
        self.object_stack = deque(maxlen=10)

        # Selector 클래스 생성
        self.line_selector = Selector(self.line_stack)
        self.obejct_selector = Selector(self.object_stack)


    def serial_write(self, num):
        if self.ser_enabled:                        # 시리얼 사용가능할 때
            if num < 0 or num > 255:                # 비정상 출력범위
                logging.error("SerialError: number out of range (0-255)")
                return
            try:                                    # 정상동작 (serial write)
                self.ser.write(bytearray([255, 85, num, 255-num, 0, 255]))
            except serial.SerialException as e:     # USB 하드웨어 issue
                logging.error(f"SerialError: {str(e)}")


    def get_rects(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # mask 이미지의 바깥쪽 외곽선 추출, 꼭짓점만 표시
        rects = []                                                                          # rects로 빈 리스트 생성
        for contour in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)                      # 전달받은 외곽선의 외접사각형 반환
            if rect_w >= self.MIN_DETECT_WIDTH and rect_h >= self.MIN_DETECT_HEIGHT:        # 그 사각형의 너비와 높이가 임계값 넘으면
                rects.append((rect_x, rect_y, rect_w, rect_h))                              # 리스트에 사각형 추가
        return rects                                                                        # rects 리스트 반환

    # 두 점으로 직사각형 그리기 (x, y), (x + w, y + h)
    def vsl_rect(self, dst, rect, *args):
        cv2.rectangle(dst, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), *args)


    # 두 점 (pos1, pos2) 사이의 라디안 각도 계산해서 반환
    def get_theta(self, pos1, pos2):
        return -np.degrees(np.arctan((pos1[0]-pos2[0]) / (pos1[1]-pos2[1])))


    def get_offset(self, theta, offset_limit=75):
        offset = int(theta * 0.8)           # 오프셋 값 설정
        if offset <= -offset_limit:         # 구한 offset이 limit보다 작으면 0리턴
            return 0    
        if offset >= offset_limit:          # 구한 offset이 limit보다 크면 limit*2 (150) 해서 리턴
            return offset_limit*2
        return offset + offset_limit        # 아니면 범위 수정만 해서 리턴


    def main(self):
        cam_ret, dst = self.cap.read()      # cam 프레임 불러오기
        if not cam_ret:                     # cam 프레임 없을 때 CameraError 로깅, False 리턴
            logging.warning("CameraError: Failed to capture camera")
            return False

        dst = cv2.GaussianBlur(dst, (0, 0), 10)         # 가우시안 블러 적용 (가우시안 커널: 자동, x방향 sigma: 10)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)      # 컬러스페이스 변환 (BGR -> HSV)
        self.cam_h, self.cam_w = dst.shape[:2]          # frame의 가로, 세로 구하기 

        line_mask = cv2.inRange(hsv, self.MIN_LINE_HSV, self.MAX_LINE_HSV)              # Line HSV 범위에 맞는 색상을 프레임에서 추출   
        object_mask = cv2.inRange(hsv, self.MIN_OBJECT_HSV, self.MAX_OBJECT_HSV)        # Object HSV 범위에 맞는 색상을 프레임에서 추출

        # 감지 범위 제한
        line_mask[:int(self.cam_h*self.UPPER_LINE_MASK_LIMIT), :] = 0
        line_mask[int(self.cam_h*self.LOWER_LINE_MASK_LIMIT):, :] = 0
        object_mask[:int(self.cam_h*self.UPPER_OBJECT_MASK_LIMIT), :] = 0
        object_mask[int(self.cam_h*self.LOWER_OBJECT_MASK_LIMIT):, :] = 0

        # Line과 Object의 윤곽선 추출
        line_rects = self.line_selector.SORT_DISTANCE_POS(self.get_rects(line_mask), (self.cam_w/2, None))
        object_rects = self.obejct_selector.SORT_DISTANCE_POS(self.get_rects(object_mask), (self.cam_w/2, None))

        # Object detected
        if object_rects:
            object_rect = object_rects[0]
            self.object_stack.append(object_rect)
            self.vsl_rect(dst, object_rect, (255, 127, 0), 3)

            # 로깅
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

        # OpenCV 창 열기
        cv2.imshow("dst", dst)
        cv2.imshow("line", line_mask)
        cv2.imshow("object", object_mask)

        key = cv2.waitKey(1)        # 키입력 1ms 대기

        # Q 누르면 종료
        if key == ord("q"):
            logging.info("Pressed 'Q'")
            return False

        return True


    def run(self):
        try:
            # 오류 뜨면 종료
            while True:
                if not self.main():
                    break   # False, None 등을 반환하면 루프 종료

        # Ctrl + C 누르면 종료
        except KeyboardInterrupt:
            logging.info("Pressed 'Ctrl+C'")

        except Exception as e:          # 예외 발생 시 로깅
            logging.error(f"UnexpectedError: {str(e)}")
        finally:
            self.cap.release()          # 카메라 프레임 캡쳐 끝
            cv2.destroyAllWindows()     # OpenCV 창 닫기
            if self.ser_enabled:        # 시리얼 비활성화되면 닫기
                self.ser.close()



if __name__ == "__main__":
    hubot = Hubot()
    hubot.run()
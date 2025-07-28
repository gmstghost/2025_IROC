#!/usr/bin/python3
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# 라인 검출 hsv 지정
MIN_LINE_HSV = np.array([45, 130, 70])
MAX_LINE_HSV = np.array([75, 255, 220])

# 라인 검출 범위 지정
LOWER_LINE_MASK_LIMIT = 1
UPPER_LINE_MASK_LIMIT = 2/3

# 오브젝트 검출 hsv 지정
MIN_OBJECT_HSV = np.array([30,100,50])
MAX_OBJECT_HSV = np.array([70,255,200])
# 오브젝트 검출 위치 지정
OBJECT_MASK_POSITION = 1/2

# 모션 범위 설정
SMALL_TURN_RANGE = 10
BIG_TURN_RANGE = 15
# SMALL_SIDE_STEP_RANGE = 150
# BIG_SIDE_STEP_RANGE = 250

counter = 0

def callback(image_msg):
    global counter
    
    # 외곽선 검출 함수
    def get_hull(input_contours):
        # 외곽선 검출 및 시각화
        for contour in input_contours:
            hull = cv2.convexHull(contour, clockwise=True)
            cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2) #v
        try:
            # 외곽선 꼭짓점을 y좌표 기준으로 정렬
            hull = sorted(hull.reshape(-1, 2).tolist(), key=lambda x:x[1])
            # 꼭짓점이 너무 적은 외곽선을 제외
            if len(hull) < 4: return (0, (0, 0), (0, 0))
            # 외곽선 최상단, 최하단 좌표 return
            else:return (1, 
                        (int(np.mean(list(zip(*hull[:2]))[0])), int(np.mean(list(zip(*hull[:2]))[1]))),
                        (int(np.mean(list(zip(*hull[-2:]))[0])), int(np.mean(list(zip(*hull[-2:]))[1]))))
        except:return (0, (0, 0), (0, 0))

    # 두 좌표 사이의 각도 계산 함수
    def get_theta(position1, position2):
        return np.degrees(np.arctan((position1[0] - position2[0]) / (position1[1] - position2[1])))

   # 모션 결정 함수
    def determine_motion():
        if object_ret:return "PU"
        elif line_ret:
            # if upper_line_center_x < -BIG_SIDE_STEP_RANGE + window_width / 2 and lower_line_center_x < -BIG_SIDE_STEP_RANGE + window_width / 2:return "SL"
            # elif upper_line_center_x > BIG_SIDE_STEP_RANGE + window_width / 2 and lower_line_center_x > BIG_SIDE_STEP_RANGE + window_width / 2:return "SR"
            # elif upper_line_center_x < -SMALL_SIDE_STEP_RANGE + window_width / 2 and lower_line_center_x < -SMALL_SIDE_STEP_RANGE + window_width / 2:return "sl"
            # elif upper_line_center_x > SMALL_SIDE_STEP_RANGE + window_width / 2 and lower_line_center_x > SMALL_SIDE_STEP_RANGE + window_width / 2:return "sr"
            if theta < -BIG_TURN_RANGE:return "TL"
            elif theta > BIG_TURN_RANGE:return "TR"
            elif theta < -SMALL_TURN_RANGE:return "tl"
            elif theta > SMALL_TURN_RANGE:return "tr"
            else:return "WF"
        else:return "ST"

   # 카메라 영상 불러와서 디코딩
    try:dst = cv2.imdecode(np.frombuffer(image_msg.data, np.uint8), cv2.IMREAD_COLOR)
    except CvBridgeError as error:
        rospy.logerr(error)
        return
    
    cv2.waitKey(1)

   # 가우시안 블러 및 상하반전
    dst = cv2.flip(cv2.GaussianBlur(dst, (0, 0), 1), -1)
   # HSV로 변환
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

   # 영상 높이, 너비 읽어오기
    window_width = int(dst.shape[1])
    window_height = int(dst.shape[0])

   # 라인 마스크 HSV 범위 지정
    line_mask = cv2.inRange(hsv, MIN_LINE_HSV, MAX_LINE_HSV)

   # 라인 마스크 검출 범위 지정
    min_line_mask = int(window_height * UPPER_LINE_MASK_LIMIT)
    max_line_mask = int(window_height * LOWER_LINE_MASK_LIMIT)
    line_mask[:min_line_mask, :] = 0
    line_mask[max_line_mask:, :] = 0

    # line_ret 라인 감지 여부/upper_line_center 라인 최상단 좌표
    line_ret, upper_line_center, _ = get_hull(cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])


   # 오브젝트 마스크 HSV 범위 지정
    object_mask = cv2.inRange(hsv, MIN_OBJECT_HSV, MAX_OBJECT_HSV)

   # 오브젝트 마스크 y좌표 범위 지정
    min_object_mask = int(window_height * OBJECT_MASK_POSITION + 10)
    max_object_mask = int(window_height * OBJECT_MASK_POSITION - 10)
    object_mask[:min_object_mask, :] = 0
    object_mask[max_object_mask:, :] = 0

   # 오브젝트 마스크 x좌표 범위 지정
    object_mask[:, :int(window_width * 1/3)] = 0
    object_mask[:, int(window_width * 2/3):] = 0

   # object_ret 오브젝트 감지 여부
    object_ret, _, _ = get_hull(cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    # 모션 발행을 위한 Publisher 생성
    chat_pub = rospy.Publisher("/chatter", String, queue_size=1)
    # 라인이 감지 되었다면 각도 계산
    if line_ret:
        theta = get_theta((window_width / 2, 0), upper_line_center)
    # 모션 발행
    try:
        motion = determine_motion()
        if counter < 3:
            counter += 1
        else:
            chat_pub.publish(motion)
            counter = 0
        rospy.loginfo(motion)
    except:pass
    # 카메라 영상 출력
    cv2.imshow("dst", dst)
if __name__ == "__main__":
    # ROS에서 OpenCV로 영상 불러오기
    bridge = CvBridge()
    # ROS 노드 설정
    rospy.init_node("main", anonymous=True)
    rospy.loginfo("Subscribe images from topic /compressed_image ...")

    # 카메라 영상를 받아오기 위한 Subscriber 생성
    image_subscriber = rospy.Subscriber("/compressed_image", CompressedImage, callback)
    # 반복
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import json
from match import match_pose_realtime
from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--upper_body_only', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 参数
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # 标准动作文件总帧数
    std_cnt = 280
    # 计数，当前进行到标准文件第count帧
    count = 0
    # dtw阈值
    threshold = 100
    # 标准动作序列文件
    file_std_pose = "pose_std.json"
    # 用户动作序列文件
    file_usr_pose = "pose_usr.json"

    # 调试用，清空已有的usr序列文件
    with open(file_usr_pose, 'w') as fp:
        fp.truncate()

    # 摄像头准备
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 模型加载
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    bre = True

    while bre:
        with open(file_std_pose, "r") as fp:
            for line in fp.readlines():
                count += 1

                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)  # 图像翻转
                debug_image = copy.deepcopy(image)

                # 姿态关键点检测
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = pose.process(image)

                # 关键点绘制
                if results.pose_landmarks is not None:
                    # 实时姿态显示
                    debug_image = draw_realtime_pose(debug_image, results.pose_landmarks, upper_body_only)
                    # 标准姿态显示
                    landmarks = json.loads(line)
                    debug_image = draw_std_pose(debug_image, landmarks)

                    # 用户姿态实时帧数据判断
                    # 帧关键点数据采集
                    image_width, image_height = image.shape[1], image.shape[0]
                    landmark_point = []
                    for index, landmark in enumerate(results.pose_landmarks.landmark):
                        if landmark.visibility < 0 or landmark.presence < 0:
                            continue
                        landmark_x = min(int(landmark.x * image_width), image_width - 1)
                        landmark_y = min(int(landmark.y * image_height), image_height - 1)
                        landmark_point.append((index, landmark_x, landmark_y))

                    # 有待优化：写进列表而不是文件
                    with open(file_usr_pose, 'a') as usr_fp:
                        json.dump(landmark_point, usr_fp)
                        usr_fp.write("\n")

                    # 实时dtw计算
                    dtw = match_pose_realtime(file_std_pose, file_usr_pose, count, std_cnt)

                    # 调试用输出
                    print(dtw)

                    # 大于指定阈值，动作不标准，输出提示
                    if (dtw > threshold):  
                        cv.putText(debug_image, "WRONG", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

                    # 标准动作演示完毕，退出
                    if (count >= std_cnt):
                        bre = False

                    # 键盘按键处理 ESC退出
                    key = cv.waitKey(1)
                    if key == 27:  # ESC
                        bre = False

                    # 图像显示
                    cv.imshow('MediaPipe Pose Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()

# 标准动作绘制
def draw_std_pose(image, landmarks, visibility_th=0.5):
    landmark_point = []
    std_pose_color = (113, 248, 249)

    for landmark in landmarks:
        index = landmark[0]
        landmark_x = landmark[1]
        landmark_y = landmark[2]

        landmark_point.append([1, (landmark_x, landmark_y)])

        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, std_pose_color, 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, std_pose_color, 2)
        if index == 16:  # 左手腕
            cv.circle(image, (landmark_x, landmark_y), 5, std_pose_color, 2)
        if index == 23:  # 左腰
            cv.circle(image, (landmark_x, landmark_y), 5, std_pose_color, 2)

    if len(landmark_point) > 0:
        # 肩
        cv.line(image, landmark_point[11][1], landmark_point[12][1], std_pose_color, 2)
        # 左腕
        cv.line(image, landmark_point[12][1], landmark_point[14][1], std_pose_color, 2)
        cv.line(image, landmark_point[14][1], landmark_point[16][1], std_pose_color, 2)
        # 左手
        cv.line(image, landmark_point[16][1], landmark_point[18][1], std_pose_color, 2)
        cv.line(image, landmark_point[18][1], landmark_point[20][1], std_pose_color, 2)
        cv.line(image, landmark_point[20][1], landmark_point[22][1], std_pose_color, 2)
        cv.line(image, landmark_point[22][1], landmark_point[16][1], std_pose_color, 2)

    return image


# 实时动作绘制
def draw_realtime_pose(image, landmarks, upper_body_only, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    realtime_pose_color = (249, 248, 113)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue
        """
        cv.putText(image, str(index),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)
        """
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, realtime_pose_color, 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, realtime_pose_color, 2)
        if index == 16:  # 左手腕
            cv.circle(image, (landmark_x, landmark_y), 5, realtime_pose_color, 2)

    if len(landmark_point) > 0:

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1], realtime_pose_color, 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1], realtime_pose_color, 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1], realtime_pose_color, 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1], realtime_pose_color, 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1], realtime_pose_color, 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1], realtime_pose_color, 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1], realtime_pose_color, 2)

    return image


if __name__ == '__main__':
    main()

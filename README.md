# 基于Mediapipe的人体姿态识别与匹配计算

## 要求

- 推荐环境（仅在此环境中测试过）：**Python 3.7.6 64-bit**
- mediapipe 0.8.3 或更高版本： `pip install mediapipe`
- OpenCV 3.4.2 或更高版本： `pip install opencv-python`
- dtaidistance：`pip install dtaidistance`

## 运行说明

- 运行`main.py`
- 启动摄像头画面后跟随黄色标线（即标准文件动作序列，相关数据保存在`pose_std.json`文件中）完成【左肘伸展弯曲】动作，若实时动作与标准动作误差较大，将在画面左上角显示红色`WRONG`提示信息。
- 用户动作关键点序列保存在`pose_usr.json`文件中

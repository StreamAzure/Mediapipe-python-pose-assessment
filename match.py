import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import json

def get_angle(point, point_1, point_2):
    """
    功能：根据三点坐标计算夹角（以point为顶点）
    参数：三个点的坐标，均为numpy数组
    返回值：以point为顶点的夹角角度（°）
    """
    a = point_1 - point
    b = point_2 - point
    # 两个向量
    cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if (cosangle < -1 or cosangle > 1):
        return 0
    # 少数情况下会出现数据越界情况，这里简单处理，直接返回0
    # 有待优化：如直接采用前一帧数据
    angle = np.arccos(cosangle)
    return np.degrees(angle)

def point_to_angle(fp, begin, end):
    """
    功能：关键点坐标数据转换为角度数据
    参数：fp 文件对象，起始行数，结束行数
    返回值：元素为一帧角度数据的一维numpy数组
    """
    angle_array = []
    for line in fp.readlines()[begin:end:1]:
        landmarks = json.loads(line)
        for landmark in landmarks:
            index = landmark[0]
            landmark_x = landmark[1]
            landmark_y = landmark[2]

            if index == 14:
                point = np.array((landmark_x,landmark_y))
            if index == 12:
                point_1 = np.array((landmark_x, landmark_y))
            if index == 16:
                point_2 = np.array((landmark_x, landmark_y))
                
        angle = get_angle(point, point_1, point_2)
        angle_array.append(angle)
    
    return np.array(angle_array)

def match_pose_realtime(std_fname, usr_fname, cnt, std_cnt):
    """
    功能：实时计算两个左肩外展屈肘动作序列的dtw距离
    参数：
        std_fname:标准动作序列文件名
        usr_fname:用户动作序列文件名（包含最新一帧动作的数据）
        cnt:标准动作/用户动作进行到的帧
        std_cnt:标准动作总帧数
    返回值：两个动作序列的dtw距离，值越小动作越匹配
    """

    #有待优化：即用户数据可以不先保存到文件，而是以列表形式
    #全部结束后再存入文件，作离线对比
    #如果不需要离线对比，只列表形式也行

    fp_std = open(std_fname, 'r')
    fp_usr = open(usr_fname, 'r')

    if (cnt > 10 and cnt < std_cnt - 15):
        std_lelbow = point_to_angle(fp_std, cnt - 10, cnt - 5)
        # 标准动作左肘角度序列
        usr_lelbow = point_to_angle(fp_usr, cnt - 5, cnt)
        # 用户动作左肘角度序列
    
    elif (cnt >= std_cnt - 15):
        std_lelbow = point_to_angle(fp_std, cnt - 5, std_cnt)
        usr_lelbow = point_to_angle(fp_usr, cnt - 15, cnt)

    fp_std.close()
    fp_usr.close()


    lelbow_d = 0

    if (cnt > 10):
        lelbow_d = dtw.distance(std_lelbow, usr_lelbow)
    # 左肘动作dtw距离

    return lelbow_d
    # 这个返回值越小越好

"""
def match_pos_offline(std_fname, usr_fname, img_lshoul, img_lwlbow):

    功能：离线计算两个左肩外展屈肘动作序列的dtw距离
    参数：两个包含关键点坐标的json文件的文件名，左肩角度序列图像，左肘角度序列图像
    返回值：两个动作序列的dtw距离，值越小动作越匹配

    std = open(std_fname, 'r')
    usr = open(usr_fname, 'r')
    std_lelbow = point_to_angle(std)
    # 标准动作左肘角度序列
    usr_lelbow = point_to_angle(usr)
    # 用户动作左肘角度序列
    std.close()
    usr.close()

    std = open(std_fname, 'r')
    usr = open(usr_fname, 'r')
    std_lshoul = point_to_angle(std)
    # 标准动作左肩角度序列
    usr_lshoul = point_to_angle(usr)
    # 用户动作左肩角度序列
    std.close()
    usr.close()

    lshoul_d = dtw.distance(std_lshoul, usr_lshoul)
    # 左肩动作dtw距离
    lelbow_d = dtw.distance(std_lelbow, usr_lelbow)
    # 左肘动作dtw距离

    path = dtw.warping_path(std_lshoul, usr_lshoul)
    dtwvis.plot_warping(std_lshoul, usr_lshoul, path, filename=img_lshoul)

    path = dtw.warping_path(std_lelbow, usr_lelbow)
    dtwvis.plot_warping(std_lelbow, usr_lelbow, path, filename=img_lwlbow)

    return 0.5 * lshoul_d + 0.5 * lelbow_d
    #左肘和左肩的权重均为0.5
    #这个返回值越小越好
"""
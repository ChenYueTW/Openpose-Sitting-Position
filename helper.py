import sys
import os
import cv2
import numpy as np
from sys import platform

# --- A. OpenPose 初始化 (最亂的一段) ---
def init_openpose():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        if platform == "win32":
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            sys.path.append('../../python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "-1x160"
    params["model_pose"] = "BODY_25"
    params["number_people_max"] = 1

    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    return wrapper, op

# --- B. 快速開啟雙鏡頭 ---
def open_cameras():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(3, 640)
        cap.set(4, 480)
    return cap1, cap2

# --- C. 簡化推論過程 (輸入 frame, 回傳 keypoints) ---
def get_keypoints(op_wrapper, op_lib, frame):
    datum = op_lib.Datum()
    datum.cvInputData = frame
    op_wrapper.emplaceAndPop(op_lib.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData

# --- D. 畫面拼接顯示 ---
def show_combined(title, img1, img2, width=640, display_h=540):
    scale = display_h / 480
    display_w = int(width * scale)
    
    # 防呆：如果 OpenPose 沒回傳圖 (例如全黑)，就用全黑圖代替以免報錯
    if img1 is None: img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    if img2 is None: img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    small1 = cv2.resize(img1, (display_w, display_h))
    small2 = cv2.resize(img2, (display_w, display_h))
    cv2.imshow(title, np.hstack((small1, small2)))
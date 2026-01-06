import sys
import os
import cv2
import numpy as np
from sys import platform

# Openpose Initialization
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
    
    params["3d"] = True
    params["3d_min_views"] = 2
    abs_path = os.path.abspath("./camera_parameters") + os.sep
    params["camera_parameter_path"] = abs_path
    params["3d_views"] = 1

    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()
    return wrapper, op

def get_keypoints(op_wrapper, op_lib, frame):
    datum = op_lib.Datum()
    datum.cvInputData = frame
    op_wrapper.emplaceAndPop(op_lib.VectorDatum([datum]))
    return datum.poseKeypoints, datum.cvOutputData

def show_combined(title, img1, img2, width=640, display_h=540):
    scale = display_h / 480
    display_w = int(width * scale)
    
    if img1 is None: img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    if img2 is None: img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    small1 = cv2.resize(img1, (display_w, display_h))
    small2 = cv2.resize(img2, (display_w, display_h))
    cv2.imshow(title, np.hstack((small1, small2)))
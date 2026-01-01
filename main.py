import cv2
import numpy as np
import helper  # 匯入剛剛寫的工具
from cam_front import FrontSystem
from cam_side import SideSystem

def main():
    # 1. 系統初始化 (全部封裝在 helper 裡了)
    op_wrapper, op_lib = helper.init_openpose()
    cap1, cap2 = helper.open_cameras()
    
    # 2. 建立邏輯物件
    front_sys = FrontSystem()
    side_sys = SideSystem()

    print("系統啟動 (按 'q' 離開, 'r' 重置)")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1: frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret2: frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

        frame2 = side_sys.undistort(frame2)

        kp1, rendered_img1 = helper.get_keypoints(op_wrapper, op_lib, frame1)
        kp2, rendered_img2 = helper.get_keypoints(op_wrapper, op_lib, frame2)

        img_to_process_1 = rendered_img1 if rendered_img1 is not None else frame1
        img_to_process_2 = rendered_img2 if rendered_img2 is not None else frame2

        out1 = front_sys.process(img_to_process_1, kp1)
        out2 = side_sys.process(img_to_process_2, kp2)

        helper.show_combined("Posture Analysis (Full Skeleton)", out1, out2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): 
            front_sys.reset()
            side_sys.reset()

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
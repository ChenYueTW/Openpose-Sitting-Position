import cv2
import numpy as np
import helper
from cam_front import FrontSystem
from cam_side import SideSystem

def main():
    op_wrapper, op_lib = helper.init_openpose()
    
    front_sys = FrontSystem(camera_id=0)
    side_sys = SideSystem(camera_id=1)

    print("系統啟動!")
    print("按 'q' 離開, 'r' 重置計數")

    while True:
        out1 = front_sys.process(op_wrapper, op_lib)
        out2 = side_sys.process(op_wrapper, op_lib)

        helper.show_combined("Posture Analysis System", out1, out2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        if key == ord('r'): 
            front_sys.reset()
            side_sys.reset()
        
    front_sys.release()
    side_sys.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
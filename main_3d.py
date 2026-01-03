import cv2
import numpy as np
import helper
from cam_front import FrontSystem
from cam_side import SideSystem

# --- 3D 合成核心函式 ---
def estimate_3d_pose(kp_front, kp_side, width_side_cam=640):
    """
    輸入: 正面關鍵點, 側面關鍵點
    輸出: 3D 關鍵點列表 [(x, y, z), ...]
    """
    points_3d = {} # 使用字典方便存取 (例如: 0是鼻子, 1是脖子...)
    
    # 確保兩邊都有抓到人
    if (kp_front is None or len(kp_front) == 0) or \
       (kp_side is None or len(kp_side) == 0):
        return None

    # 取得第一個人 (Index 0)
    p1 = kp_front[0]
    p2 = kp_side[0]
    
    # OpenPose BODY_25 有 25 個點
    for i in range(25):
        # 格式: [x, y, confidence]
        fx, fy, fc = p1[i]
        sx, sy, sc = p2[i]

        # 兩邊信心度都大於 0.1 才算有效點
        if fc > 0.1 and sc > 0.1:
            # 1. X 座標 = 正面的 X
            x_3d = fx
            
            # 2. Y 座標 = 兩邊 Y 的平均 (校正鏡頭架設高度誤差)
            y_3d = (fy + sy) / 2
            
            # 3. Z 座標 = 側面的 X 
            # (注意：視側面鏡頭是放在受測者的左手邊還是右手邊)
            # 假設側面鏡頭在受測者的「左手邊」，人往後靠(遠離鏡頭)，X會變大
            # 這裡直接用 sx 當作深度
            z_3d = sx 
            
            points_3d[i] = (int(x_3d), int(y_3d), int(z_3d))
            
    return points_3d

def main():
    # 1. 初始化 OpenPose
    print("正在載入 OpenPose 模型...")
    op_wrapper, op_lib = helper.init_openpose()
    
    # 2. 啟動雙鏡頭系統
    print("正在開啟鏡頭...")
    front_sys = FrontSystem(0) # 請確認 ID
    side_sys = SideSystem(2)   # 請確認 ID

    print("--- 3D 姿態估測系統啟動 ---")
    print("按 'q' 離開")

    while True:
        # 3. 讀取影像
        # 注意：我們已經在 FrontSystem/SideSystem 裡寫好 undistort 了
        # 所以這裡呼叫 process 就會拿到「校正後」的影像
        # 但為了要做 3D 運算，我們需要 process 回傳「關鍵點」
        
        # 為了不改壞原本的 class，我們這裡手動操作一下流程
        
        # A. 讀取原始畫面
        ret1, raw1 = front_sys.cap.read()
        ret2, raw2 = side_sys.cap.read()
        
        if not ret1 or not ret2:
            print("遺失鏡頭訊號")
            continue

        # B. 執行校正 (Undistort) - 使用您算出來的參數
        frame1 = cv2.undistort(raw1, front_sys.K, front_sys.D)
        frame2 = cv2.undistort(raw2, side_sys.K, side_sys.D)

        # C. 取得關鍵點 (OpenPose 推論)
        kp1, rendered1 = helper.get_keypoints(op_wrapper, op_lib, frame1)
        kp2, rendered2 = helper.get_keypoints(op_wrapper, op_lib, frame2)

        # 處理畫面顯示 (如果沒抓到骨架就顯示原圖)
        img1 = rendered1 if rendered1 is not None else frame1
        img2 = rendered2 if rendered2 is not None else frame2

        # --- 4. 執行 3D 運算 ---
        pose_3d = estimate_3d_pose(kp1, kp2)

        if pose_3d:
            # 取得 頸部 (Index 1) 的 3D 座標
            # OpenPose Body_25: 0=Nose, 1=Neck, 2=RShoulder, 5=LShoulder...
            if 1 in pose_3d:
                neck_x, neck_y, neck_z = pose_3d[1]
                
                # 顯示數據在正面畫面上
                info = f"Neck 3D: X={neck_x} Y={neck_y} Z={neck_z}"
                cv2.putText(img1, info, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # --- 5. 簡單的 3D 姿態判斷 (Demo 亮點) ---
                # 判斷是否前傾：看 Z 值 (深度)
                # 假設標準坐姿的 Z 約為 300 (需依實際距離調整)
                standard_z = 320 
                threshold = 50 
                
                diff = neck_z - standard_z
                
                status = "Normal"
                color = (0, 255, 0)
                
                if diff > threshold: # Z 變大 (看側面鏡頭方向，通常是遠離鏡頭)
                    status = "Backward (Lean Back)"
                    color = (0, 255, 255)
                elif diff < -threshold: # Z 變小
                    status = "Forward (Turtle Neck)"
                    color = (0, 0, 255) # 紅色警告

                cv2.putText(img1, f"Status: {status}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 6. 畫面拼接顯示
        # 調整大小以防萬一
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        combined = np.hstack((img1, img2))
        
        # 加一點標示區隔
        cv2.line(combined, (640, 0), (640, 480), (255, 255, 255), 2)
        
        cv2.imshow("3D Posture Estimation System", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    front_sys.release()
    side_sys.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
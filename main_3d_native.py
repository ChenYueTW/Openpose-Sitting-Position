import cv2
import numpy as np
import helper
from cam_front import FrontSystem
from cam_side import SideSystem

def main():
    # 1. 初始化 OpenPose (開啟 3D 模式)
    # 確保 'camera_parameters' 資料夾內有 0.xml 和 1.xml
    # 且 helper.py 裡面已經開啟了 params["3d"] = True 和 params["3d_views"] = 1
    print("正在啟動 OpenPose 3D 模組...")
    op_wrapper, op_lib = helper.init_openpose()
    
    # 2. 使用您的類別來開啟鏡頭
    print("正在開啟鏡頭系統...")
    
    # 初始化正面鏡頭 (ID 0)
    front_sys = FrontSystem(0) 
    
    # 初始化側面鏡頭 (ID 2，請依實際情況確認)
    side_sys = SideSystem(2)   

    if not front_sys.cap.isOpened() or not side_sys.cap.isOpened():
        print("無法開啟鏡頭，請檢查 ID 連接")
        return

    print("--- OpenPose 3D 系統 (模組化版) 啟動 ---")
    print("按 'q' 離開")

    while True:
        # 3. 使用您的類別讀取影像
        ret1, frame1 = front_sys.cap.read()
        ret2, frame2 = side_sys.cap.read()
        
        if not ret1 or not ret2:
            print("遺失鏡頭訊號")
            continue

        # 4. 建立 OpenPose 輸入數據
        # 注意：直接傳入 frame1, frame2 (原始影像)，OpenPose 會根據 XML 自動做 undistort
        datum1 = op_lib.Datum()
        datum1.cvInputData = frame1
        
        datum2 = op_lib.Datum()
        datum2.cvInputData = frame2
        
        # 放入 VectorDatum (順序很重要：index 0 對應 0.xml, index 1 對應 1.xml)
        vector_datum = op_lib.VectorDatum([datum1, datum2])
        
        # 5. 推論與 3D 重建
        op_wrapper.emplaceAndPop(vector_datum)

        # 6. 取得 OpenPose 處理後的畫面 (已畫好骨架且去畸變)
        img1_out = datum1.cvOutputData
        img2_out = datum2.cvOutputData
        
        # 如果 OpenPose 沒畫圖 (例如沒偵測到人)，就用原始圖
        if img1_out is None: img1_out = frame1
        if img2_out is None: img2_out = frame2

        # 7. 取得 3D 數據進行邏輯判斷
        # keypoints3d 形狀: [人數, 25個關節, 4] -> (x, y, z, score)
        kp_3d = datum1.poseKeypoints3D 
        
        if kp_3d is not None and len(kp_3d) > 0:
            # 取得第一個人
            person_3d = kp_3d[0]
            
            # 取得 頸部 (Neck, ID=1)
            # 座標單位: 取決於您生成 XML 時的設定 (通常是 mm 或 m)
            neck = person_3d[1]
            nx, ny, nz, n_score = neck[0], neck[1], neck[2], neck[3]

            if n_score > 0.0:
                # 顯示 Z 軸深度 (假設單位是 mm，除以 10 變 cm)
                info = f"Neck Z: {nz/10:.1f} cm"
                cv2.putText(img1_out, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # --- 駝背判斷邏輯 ---
                # 假設標準坐姿距離鏡頭 80cm (800mm)
                # 當人往前傾 (駝背/烏龜頸)，Z 值會變小
                # 當人往後靠，Z 值會變大
                standard_depth = 800.0 
                tolerance = 100.0      # 容許範圍 10cm
                
                if nz < (standard_depth - tolerance):
                    status = "WARNING: Forward/Turtle"
                    color = (0, 0, 255) # 紅字警告
                elif nz > (standard_depth + tolerance):
                    status = "Backward"
                    color = (255, 0, 0)
                else:
                    status = "Good Posture"
                    color = (0, 255, 0)
                
                cv2.putText(img1_out, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # 8. 顯示結果 (合併視窗)
        # 調整大小以防萬一
        if img1_out.shape != img2_out.shape:
            img2_out = cv2.resize(img2_out, (img1_out.shape[1], img1_out.shape[0]))
            
        combined = np.hstack((img1_out, img2_out))
        cv2.imshow("3D Estimator (Modules)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    front_sys.cap.release()
    side_sys.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
from ultralytics import YOLO
import os
import cv2
import numpy
import json
import torch
import pandas as pd
import numpy as np


model = YOLO('yolov8m-pose.pt')
dir_path = "C:\\Users\\imlab\\Downloads\\eval_video"
index_pth= "C:\\Users\\imlab\\Desktop\\T2M\\HumanML3D\\index.csv"
dir = os.listdir(dir_path)
frame_buffer = []
frame_step = 8
count = 0

index_file = pd.read_csv(index_pth)
idx_count = 0

for item in dir:
    if item.lower().endswith(".mp4"):
        print(dir_path + "\\" + item)
        output = []
        cap = cv2.VideoCapture(dir_path +"\\" + item)

        #畫面設定
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        #輸出設定
        os.makedirs("./eval_joints", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("./eval_joints\\" + item[:-4] + '_output.mp4', fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            frame_buffer.append(frame)

            if(len(frame_buffer) >= 8):
                results = model.predict(frame_buffer, save=False, conf=0.3)
                for result in results:
                    keypoints = result.keypoints.xyn.cpu().numpy() #(frame, xyz)
                    keypoints = keypoints.tolist()[0]
                    output.append(keypoints)
                    annotated_frame = result.plot()
                    out.write(annotated_frame)
                frame_buffer = []
        # while(item != index_file.loc[count]['new_name'].replace(".npy", ".mp4")):
        #           print(item, index_file.loc[count]['new_name'].replace(".npy", ".mp4"))
        #           count += 1
        #utput = output[index_file.loc[count]['start_frame']:index_file.loc[count]['end_frame']]
        with open("./eval_joints\\" + item[:-4] +".json", "w") as f:
                        json.dump(output, f, indent=2)
        # output_m = output.copy()
        # for frame in output_m:
        #     for joint in frame:
        #         joint[0] = 1 -joint[0]
        
        # with open("./test_video\\M" + item[:-4] +".json", "w") as f:
        #                 json.dump(output_m, f, indent=2)
        count+=1
        cap.release()
        out.release()
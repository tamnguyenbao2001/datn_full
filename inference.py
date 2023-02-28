import threading
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import torch
import tensorflow as tf
import tensorflow_hub as hub

from my_utils import patch_extractor, distortion_free_resize, draw_connections, convert_to_xywh
from my_utils import get_detector, get_pose_estimator
from model import transformer
from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'


if str(ROOT /'StrongSORT-YOLO') not in sys.path:
    sys.path.append(str(ROOT /'StrongSORT-YOLO'))
if str(ROOT /'StrongSORT-YOLO'/ 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT /'StrongSORT-YOLO'/ 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2
### yolov5 weight 
model_dir = "weights/best.pt"
strong_sort_weights = "StrongSORT-YOLO\weights\osnet_x0_25_msmt17.pt"
config_strongsort= 'StrongSORT-YOLO\strong_sort\configs\strong_sort.yaml'
### Pose estimator
movenet_thunder = get_pose_estimator()

model = torch.hub.load("ultralytics/yolov5", "custom", path = model_dir, force_reload=False)

cfg = get_config()
cfg.merge_from_file(config_strongsort)
strongsort_model =  StrongSORT(
                strong_sort_weights,
                'cuda',
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
### Action recognizer
recognizer = transformer()

# Video dir
test_video_dir = "4106346062254.mp4"

def visualisation():
    # Step 1: extract patch
    cropped_img, left, right, top, bot = distortion_free_resize(frame[y1:y2, x1:x2, [2, 1, 0]], (256, 256))
    # Step 2: Run pose estimator
    outputs_t = movenet_thunder(cropped_img[np.newaxis,...])
    visualizing_keypoints_t = (outputs_t["output_0"].numpy())
    # Step 3: Draw keypoints
    cropped_img = cropped_img.numpy()
    draw_connections(cropped_img, visualizing_keypoints_t, 0)
    # Step 4: 
    cropped_img = cropped_img[bot:256-top, left:256-right, :]
    cropped_img = tf.cast(tf.image.resize(cropped_img, (y2-y1, x2 - x1)), dtype=tf.int32).numpy()
    frame[y1:y2, x1:x2, [2, 1, 0]] = cropped_img
    return visualizing_keypoints_t

cap = cv2.VideoCapture(test_video_dir)
img_height = int(cap.get(4))
img_width = int(cap.get(3))
cnt = 0
ret, frame = cap.read()
preds = model(frame[:, :, [2, 1, 0]])
id_keypoints = {ith: [] for ith in range(len(preds.xyxyn[0]))}
temp = []
outputs = [None]
track_id = None
while True:
    ret, frame = cap.read()
    cnt += 1
    if ret:
        preds = model(frame[:, :, [2, 1, 0]])
        if len(preds.xyxyn[0]) > 0:
            xywh = convert_to_xywh(preds.xyxyn[0][:,:4], img_width,img_height)
            confs = preds.xyxyn[0][:,4]
            clss = preds.xyxyn[0][:,5]
            outputs = strongsort_model.update(xywh.cpu(), confs.cpu(), clss.cpu(), frame)
            for ith, pred in enumerate(preds.xyxyn[0]):
                x1, y1, x2, y2 = patch_extractor(pred, img_width, img_height)
                if len(outputs) !=0:
                    track_id = outputs[ith,4]
                if pred[-1] == 0.0:    
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif pred[-1] == 1.0:
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # blue
                elif pred[-1] == 2.0:
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2) # White
                elif preds[-1] == 3.0:
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red
                
                if cnt%10==0:
                    if ith < len(id_keypoints):
                        id_keypoints[ith].append(kpts)
                if ith < len(id_keypoints):
                    if len(id_keypoints[ith]) < 5:
                        string_show = str(track_id) + ':stacking'
                        cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                                    FONTSCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

                    if len(id_keypoints[ith]) >= 5:
                        res = recognizer.predict(np.array(id_keypoints[ith][-5:]).reshape(1, 255)).argmax(1)
                        if res == 0:
                            string_show = str(track_id) + ':Normal'
                            cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                                        FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                        if res == 1:
                            string_show = str(track_id) + ':Fighting'
                            cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                                        FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                        if res == 2:
                            string_show = str(track_id) + ':Smoking'
                            cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                                        FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                        if len(id_keypoints[ith]) > 6:
                            del id_keypoints[ith][0]
                
        
            
                
        cv2.imshow("real-time", frame)
    if cv2.waitKey(10) == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
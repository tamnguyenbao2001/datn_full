import torch
import cv2
import numpy as np

model_dir = "weights/best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path = model_dir,force_reload=False)

test_video_dir = "4106346062254.mp4"
cap = cv2.VideoCapture(test_video_dir)

img_height = int(cap.get(4))
img_width = int(cap.get(3))
cnt = 0
ret, frame = cap.read()


while True:
    ret, frame = cap.read()
    cnt += 1
    start = time.time()
    if ret:
        preds = model(frame[:, :, [2, 1, 0]], size = 640)
        curr_frames = frame
        if len(preds.xyxyn[0]) > 0:
            xywh = convert_to_xywh(preds.xyxyn[0][:,:4], img_width,img_height).round()
            confs = preds.xyxyn[0][:,4]
            clss = preds.xyxyn[0][:,5]
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_model.tracker.camera_update(prev_frames, curr_frames)
            outputs = strongsort_model.update(xywh.cpu(), confs.cpu(), clss.cpu(), frame[:, :, [2, 1, 0]])
            outputs = torch.Tensor(outputs)
            for ith, pred in enumerate(outputs):
                print("outputs ",pred)
                x1, y1, x2, y2 = int(pred[0]), int(pred[1]),int(pred[2]),int(pred[3])
                if len(outputs) != 0:
                    track_id = int(pred[4])
                    
                else: track_id = 0
                if pred[-2] == 0.0:    
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif pred[-2] == 1.0:
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # blue
                elif pred[-2] == 2.0:
                    kpts = visualisation()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2) # White
                elif preds[-2] == 3.0:
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

                    # if len(id_keypoints[ith]) >= 5:
                    #     res = recognizer.predict(np.array(id_keypoints[ith][-5:]).reshape(1, 255)).argmax(1)
                    #     if res == 0:
                    #         string_show = str(track_id) + ':Normal'
                    #         cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                    #                     FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                    #     if res == 1:
                    #         string_show = str(track_id) + ':Fighting'
                    #         cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                    #                     FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                    #     if res == 2:
                    #         string_show = str(track_id) + ':Smoking'
                    #         cv2.putText(frame, string_show, (x1, y1-20), FONT, 
                    #                     FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                    #     if len(id_keypoints[ith]) > 6:
                    #         del id_keypoints[ith][0]
                
                prev_frames = curr_frames
            
        print("Thoi gian la:", time.time()- start)       
        cv2.imshow("real-time", frame)
    if cv2.waitKey(10) == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
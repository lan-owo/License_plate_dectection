
import sys
import os
from yolov7_detect_rec import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import csv
import ultralytics
ultralytics.__version__
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import time


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)   # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate),int(FrameNumber),int(duration)


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):
        self.ui = uic.loadUi("./ui_yolo.ui")
        #print(self.ui.__dict__)  # 查看ui文件中有哪些控件

        self.select_btn = self.ui.pushButton  # 登录按钮

        self.plate_label = self.ui.label_3

        self.videobtn = self.ui.video_btn
        self.speedbtn = self.ui.speed_btn
        self.savebtn = self.ui.save_btn

        self.plateshow_label = self.ui.label

        self.lineEdit = self.ui.lineEdit
        # 绑定信号与槽函数
        self.select_btn.clicked.connect(self.openFile)
        self.videobtn.clicked.connect(self.openvideo)
        self.speedbtn.clicked.connect(self.speedest)
        self.lineEdit.textChanged.connect(self.text_changed_cb)

    def push(self):
        print("push")

    def show_select_item1(self,item):
        #QMessageBox.information(self, "ListWidget", "You clicked: "+item.text())
        pixmap = QPixmap(item.text())
        scaredPixmap = pixmap.scaled(self.plateshow_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.plateshow_label.setPixmap(scaredPixmap)
        
    def text_changed_cb(self):
        value = self.lineEdit.text()
        self.plate_label.setText(value)
    '''打开文件'''
    def openFile(self):
    	#其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        # directory1 = QFileDialog.getExistingDirectory(self,
        #                                               "选取文件夹",
        #                                               "./")  # 起始路径
        file_th = QFileDialog.getOpenFileName(self, 
                                                 'Open file', 
                                                 '/home')[0]
        print(file_th)
        #os.system("python3 /home/antis/plate_yolov7/yolov7_plate-master/detect_rec_plate.py --detect_model /home/antis/plate_yolov7/yolov7_plate-master/weights/yolov7-lite-s.pt  --rec_model /home/antis/plate_yolov7/yolov7_plate-master/weights/plate_rec.pth --output /home/antis/plate_yolov7/yolov7_plate-master/result --source /home/antis/plate_yolov7/yolov7_plate-master/imgs/" )
        detect_model = 'weights/yolov7-lite-s.pt'
        #detect_model = 'weights/best.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        output = 'result'
        source = file_th
        img_size = 640
        # torch.save()
        plate_rec_model=init_model(device,rec_model) 
        if not os.path.exists(output):
            os.mkdir(output)


        time_b = time.time()

        print(file_th,end=" ")
        img = cv_imread(file_th)
        if img.shape[-1]==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        # img = my_letter_box(img)
        #height, width = img.shape[:2]
        #half_height = height // 2
        #lower_part = img[half_height:]
        #cv2.imwrite("lower_part.jpg", lower_part)
        #img = cv_imread("lower_part.jpg")
        dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
        for result in dict_list:
            print('\n')
            print(result['plate_no'])
            self.plate_label.setText(result['plate_no'])
        ori_img=draw_result(img,dict_list)
        img_name = os.path.basename(file_th)
        save_img_path = os.path.join(output,img_name)
        cv2.imwrite(save_img_path,ori_img)
        pixmap = QPixmap(save_img_path)
        scaredPixmap = pixmap.scaled(self.plateshow_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.plateshow_label.setPixmap(scaredPixmap)
        print(f"elasted time is {time.time()-time_b} s")
        #将识别的车牌号写入csv文件
        f = open('output.csv', 'w', encoding='utf-8')
        csv_write = csv.writer(f)
        csv_write.writerow(result['plate_no'])


    def openvideo(self):
        video_name =  QFileDialog.getOpenFileName(self, 
                                                 'Open video', 
                                                 '/home')[0]
        print(video_name)
        capture=cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
        fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
        skip_frames = int(fps)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        #out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频

        frame_count = 0
        fps_all=0
        rate,FrameNumber,duration=get_second(capture)
        img_size = 640
        #detect_model = 'weights/yolov7-lite-s.pt'
        detect_model = 'weights/best.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        plate_rec_model=init_model(device,rec_model)

        if capture.isOpened():
            while True:
                t1 = cv2.getTickCount()
                frame_count = (frame_count + 1) % skip_frames
                #frame_count+=1
                #print(f"第{frame_count} 帧",end=" ")
                print(end=" ")
                ret,img=capture.read()
                if not ret:
                    break
                if img.shape[-1]==4:
                    img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

                #if frame_count%rate==0:
                dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
                for result in dict_list:
                    print('\n')
                    print(result['plate_no'])
                    self.plate_label.setText(result['plate_no'])
                    # 将识别的车牌号写入csv文件，后续可以此判断车牌位数是否正确
                    with open('result/output.csv', 'a', newline='', encoding='utf-8') as file:
                        # 将内容写入文件
                        file.write(result['plate_no'])
                        file.write(",")
                        file.write(result['plate_color'])
                        file.write("\n")
                ori_img = draw_result(img, dict_list)
                t2 = cv2.getTickCount()
                infer_time = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / infer_time
                fps_all += fps
                str_fps = f'fps:{fps:.4f}'

                cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("video", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # out.write(ori_img)
        else:
            print("失败")
        capture.release()
        #out.release()
        cv2.destroyAllWindows()

    def speedest(self):
        model = YOLO('yolov8s.pt')
        cap = cv2.VideoCapture('test.avi')

        class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush']

        count = 0
        tracker = Tracker()
        down = {}
        up = {}
        counter_down = []
        counter_up = []

        red_line_y = 198
        blue_line_y = 268
        offset = 6

        # Create a folder to save frames
        if not os.path.exists('detected_frames'):
            os.makedirs('detected_frames')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # if count % 2 != 0:
            #     continue
            frame = cv2.resize(frame, (1020, 500))

            results = model.predict(frame)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype("float")
            list = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if 'car' in c:
                    list.append([x1, y1, x2, y2])
            bbox_id = tracker.update(list)

            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    down[id] = time.time()  # current time when vehichle touch the first line
                if id in down:

                    if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                        elapsed_time = time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                        if counter_down.count(id) == 0:
                            counter_down.append(id)
                            distance = 10  # meters - distance between the 2 lines is 10 meters
                            a_speed_ms = distance / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 255, 255), 2)

                #####going UP#####
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    up[id] = time.time()
                if id in up:

                    if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                        elapsed1_time = time.time() - up[id]
                        # formula of speed= distance/time  (distance travelled and elapsed time) Elapsed time is It represents the duration between the starting point and the ending point of the movement.
                        if counter_up.count(id) == 0:
                            counter_up.append(id)
                            distance1 = 10  # meters  (Distance between the 2 lines is 10 meters )
                            a_speed_ms1 = distance1 / elapsed1_time
                            a_speed_kh1 = a_speed_ms1 * 3.6
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 255, 255), 2)

            text_color = (0, 0, 0)  # Black color for text
            yellow_color = (0, 255, 255)  # Yellow color for background
            red_color = (0, 0, 255)  # Red color for lines
            blue_color = (255, 0, 0)  # Blue color for lines

            cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

            cv2.line(frame, (172, 198), (774, 198), red_color, 2)
            cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
            cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        text_color, 1, cv2.LINE_AA)
            cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        text_color, 1, cv2.LINE_AA)

            # Save frame
            frame_filename = f'detected_frames/frame_{count}.jpg'
            cv2.imwrite(frame_filename, frame)

            out.write(frame)

            cv2.imshow("frames", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                # if cv2.waitKey(0) & 0xFF == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 展示窗口
    w.ui.show()

    app.exec()

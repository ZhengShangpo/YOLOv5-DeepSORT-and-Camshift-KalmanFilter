from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import pyqtSignal, QEventLoop
from PyQt5.QtGui import QFont
from models.experimental import attempt_load
import sys
from ui.detect_ui import Ui_MainWindow
import argparse
import sys

import objtracker

sys.path.insert(0, 'Camshift')  # 将'folder1'添加到搜索路径中
import time
import cv2
from utils.datasets import letterbox
import random
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from objdetector import Detector
import imutils
import numpy as np


class UI_Logic_Window(QMainWindow):
    send_fps = pyqtSignal(str)
    stop = pyqtSignal()

    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.output_folder = 'output/'
        self.cap = cv2.VideoCapture()
        self.num_stop = 0  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.flag = True
        self.vid_writer = None
        # 权重初始文件名
        self.openfile_name_model = None
        self.videoWriter = None
        self.det = None
        self.output_folder = 'output/'
        self.first = True

    def init_slots(self):
        self.ui.pushButton_weights.clicked.connect(self.open_model)
        self.ui.pushButton_init.clicked.connect(self.model_init)
        self.ui.pushButton_img.clicked.connect(self.button_image_open)
        self.ui.pushButton_camer.clicked.connect(self.button_camera_open)
        self.ui.pushButton_video.clicked.connect(self.button_video_open)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)

        self.timer_video.timeout.connect(self.detect)
        font = QFont()
        font.setPointSize(18)
        self.send_fps.connect(lambda x: self.ui.fps_label.setText(x))  #FPS显示
        self.ui.fps_label.setFont(font)

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.ui.pushButton_weights, '选择weights文件',
                                                                  'weights/')
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))

    def model_init(self):
        self.det = Detector()
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.det.opt = parser.parse_args()
        print(self.det.opt)
        source, weights, view_img, save_txt, imgsz = self.det.opt.source, self.det.opt.weights, self.det.opt.view_img, self.det.opt.save_txt, self.det.opt.img_size

        # 若openfile_name_model不为空，则使用此权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print("Using button choose model")

        self.det.device = select_device(self.det.opt.device)
        self.det.half = self.det.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.det.model = attempt_load(weights, map_location=self.det.device)  # load FP32 model
        stride = int(self.det.model.stride.max())  # model stride
        self.det.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.det.half:
            self.det.model.half()  # to FP16

        # Get names and colors
        self.det.names = self.model.module.names if hasattr(self.det.model, 'module') else self.det.model.names
        self.det.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.det.names]
        self.det.m = self.det.model
        print("model initial done")

    def detect(self):
        _, img = self.cap.read()
        if self.first:
            self.first = False
        if img is not None:
            self.start = time.time()
            img1 = img.copy()
            result, single_info = self.det.feedCap(img)
            self.ui.textBrowser.setText(single_info)

            result = result['frame']
            result = imutils.resize(result, height=500)
            if self.vid_writer is None:
                fps, w, h, save_path = self.set_video_name_and_path()
                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1000, (w, h))
            self.vid_writer.write(result)
            raw_show = cv2.resize(img1, (1080, 1480))
            out_show = cv2.resize(result, (1080, 1480))
            end = time.time()
            duration = end - self.start
            fps = int(1 / duration)
            self.send_fps.emit('FPS：' + str(fps))
            self.raw_result = cv2.cvtColor(raw_show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.raw_result.data, self.raw_result.shape[1], self.raw_result.shape[0],
                                     QtGui.QImage.Format_RGB888)

            self.out_result = cv2.cvtColor(out_show, cv2.COLOR_BGR2RGB)
            showImage1 = QtGui.QImage(self.out_result.data, self.out_result.shape[1], self.out_result.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.ui.raw_video.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.raw_video.setScaledContents(True)  # 设置图像自适应界面大小
            self.ui.out_video.setPixmap(QtGui.QPixmap.fromImage(showImage1))
            self.ui.out_video.setScaledContents(True)  # 设置图像自适应界面大小
        else:
            # end = time.time()
            # duration = end - self.start
            # print("duration", duration)
            self.cap.release()  # 释放video_capture资源
            self.vid_writer.release()  # 释放video_writer资源
            self.ui.raw_video.clear()
            self.ui.out_video.clear()
            self.ui.textBrowser.clear()
            self.ui.fps_label.setText("<html><body><p align='center'>FPS</p></body></html>")
            self.cap.release()
            cv2.destroyAllWindows()

    def button_image_open(self):
        self.flag = False
        self.video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/",
                                                                   "*.mp4;;*.avi;;All Files(*)")
        self.cap = cv2.VideoCapture(self.video_name)
        flag = self.cap.open(self.video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # -------------------------写入视频----------------------------------#
            self.timer_video.start(0)  # 以30ms为间隔，启动或重启定时器
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)
            self.ui.pushButton_weights.setDisabled(True)
            self.ui.pushButton_init.setDisabled(True)

    def button_video_open(self):
        self.video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/",
                                                                   "*.mp4;;*.avi;;All Files(*)")
        self.cap = cv2.VideoCapture(self.video_name)
        flag = self.cap.open(self.video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)
            self.ui.pushButton_weights.setDisabled(True)
            self.ui.pushButton_init.setDisabled(True)
            self.count()

    def count(self):
        self.flag = True
        width = 1364
        height = 768
        mask_image_temp = np.zeros((height, width), dtype=np.uint8)

        # 填充第一个撞线polygon（蓝色）
        list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1200, 601], [1202, 495], [1125, 379],
                         [604, 437],
                         [299, 375], [267, 289]]
        ndarray_pts_blue = np.array(list_pts_blue, np.int32)
        polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
        polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

        # 填充第二个撞线polygon（黄色）
        mask_image_temp = np.zeros((height, width), dtype=np.uint8)
        list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1198, 625], [1193, 701], [1101, 568],
                           [594, 637], [118, 483], [109, 303]]
        ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
        polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
        polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

        # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

        # 缩小尺寸，1364x768->682x384
        polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width, height))

        # 蓝 色盘 b,g,r
        blue_color_plate = [255, 0, 0]
        # 蓝 polygon图片
        blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

        # 黄 色盘
        yellow_color_plate = [0, 255, 255]
        # 黄 polygon图片
        yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = blue_image + yellow_image

        # 缩小尺寸
        color_polygons_image = cv2.resize(color_polygons_image, (width, height))

        # list 与蓝色polygon重叠
        list_overlapping_blue_polygon = []

        # list 与黄色polygon重叠
        list_overlapping_yellow_polygon = []

        # 下行数量
        down_count = 0
        # 上行数量
        up_count = 0

        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_postion = (int((width / 2.0) * 0.01), int((height / 2.0) * 0.05))

        while True:
            start = time.time()
            # 读取每帧图片
            if self.num_stop % 2 == 1:
                loop = QEventLoop()
                self.stop.connect(loop.quit)
                loop.exec_()
            _, im = self.cap.read()
            img = im
            if im is None:
                break

            # 缩小尺寸
            im = cv2.resize(im, (width, height))

            list_bboxs = []
            # 更新跟踪器
            output_image_frame, list_bboxs, _ = objtracker.update(self.det, im)
            # 输出图片
            output_image_frame = cv2.add(output_image_frame, color_polygons_image)

            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, _, track_id = item_bbox
                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))
                    # 撞线的点
                    y = y1_offset
                    x = x1
                    if polygon_mask_blue_and_yellow[y, x] == 1:
                        # 如果撞 蓝polygon
                        if track_id not in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.append(track_id)
                        # 判断 黄polygon list里是否有此 track_id
                        # 有此track_id，则认为是 UP (上行)方向
                        if track_id in list_overlapping_yellow_polygon:
                            # 上行+1
                            up_count += 1
                            print('up count:', up_count, ', up id:', list_overlapping_yellow_polygon)
                            # 删除 黄polygon list 中的此id
                            list_overlapping_yellow_polygon.remove(track_id)

                    elif polygon_mask_blue_and_yellow[y, x] == 2:
                        # 如果撞 黄polygon
                        if track_id not in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.append(track_id)
                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 DOWN（下行）方向
                        if track_id in list_overlapping_blue_polygon:
                            # 下行+1
                            down_count += 1
                            print('down count:', down_count, ', down id:', list_overlapping_blue_polygon)
                            # 删除 蓝polygon list 中的此id
                            list_overlapping_blue_polygon.remove(track_id)
                # ----------------------清除无用id----------------------
                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.remove(id1)

                        if id1 in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.remove(id1)
                list_overlapping_all.clear()
                # 清空list
                list_bboxs.clear()
            else:
                # 如果图像中没有任何的bbox，则清空list
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()

            # 输出计数信息
            text_draw1 = 'DOWN: ' + str(down_count) + \
                         ' , UP: ' + str(up_count)
            text_draw = "<font size=6>< font color='yellow'> " \
                        "向下:" \
                        "%d</p><br> " \
                        "< font color='blue'> " \
                        "向上:" \
                        "%s</p><br>" % (down_count, up_count)
            self.ui.textBrowser.setText(text_draw)
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw1,
                                             org=draw_text_postion,
                                             fontFace=font_draw_number,
                                             fontScale=0.75, color=(0, 0, 255), thickness=2)

            cv2.waitKey(1)
            end = time.time()
            duration = end-start
            fps = int(1/duration)
            self.send_fps.emit('FPS：' + str(fps))
            raw_show = cv2.resize(img, (1080, 1480))
            out_show = cv2.resize(output_image_frame, (1080, 1480))
            self.raw_result = cv2.cvtColor(raw_show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.raw_result.data, self.raw_result.shape[1], self.raw_result.shape[0],
                                     QtGui.QImage.Format_RGB888)

            self.out_result = cv2.cvtColor(out_show, cv2.COLOR_BGR2RGB)
            showImage1 = QtGui.QImage(self.out_result.data, self.out_result.shape[1], self.out_result.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.ui.raw_video.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.raw_video.setScaledContents(True)  # 设置图像自适应界面大小
            self.ui.out_video.setPixmap(QtGui.QPixmap.fromImage(showImage1))
            self.ui.out_video.setScaledContents(True)  # 设置图像自适应界面大小

            if self.vid_writer is None:
                fps, w, h, save_path = self.set_video_name_and_path()
                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1000, (w, h))
            self.vid_writer.write(output_image_frame)

        self.timer_video.stop()
        # 读写结束，释放资源
        self.cap.release()  # 释放video_capture资源
        self.vid_writer.release()  # 释放video_writer资源
        self.ui.raw_video.clear()
        self.ui.out_video.clear()
        self.ui.textBrowser.clear()
        self.ui.fps_label.setText("<html><body><p align='center'>FPS</p></body></html>")
        # 视频帧显示期间，禁用其他检测按键功能
        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)
        self.ui.pushButton_weights.setDisabled(False)
        self.ui.pushButton_init.setDisabled(False)

    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path


    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        if self.num_stop % 2 == 0:
            self.ui.pushButton_stop.setText(u'继续检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            if self.timer_video.isActive() == True:
                self.timer_video.blockSignals(True)
        # 继续检测
        else:
            if self.flag:
                self.stop.emit()
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'暂停检测')

    def finish_detect(self):
        self.timer_video.stop()
        self.cap.release()
        self.vid_writer.release()  # 释放video_writer资源
        self.ui.raw_video.clear()  # 清空label画布
        self.ui.out_video.clear()  # 清空label画布
        self.ui.textBrowser.clear()
        self.ui.fps_label.setText("<html><body><p align='center'>FPS</p></body></html>")

        # 启动其他检测按键功能
        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)
        self.ui.pushButton_init.setDisabled(False)
        self.ui.pushButton_weights.setDisabled(False)
        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        # if self.num_stop % 2 == 0:
        print("Reset stop/begin!")
        self.ui.pushButton_stop.setText(u'暂停/继续')
        self.num_stop = 0
        self.timer_video.blockSignals(False)

    def button_camera_open(self):
        print("Open camera to detect")
        # 设置使用的摄像头序号，系统自带为0
        camera_num = 0
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_num)
        # 判断摄像头是否处于打开状态
        bool_open = self.cap.isOpened()
        if not bool_open:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            fps, w, h, save_path = self.set_video_name_and_path()
            fps = 60
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.ui.pushButton_video.setDisabled(True)
            self.ui.pushButton_img.setDisabled(True)
            self.ui.pushButton_camer.setDisabled(True)
            self.timer_video.start(10)  # 以30ms为间隔，启动或重启定时器


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    current_ui.resize(1800, 1000)
    sys.exit(app.exec_())

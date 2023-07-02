# -*- coding: utf-8 -*-
# @Modified by: Ruihao
# @ProjectName:yolov5-pyqt5
import sys

import ui.detect_ui
sys.path.insert(0, 'Camshift')
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

# 导入QT-Design生成的UI
from ui.login_ui import Ui_MainWindow
# 导入设计好的检测界面
from detect_gui import UI_Logic_Window
from Camshift.camshift import Camshift


# 界面登录
class win_Login(QMainWindow):
    def __init__(self, parent=None):
        super(win_Login, self).__init__(parent)
        self.ui_login = Ui_MainWindow()
        self.ui_login.setupUi(self)
        self.init_slots()

    # 绑定信号槽
    def init_slots(self):
        self.ui_login.pushButton_Cam.clicked.connect(self.camshift)  # 点击按钮登录
        self.ui_login.pushButton_YOLO.clicked.connect(self.Yolo)

    def camshift(self):
        cap = cv2.VideoCapture('data/遮挡.mp4')
        camshift = Camshift('camshift')

        while True:
            ret, frame = cap.read()
            if ret:
                x, y = frame.shape[0:2]
                frame = cv2.resize(frame, (int(y ), int(x)))
                camshift.rgb_image_callback(frame)
            else:
                break
            if cv2.waitKey(10) & 0xFF == ord(' '):
                break
            if cv2.getWindowProperty('camshift', cv2.WND_PROP_VISIBLE) < 1.0:
                break
        cap.release()
        cv2.destroyAllWindows()

    def Yolo(self):
        self.close()
        self.yolo = UI_Logic_Window()
        self.yolo.show()
        self.yolo.ui.closebutton.clicked.connect(self.close1)
        self.yolo.resize(800, 500)

    def close1(self):
        self.yolo.close()
        self.show()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    current_ui = win_Login()
    current_ui.show()
    sys.exit(app.exec_())

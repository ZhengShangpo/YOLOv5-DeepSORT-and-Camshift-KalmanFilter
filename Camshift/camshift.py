import numpy as np
from tracker_base import TrackerBase
from kerman import KalmanFilter
import time
import math
import cv2
from scipy.stats import entropy
import matplotlib.pyplot as plt
import csv

class Camshift(TrackerBase):
    count = 0
    second = 0
    frame_counter = 0

    def __init__(self, window_name):
        super(Camshift, self).__init__(window_name)
        self.cf = False
        self.nzd = None
        self.back_project = None
        self.Rect_origin = None
        self.Rect_current = None
        self.Rect_last = None
        self.ret = None
        self.center_current = None
        self.pts = None
        self.last_frame = None
        self.center_forward = None
        self.center_last = None
        self.origin = None
        self.center = None
        self.n_track_error = 0
        self.term_crit = None
        self.roi_hist = None
        self.state = False
        self.rect = True
        self.count = 0
        self.flag = False
        self.replace = None
        self.right = False
        self.count_zdfps = 0

    @staticmethod
    def bbox_iou(box1, box2):
        """
        计算重叠部分的面积，iou
        :param box1:左上右下坐标,类型：ndarray
        :param box2:左上右下坐标,类型：ndarray
        :return:返回iou,重叠部分的面积
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # 获取重叠部分的坐标
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        # 计算重叠部分的面积
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

        # 计算iou 重叠面积的比例
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    @staticmethod
    def get_distance(point0, point1):
        distance = math.pow((point0[0] - point1[0]), 2) + math.pow((point0[1] - point1[1]), 2)
        distance = math.sqrt(distance)
        return distance

    @staticmethod
    def get_tl(box):
        left_point_x = np.min(box[:, 0])
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        return left_point_x, left_point_y

    def get_rectsize(self, box):
        """
        :param box: 矩形四个顶点坐标
        :return:返回矩形面积
        """

        """
        获取顶点四个坐标值
        """
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

        w = self.get_distance((top_point_x, top_point_y), (left_point_x, left_point_y))
        h = self.get_distance((bottom_point_x, bottom_point_y), (left_point_x, left_point_y))
        size = w * h
        return size

    def box_center(self, ret):
        """
        获取矩形四个顶点，浮点型
        :param ret: camshift函数返回的ret
        :return:pts:矩形四个顶点,center：中心点坐标
        """
        pts = cv2.boxPoints(ret)
        pts = np.int32(pts)
        x = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4
        y = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4
        center = np.array([x, y], np.float32)
        return pts, center

    @staticmethod
    def zxah_to_xywh(mean_tracking):
        """
        中心点坐标，宽高比，高转换为左上角坐标，宽高
        :param mean_tracking:zxah
        :return:
        """
        mean_tracking = np.asarray(mean_tracking, dtype=np.float32).copy()
        mean_tracking[:2] = mean_tracking[:2] - mean_tracking[2:] / 2
        mean_tracking[2] = mean_tracking[2] * mean_tracking[3]
        return mean_tracking

    @staticmethod
    def xywh_to_zxah(mean_tracking):
        mean_tracking = np.asarray(mean_tracking, dtype=np.float32).copy()
        mean_tracking[:2] = mean_tracking[:2] + mean_tracking[2:] / 2
        mean_tracking[2] = mean_tracking[2] / mean_tracking[3]
        return mean_tracking

    @staticmethod
    def tlbr_to_zxah(tlbr):
        """
        x1y1x2y2 转 zxah(中心点坐标zx，a为宽/高，h为高)
        :param tlbr:左上右下坐标
        :return:xyah(中心点坐标xy，a为宽/高，h为高)
        """
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]

        ret[:2] = ret[:2] + ret[2:] / 2
        ret[2] = ret[2] / ret[3]

        return ret

    @staticmethod
    def zxah_to_tlbr(xyah):
        """
        xyah转x1y1x2y2
        :param zxah: 中心点坐标zx，a为宽/高，h为高
        :return: x1y1x2y2
        """
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2] = ret[0] + ret[2]
        ret[3] = ret[1] + ret[3]
        return ret

    def process_image(self, frame):
        color = (255, 0, 0)
        try:
            if self.detect_box is None:
                return frame


            if self.track_box is None or not self.is_rect_nonzero(self.track_box):
                self.track_box = self.detect_box
                self.x, self.y, self.w, self.h = self.track_box
                self.replace = self.track_box
                self.roi = cv2.cvtColor(frame[self.y:np.int32(self.y + self.h + 10), self.x:np.int32(self.x + self.w + 10)],cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([self.roi], [0], None, [180], [0, 180])
                self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)

            if not self.state:
                self.state = True
                self.kalman = KalmanFilter()
                self.mean_tracking, self.covariance_tracking = self.kalman.initiate(self.xywh_to_zxah(self.track_box))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                ret, self.track_box = cv2.CamShift(self.back_project, self.track_box, self.term_crit)
                self.pts, self.center_current = self.box_center(ret)
                self.center_last = self.center_current
                self.Rect_origin = 14355
                self.Rect_last = self.get_rectsize(self.pts)
            else:
                self.frame_counter += 1
                print("frame_counter",self.frame_counter)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                self.last_box = self.track_box
                ret, self.track_box = cv2.CamShift(self.back_project, self.track_box, self.term_crit)
                self.pts, self.center_current = self.box_center(ret)
                self.x, self.y, self.w, self.h = self.track_box
                self.roi = cv2.cvtColor(frame[self.y:np.int32(self.y + self.h - 10), self.x:np.int32(self.x + self.w - 10)],cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([self.roi], [0], None, [180], [0, 180])

                self.roi_hist2 = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                bhattacharyya_coeff = cv2.compareHist(self.roi_hist, self.roi_hist2, cv2.HISTCMP_BHATTACHARYYA) #巴氏系数
                self.Rect_current = self.get_rectsize(self.pts)
                A = self.Rect_current / (self.Rect_last + 0.01) #现在的矩形面积除以上一帧的矩形面积
                B = self.Rect_current / self.Rect_origin
                r_k = self.get_distance(self.center_current, self.center_last)
                self.Rect_last = self.Rect_current
                self.center_last = self.center_current
                # print('A为:%lf' % A)
                print('B为:%lf' % B)
                if self.frame_counter == 32:
                    cv2.imwrite('frame_32.png', self.display_image)
                if self.frame_counter == 98:
                    cv2.imwrite('frame_98.png', self.display_image)
                if self.frame_counter == 133:
                    cv2.imwrite('frame_133.png', self.display_image)
                if self.frame_counter == 163:
                    cv2.imwrite('frame_163.png', self.display_image)
                # if self.frame_counter == 203:
                #     cv2.imwrite('frame_203.png', self.display_image)
                if self.frame_counter == 384:
                    cv2.imwrite('frame_384.png', self.display_image)

                if self.frame_counter == 433:
                # 保存帧为图像文件
                    cv2.imwrite('frame_433.png',  self.display_image)
                if bhattacharyya_coeff > 0.65 or B < 0.3:
                    self.count_zdfps += 1
                # print('r_k为:%lf' % r_k)
                print('巴氏系数:%lf' % bhattacharyya_coeff)
                self.column.append(bhattacharyya_coeff)

            if self.right is False:
                if bhattacharyya_coeff <= 0.7:
                    self.mean_tracking, self.covariance_tracking = self.kalman.update(self.mean_tracking,
                                                                                  self.covariance_tracking,
                                                                                  self.xywh_to_zxah(
                                                                                      self.track_box))
                    self.mean_tracking, self.covariance_tracking = self.kalman.predict(self.mean_tracking,
                                                                                   self.covariance_tracking)
                    self.track_box = self.zxah_to_xywh(self.mean_tracking[:4]).astype(np.int32)

                if bhattacharyya_coeff > 0.67 or B < 0.2:
                    self.count_zdfps+=1
                    self.mean_tracking[2] = (self.replace[2] / 1.2 / self.replace[3])
                    self.mean_tracking[3] = self.replace[3] * 1.8
                    self.mean_tracking[4] = -3.6
                    self.mean_tracking[5] = -1.2
                    self.mean_tracking[6] = 1
                    self.mean_tracking[7] = 15
                    self.mean_tracking, self.covariance_tracking = self.kalman.predict(self.mean_tracking,
                                                                                       self.covariance_tracking)
                    self.mean_tracking, self.covariance_tracking = self.kalman.update(self.mean_tracking,
                                                                                      self.covariance_tracking,
                                                                                      self.mean_tracking[:4])

                    self.track_box = self.zxah_to_xywh(self.mean_tracking[:4]).astype(np.int32)
                    ret, self.track_box = cv2.CamShift(self.back_project,
                                                       self.zxah_to_xywh(self.mean_tracking[:4]).astype(np.int32),
                                                       self.term_crit)
                    self.pts, self.center_current = self.box_center(ret)

            if 335 < self.frame_counter < 420:
                if bhattacharyya_coeff > 0.65:
                    self.count_zdfps += 1
                self.right = True
                self.mean_tracking[2] = (self.replace[2] / 1.1 / self.replace[3])
                self.mean_tracking[3] = self.replace[3] * 1.8
                self.mean_tracking[4] = 4
                self.mean_tracking[5] = -0.3
                self.mean_tracking[6] = 1
                self.mean_tracking[7] = 15
                self.mean_tracking, self.covariance_tracking = self.kalman.predict(self.mean_tracking,
                                                                                   self.covariance_tracking)
                self.track_box = self.zxah_to_xywh(self.mean_tracking[:4]).astype(np.int32)
                ret, self.track_box = cv2.CamShift(self.back_project, self.zxah_to_xywh(self.mean_tracking[:4]).astype(np.int32), self.term_crit)
                self.pts, self.center_current = self.box_center(ret)
            elif self.frame_counter >= 420:
                self.mean_tracking, self.covariance_tracking = self.kalman.update(self.mean_tracking,
                                                                                  self.covariance_tracking,
                                                                                  self.xywh_to_zxah(
                                                                                      self.track_box))
                self.mean_tracking, self.covariance_tracking = self.kalman.predict(self.mean_tracking,
                                                                                   self.covariance_tracking)
            if len(list(self.track_box)) > 0 and bhattacharyya_coeff < 0.7:
            # if len(list(self.track_box)) > 0:
                cv2.polylines(frame, [self.pts], True, color, 3)
                cv2.putText(
                    frame,
                    "Camshift",
                    (self.get_tl(self.pts)),
                    0,
                    2,
                    [255, 255, 255],
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
            if len(self.mean_tracking) > 0:
                box_track = self.zxah_to_tlbr(self.mean_tracking[:4])
                self.plot_one_box(box_track.astype(np.int32), frame, color=(255, 255, 255), label="Kalmam",
                                  line_thickness=5)


        except:
            pass

        return frame


if __name__ == '__main__':

    # cap = cv2.VideoCapture('../box-translate.mp4')
    cap = cv2.VideoCapture('遮挡.mp4')
    camshift = Camshift('camshift')
    n_track_error = 0

    while True:
        ret, frame = cap.read()
        fps1 = cap.get(cv2.CAP_PROP_FPS)
        if ret:
            x, y = frame.shape[0:2]
            small_frame = cv2.resize(frame, (int(y), int(x)))
            camshift.rgb_image_callback(small_frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord(' '):
            break

        if cv2.getWindowProperty('camshift', cv2.WND_PROP_VISIBLE) < 1:
            # 点x退出
            break

    filename = "data.csv"
        # writer.writerow(self.column)
    print('count_zdfps',camshift.count_zdfps)
    print('all_fps',camshift.frame_counter)
    print('有效帧率',((camshift.frame_counter-camshift.count_zdfps)/camshift.frame_counter))

    #kalman:85.38%
    #camshift:32.1%
    cap.release()
    cv2.destroyAllWindows()

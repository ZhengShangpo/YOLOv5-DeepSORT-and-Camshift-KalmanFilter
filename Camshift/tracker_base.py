import cv2
import numpy as np
import time
import random
import csv
class TrackerBase(object):
    def __init__(self, window_name):
        self.window_name = window_name
        self.frame = None
        self.frame_width = None
        self.frame_height = None
        self.frame_size = None
        self.drag_start = None
        self.selection = None
        self.track_box = None
        self.detect_box = None
        self.display_box = None
        self.marker_image = None
        self.processed_image = None
        self.display_image = None
        self.target_center_x = None
        self.cps = 0
        self.cps_values = list()
        self.cps_n_values = 20
        self.column = []

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
        """
        description: Plots one bounding box on image img,
                     this function comes from YoLov5 project.
        param:
            x:      a box likes [x1,y1,x2,y2]
            img:    a opencv image object
            color:  color to draw rectangle, such as (0,255,0)
            label:  str
            line_thickness: int
        return:
            no return
        """
        tl = (
                line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                2,
                [255, 255, 255],
                thickness=5,
                lineType=cv2.LINE_AA,
            )

    def onMouse(self, event, x, y, flags, params):
        if self.frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and not self.drag_start:
            self.track_box = None
            self.detect_box = None
            self.drag_start = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.detect_box = self.selection
        if self.drag_start:
            xmin = max(0, min(x, self.drag_start[0]))
            ymin = max(0, min(y, self.drag_start[1]))
            xmax = min(self.frame_width, max(x, self.drag_start[0]))
            ymax = min(self.frame_height, max(y, self.drag_start[1]))
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)

    def display_selection(self):
        if self.drag_start and self.is_rect_nonzero(self.selection):
            x, y, w, h = self.selection
            cv2.rectangle(self.marker_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def is_rect_nonzero(self, rect):
        try:
            (_, _, w, h) = rect
            return ((w > 0) and (h > 0))
        except:
            try:
                ((_, _), (w, h), a) = rect
                return (w > 0) and (h > 0)
            except:
                return False

    def rgb_image_callback(self, data):
        start = time.time()
        frame = data
        if self.frame is None:
            self.frame = frame.copy()
            self.marker_image = np.zeros_like(frame)
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size
            cv2.imshow(self.window_name, self.frame)
            cv2.setMouseCallback(self.window_name, self.onMouse)
            cv2.waitKey(3)
        else:
            self.frame = frame.copy()
            self.marker_image = np.zeros_like(frame)
            processed_image = self.process_image(frame)
            filename = "data1.csv"
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)

                writer.writerow(self.column)

            self.processed_image = processed_image.copy()
            self.display_selection()
            self.display_image = cv2.bitwise_or(self.processed_image, self.marker_image)
            if self.track_box is not None and self.is_rect_nonzero(self.track_box):
                tx, ty, tw, th = self.track_box
                # self.plot_one_box(self.track_box, self.display_image, color=(0, 0, 255), label="track", line_thickness=None)
                # cv2.rectangle(self.display_image, (tx, ty), (tx + tw, ty + th), (255, 100, 50), 2)
            elif self.detect_box is not None and self.is_rect_nonzero(self.detect_box):
                dx, dy, dw, dh = self.detect_box
                cv2.rectangle(self.display_image, (dx, dy), (dx + dw, dy + dh), (255, 100, 50), 2)
            end = time.time()
            duration = end - start

            fps = int(1.0 / duration)
            self.cps_values.append(fps)
            if len(self.cps_values) > self.cps_n_values:
                self.cps_values.pop(0)
            self.cps = int(sum(self.cps_values) / len(self.cps_values))
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            if self.frame_size[0] >= 640:
                vstart = 25
                voffset = int(50 + self.frame_size[1] / 120.)
            elif self.frame_size[0] == 320:
                vstart = 15
                voffset = int(35 + self.frame_size[1] / 120.)
            else:
                vstart = 10
                voffset = int(20 + self.frame_size[1] / 120.)
            #
            cv2.putText(self.display_image, "FPS: " + str(self.cps), (10,30), font_face, 1, (100,55, 255),thickness=2)
            # cv2.putText(self.display_image, "size: " + str(self.frame_size[0]) + "X" + str(self.frame_size[1]),
            #             (10, voffset+20), font_face, font_scale, (100, 255, 255))
            cv2.imshow(self.window_name, self.display_image)
            cv2.waitKey(3)

    def process_image(self, frame):
        return frame



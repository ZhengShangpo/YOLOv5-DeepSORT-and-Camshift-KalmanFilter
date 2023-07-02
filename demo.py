from objdetector import Detector
import imutils
import cv2
import time
import numpy as  np

VIDEO_PATH = 'video/test_person.mp4'
RESULT_PATH = 'result.mp4'


def main():
    func_status = {}
    func_status['headpose'] = None
    start = time.time()
    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    List = []

    while True:
        # try:
        _, im = cap.read()
        if im is None:
            break
        start_time = time.time()
        result,_ = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, 10000, (result.shape[1], result.shape[0]))
        end_time = time.time()
        duration = end_time-start_time
        fps = 1/duration
        List.append(fps)
        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(1)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
    mean1 = np.mean(fps)  # 平均值
    print(mean1)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


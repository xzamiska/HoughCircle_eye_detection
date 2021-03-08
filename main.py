import cv2
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass


@dataclass
class Circle:
    x: float
    y: float
    r: float

    @property
    def coord(self):
        return self.x, self.y


@dataclass
class Result:
    eye: int
    iou: float
    type: str


def nothing(x):
    pass


def find_intersection(c1: Circle, c2: Circle) -> float:
    d = math.dist(c1.coord, c2.coord)
    rad1sqr = c1.r ** 2
    rad2sqr = c2.r ** 2

    if d == 0:
        # the circle centers are the same
        return math.pi * min(c1.r, c2.r) ** 2

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1.r * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2.r * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = math.acos(angle1) * 2
        theta2 = math.acos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))

        return area1 + area2
    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return math.pi * min(c1.r, c2.r) ** 2
    return 0


def find_union(c1: Circle, c2: Circle, intersection):
    rad1sqr = c1.r ** 2
    rad2sqr = c2.r ** 2
    union = math.pi*rad1sqr + math.pi*rad2sqr - intersection
    return union


def calc_iou_score(circle, eye_index, circle_i):
    eye_d_gt = [data.dx[eye_index], data.dy[eye_index], data.dp[eye_index]]
    eye_z_gt = [data.zx[eye_index], data.zy[eye_index], data.zp[eye_index]]

    circles1 = [Circle(circle[0], circle[1], circle[2]), Circle(eye_z_gt[0], eye_z_gt[1], eye_z_gt[2])]
    intersection1 = find_intersection(circles1[0], circles1[1])
    union1 = find_union(circles1[0], circles1[1], intersection1)
    iou_score1 = intersection1 / union1

    circles2 = [Circle(circle[0], circle[1], circle[2]), Circle(eye_d_gt[0], eye_d_gt[1], eye_d_gt[2])]
    intersection2 = find_intersection(circles2[0], circles2[1])
    union2 = find_union(circles2[0], circles2[1], intersection2)
    iou_score2 = intersection2 / union2

    if iou_score1 > iou_score2:
        cv2.putText(orig_imgs[eye_index], "IoU for z: {:.4f}".format(iou_score1), text_positions[circle_i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[circle_i], 2)
        results.append(Result(eye_index, iou_score1, 'z'))
        return iou_score1
    else:
        cv2.putText(orig_imgs[eye_index], "IoU for d: {:.4f}".format(iou_score2), text_positions[circle_i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[circle_i], 2)
        results.append(Result(eye_index, iou_score2, 'd'))
        return iou_score2


def show_circles(img, for_hough_img, dp, min_dist, param_1, param_2, min_radius, max_radius, eye_index):
    circles = cv2.HoughCircles(for_hough_img, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param_1, param2=param_2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        np.sort(circles)
        count = 0
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], colors[count], 6)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            iou = calc_iou_score(i, eye_index, count)
            if count < 2:
                count += 1
            else:
                count = 2

    # cv2.imshow("blured_img", blured_img)
    img_window_name = ''.join(('image ', str(eye_index)))
    edge_img_window_name = ''.join(('edge_img ', str(eye_index)))
    cv2.imshow(edge_img_window_name, for_hough_img)
    cv2.imshow(img_window_name, img)


def load_data():
    data = pd.read_csv('data-z1/duhovka.csv')
    return data


def create_trackbars():
    cv2.namedWindow('params', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('kernelSize', 'params', 7, 100, nothing)
    cv2.createTrackbar('sigma', 'params', 54, 100, nothing)
    cv2.createTrackbar('threshold1', 'params', 60, 255, nothing)  # change the maximum to whatever you like
    cv2.createTrackbar('threshold2', 'params', 72, 255, nothing)  # change the maximum to whatever you like
    cv2.createTrackbar('dp', 'params', 1, 10, nothing)
    cv2.createTrackbar('minDist', 'params', 7, 200, nothing)
    cv2.createTrackbar('minRadius', 'params', 9, 100, nothing)
    cv2.createTrackbar('maxRadius', 'params', 300, 300, nothing)
    cv2.createTrackbar('Param 1', 'params', 300, 300, nothing)
    cv2.createTrackbar('Param 2', 'params', 39, 300, nothing)


data = load_data()
colors = [
        (0, 255, 0),
        (210, 10, 52),
        (98, 24, 159)
    ]
text_positions = [
    (10, 30),
    (10, 55),
    (10, 80)
]
results = []
if __name__ == '__main__':
    orig_imgs = []
    gray_imgs = []
    for index in range(data.shape[0]):
        orig_imgs.append(cv2.imread('data-z1/' + data.nazov[index]))
        gray_imgs.append(cv2.cvtColor(cv2.imread('data-z1/' + data.nazov[index]), cv2.COLOR_BGR2GRAY))

    create_trackbars()
    while 1:
        k = cv2.waitKey(1) & 0XFF
        if k == 27:
            break

        # get params for trackbars
        kernel_size_tmp = cv2.getTrackbarPos('kernelSize', 'params')
        if kernel_size_tmp == 0:
            kernel_size_tmp += 1
        elif kernel_size_tmp % 2 != 1:
            kernel_size_tmp += 1
        kernel_size = kernel_size_tmp
        sigma = cv2.getTrackbarPos('sigma', 'params') / 10
        th1 = cv2.getTrackbarPos('threshold1', 'params')
        th2 = cv2.getTrackbarPos('threshold2', 'params')
        dp = cv2.getTrackbarPos('dp', 'params') | 1
        minDist = cv2.getTrackbarPos('minDist', 'params') | 1
        minRadius = cv2.getTrackbarPos('minRadius', 'params')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'params')
        param1 = cv2.getTrackbarPos('Param 1', 'params') | 1
        param2 = cv2.getTrackbarPos('Param 2', 'params') | 1
        results = []
        for y in range(data.shape[0]):
            blurred_img = cv2.GaussianBlur(gray_imgs[y], (kernel_size, kernel_size), sigma)
            edge_img = cv2.Canny(blurred_img, th1, th2, apertureSize=3, L2gradient=0)
            show_circles(orig_imgs[y], edge_img, dp, minDist, param1, param2, minRadius, maxRadius, y)
            orig_imgs[y] = cv2.imread('data-z1/' + data.nazov[y])
            edge_img = cv2.imread('data-z1/' + data.nazov[y])

    cv2.destroyAllWindows()
    iou_limit = 0.5
    glob_tp = 0
    glob_fp = 0
    glob_fn = 0
    for ind in range(data.shape[0]):
        tp = 0
        fp = 0
        fn = 0

        eye_filtered = list(filter(lambda x: x.eye == ind, results))
        eye_filtered.sort(key=lambda x: x.iou, reverse=True)
        z_filtered = list(filter(lambda x: x.type == 'z', eye_filtered))
        d_filtered = list(filter(lambda x: x.type == 'd', eye_filtered))

        if len(z_filtered) == 0:
            fn += 1
        if len(d_filtered) == 0:
            fn += 1

        if len(z_filtered) >= 1:
            tp += 1
        if len(d_filtered) >= 1:
            tp += 1

        if len(z_filtered) > 1:
            fp = len(z_filtered) - 1
        if len(d_filtered) > 1:
            fp = len(d_filtered) - 1

        print('Summary for eye ', data.nazov[ind])
        print('TP:', tp)
        print('FP:', fp)
        print('FN:', fn)
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        print('Precision', Precision)
        print('Recall', Recall)
        print('--------------------------------------')
        glob_tp += tp
        glob_fp += fp
        glob_fn += fn

    print('Total summary: ')
    print('TP:', glob_tp)
    print('FP:', glob_fp)
    print('FN:', glob_fn)
    Precision = glob_tp / (glob_tp + glob_fp)
    Recall = glob_tp / (glob_tp + glob_fn)
    print('Precision', Precision)
    print('Recall', Recall)
    print('--------------------------------------')


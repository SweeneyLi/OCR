# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
import predict_det
import predict_rec
import copy
import numpy as np
import re
import time


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)

    def get_rotate_crop_image(self, img, points):
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], \
                              [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: x[0][1])
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


# tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/det/"  --rec_model_dir="./inference/rec/"


args = utility.parse_args()
text_sys = TextSystem(args)


def precict_image(img, image_name, mult=True):
    if img is None:
        logger.info("error in loading image:{}".format(image_name))
        return 0, {'message': "No img"}
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    starttime = time.time()
    # img = cv2.imread(r"D:\WorkSpaces\License_Identification\Mine_Identification\imgs\1.jpg")

    dt_boxes, rec_res = text_sys(img)
    elapse = time.time() - starttime
    print("Predict time of %s: %.3fs" % (image_name, elapse))
    dt_num = len(dt_boxes)
    res_list = []
    for dno in range(dt_num):
        text, score = rec_res[dno]
        if score >= 0.5:
            res_list.append(text)
    info = extract_info(res_list)

    res = {
        "filename": image_name,
        "time": elapse,
        'info': info,
        # 'all_text': res_list
    }
    if not mult:
        res['all_text'] = res_list
    return res


def extract_info(img_text):
    info = {
        "credit_code": None,
        "name": None,
        "operator": None
    }

    last_name = False
    last_operator = False

    for line in img_text:
        if not info["credit_code"] and re.search(r'\w{17,19}', line, re.A):
            info["credit_code"] = re.search(r'\w{17,19}', line, re.A)[0]

        if not info["operator"]:
            if re.search(r"代表人.*", line):
                info["operator"] = re.search(r"代表人.*", line)[0].lstrip("代表人")
                last_operator = True if info["operator"] == '' else False
            elif last_operator:
                info["operator"] = line
                last_operator = False

        if not info["name"]:
            if re.search(r"称.*", line):
                info["name"] = re.search(r"称.*", line)[0].lstrip("称")
                last_name = True if info["name"] == '' else False
            elif last_name and line != "名":
                info["name"] = line
                last_name = False
    return info

# import glob
# # img_list = glob.glob(r"D:\WorkSpaces\License_Identification\License_pic\*.jpg")
# img_list = glob.glob(r"C:\Users\DELL\Desktop\*.*")
# for img in img_list:
#     precict_image(img, )

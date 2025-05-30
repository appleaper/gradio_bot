import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

import pandas as pd

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import copy
import numpy as np
import time
from PIL import Image
import local.function.paddle_ocr_torch.pytorchocr_utility as utility
import local.function.paddle_ocr_torch.predict_rec as predict_rec
import local.function.paddle_ocr_torch.predict_det as predict_det
import local.function.paddle_ocr_torch.predict_cls as predict_cls
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from local.function.paddle_ocr_torch.pytorchocr_utility import draw_ocr_box_txt
from utils.tool import read_yaml
from flask import request, jsonify, Blueprint

class TextSystem(object):
    def __init__(self, args, **kwargs):
        self.text_detector = predict_det.TextDetector(args, **kwargs)
        self.text_recognizer = predict_rec.TextRecognizer(args, **kwargs)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args, **kwargs)


    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
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
        print("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))

        for text, score in rec_res:
            print("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))

def imgs_sys_predict(img_path, model_name):
    image_file_list = get_image_file_list(img_path)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    draw_img_save = os.path.join(project_dir, 'data', 'tmp')
    args = utility.iniit_config(image_dir=img_path)
    config_path = os.path.join(os.path.dirname(__file__), 'model.yaml')
    config = read_yaml(config_path)
    args.det_model_path = config['sys'][model_name].get('det_model_path', args.det_model_path)
    args.rec_image_shape = config['sys'][model_name].get('rec_image_shape', args.rec_image_shape)
    args.rec_model_path = config['sys'][model_name].get('rec_model_path', args.rec_model_path)
    args.det_yaml_path = config['sys'][model_name].get('det_yaml_path', args.det_yaml_path)
    args.rec_yaml_path = config['sys'][model_name].get('rec_yaml_path', args.rec_yaml_path)

    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score

    res_info_list = []
    res_img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))


        for index, (text, score) in enumerate(rec_res):
            info = {}
            info['text'] = text
            info['score'] = score
            box = dt_boxes[index]

            info['x1y1'] = box[0]
            info['x2y1'] = box[1]
            info['x2y2'] = box[2]
            info['x1y2'] = box[3]
            res_info_list.append(info)
            # print("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            save_path = os.path.join(draw_img_save, os.path.basename(image_file))
            res_img_list.append(save_path)
            # print("The visualized image saved in {}".format(
            #     os.path.join(draw_img_save, os.path.basename(image_file))))
    return pd.DataFrame(res_info_list), res_img_list[0], []

small_model_text_recognition_route = Blueprint('small_model_text_recognition', __name__)
@small_model_text_recognition_route.route('/small_model_text_recognition', methods=['POST'])
def text_detection():
    try:
        data = request.get_json()
        sys_img_path = data.get('sys_img_path')
        sys_model_type = data.get('sys_model_type')

        if not sys_img_path or not sys_model_type:
            return jsonify({"error": "sys_img_path 和 sys_model_type 是必需的参数"}), 400

        sys_df_result, sys_img_result, sys_img_path = imgs_sys_predict(sys_img_path, sys_model_type)
        return jsonify({
            "sys_df_result": sys_df_result,
            'sys_img_result': sys_img_result,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    img_path = r'C:\use\code\PaddleOCR2Pytorch-main\doc\imgs\00018069.jpg'
    model_name = 'method_one'
    imgs_sys_predict(img_path, model_name)
    # main(utility.parse_args())
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import copy
import cv2
import numpy as np
import time
import json
import torch
from local.function.paddle_ocr_torch.pytorchocr.base_ocr_v20 import BaseOCRV20
import local.function.paddle_ocr_torch.pytorchocr_utility as utility
from local.function.paddle_ocr_torch.pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from local.function.paddle_ocr_torch.pytorchocr.data import create_operators, transform
from local.function.paddle_ocr_torch.pytorchocr.postprocess import build_post_process
from utils.tool import read_yaml
from flask import request, jsonify, Blueprint


class TextDetector(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                        [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_pse_box_type
            postprocess_params["scale"] = args.det_pse_scale
            self.det_pse_box_type = args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_fce_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs['maps'].cpu().numpy()
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        if (self.det_algorithm == "SAST" and
            self.det_sast_polygon) or (self.det_algorithm in ["PSE", "FCE"] and
                                       self.postprocess_op.box_type == 'poly'):
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        elapse = time.time() - starttime
        return dt_boxes, elapse

def imgs_det_predict(img_path, model_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    draw_img_save = os.path.join(project_dir, 'data', 'tmp')
    args = utility.iniit_config(image_dir=img_path)
    config_path = os.path.join(os.path.dirname(__file__), 'model.yaml')
    config = read_yaml(config_path)
    args.det_algorithm = config['det'][model_name].get('det_algorithm', args.det_algorithm)
    args.det_yaml_path = config['det'][model_name].get('det_yaml_path', args.det_yaml_path)
    args.det_model_path = config['det'][model_name].get('det_model_path', args.det_model_path)
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    count = 0
    total_time = 0
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    img_path_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        save_pred = os.path.basename(image_file) + "\t" + str(
            json.dumps(np.array(dt_boxes).astype(np.int32).tolist())) + "\n"
        print(save_pred)
        print("Predict time of {}: {}".format(image_file, elapse))
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        img_path_list.append(img_path)
        print("The visualized image saved in {}".format(img_path))
    if count > 1:
        print("Avg Time: {}".format(total_time / (count - 1)))
    return img_path_list[0], []

text_detection_route = Blueprint('text_detection', __name__)
@text_detection_route.route('/text_detection', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        det_img_path = data.get('det_img_path')
        det_model_type = data.get('det_model_type')

        if not det_img_path or not det_model_type:
            return jsonify({"error": "det_img_path 和 det_model_type 是必需的参数"}), 400

        msg, [] = imgs_det_predict(det_img_path, det_model_type)
        return jsonify({
            "det_result": msg,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    img_path = r'C:\use\code\PaddleOCR2Pytorch-main\doc\imgs_en\img623.jpg'
    imgs_det_predict(img_path, 'multilingual_ptocr_v3_det_infer')

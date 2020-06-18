from __future__ import division

from models.my_yolo import *

import warnings
import cv2
from matplotlib.ticker import NullLocator
from utils.utils import *
from utils.datasets import *

def get_model_yolov3(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyYolov3_CV(num_class=2, img_size=416).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def get_model_yolov3tiny(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyTinyYolov3(num_class=2,img_size=416,is_trained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def loadcv2dnnNetONNX(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print('load successful')
    return net



if __name__ == '__main__':

    model_path_yolov3= './weights/yolov3-myyolov3_99_0.96_warehouse.pth'
    model_path_yolov3tiny = './weights/yolov3-mytiny_98_0.96_warehouse.pth'


    model = get_model_yolov3(model_path_yolov3)
    # model = get_model_yolov3tiny(model_path_yolov3tiny)

    test_input = torch.randn([1,3,416,416])

    onnx_path = "./weights/test2.onnx"

    torch.onnx.export(model, test_input, onnx_path, export_params=True)

    loadcv2dnnNetONNX(onnx_path)



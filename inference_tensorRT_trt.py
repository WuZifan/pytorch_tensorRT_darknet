from utils import common
import tensorrt as trt
import cv2
import numpy as np
TRT_LOGGER = trt.Logger()
from models.yolov3_output import YOLO_NP
from utils import *
import time
from PIL import ImageOps,Image

def pad2square_cv2(image):
    h,w,c = image.shape
    dim_diff = np.abs(h-w)
    pad1,pad2= dim_diff//2 ,dim_diff-dim_diff//2

    if h<=w:
        image = cv2.copyMakeBorder(image,pad1,pad2,0,0,cv2.BORDER_CONSTANT,value=0)
    else:
        image = cv2.copyMakeBorder(image,0,0,pad1,pad2,cv2.BORDER_CONSTANT,value=0)

    return image

def pad_to_square_pil(img):
    print(img.size)
    w,h = img.size
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    # left right,top,bottom
    # pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # left,top,right,bottom
    pad = (0,pad1,0,pad2) if h<=w else(pad1,0,pad2,0)
    # Add padding
    #left,top,right,bottom
    img = ImageOps.expand(img, border=pad, fill=0)
    # img = F.pad(img, pad, "constant", value=pad_value)

    return img#, pad

def load_engine(trt_path):
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def get_sample(img_path='./data/pics/warehouse1.jpg'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w = img.shape[:2]
    img = pad2square_cv2(img)
    img = img/255

    img = cv2.resize(img,(416,416),interpolation = cv2.INTER_CUBIC)
    img = img.transpose((2,0,1))
    img = np.expand_dims(img,axis=0)
    img = np.array(img,dtype=np.float32,order='C')
    print(img.shape)
    img = np.reshape(img,(-1,))

    return img,(h,w)

def get_sample2(img_path='./data/pics/warehouse1.jpg'):
    image_raw = Image.open(img_path)
    w,h = image_raw.size
    image_raw = pad_to_square_pil(image_raw)

    new_resolution = (416,416)
    image_resized = image_raw.resize(new_resolution, resample=Image.BICUBIC)
    image_resized = np.array(image_resized, dtype=np.float32, order='C')

    image_resized /= 255.0
    # HWC to CHW format:
    image_resized = np.transpose(image_resized, [2, 0, 1])
    # CHW to NCHW format
    image_resized = np.expand_dims(image_resized, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image_resized = np.array(image_resized, dtype=np.float32, order='C')
    image_resized = np.reshape(image_resized,(-1,))
    return image_resized,(h,w)

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def main_yolov3_test():
    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)],  # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)],  # 13*13 上预测最小的
    ]
    yolo1 = YOLO_NP(anchors[0], 2, 416)
    yolo2 = YOLO_NP(anchors[1], 2, 416)
    yolo3 = YOLO_NP(anchors[2], 2, 416)

    img, org_size = get_sample()
    # img1,org_size1 = get_sample()

    # print(sum(img-img1))
    #
    # time.sleep(100000)

    print(img.shape)
    # trt_engine = './weights/yolov3-myyolov3_99_0.96_warehouse_2.trt' # 128 2.6s一张
    # trt_engine = './weights/yolov3-myyolov3_99_0.96_warehouse_3.trt'  # 256 3.0s
    # trt_engine = './weights/yolov3-myyolov3_99_0.96_warehouse_4.trt'  # 64 2.8s
    trt_engine = './weights/yolov3-myyolov3_99_0.96_warehouse_5.trt'  # 128 float16 0.4s

    #
    engine = load_engine(trt_engine)
    #
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    #
    with engine.create_execution_context() as context:
        # case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        np.copyto(inputs[0].host, img)
        t1 = time.time()
        res = common.do_inference(context, bindings=bindings, inputs=inputs,
                                  outputs=outputs, stream=stream)
        t2 = time.time()
        # print(len(res))
        # print(res[0].shape,res[1].shape,res[2].shape)

        o1 = res[0].reshape((1, 21, 13, 13))
        o2 = res[1].reshape((1, 21, 26, 26))
        o3 = res[2].reshape((1, 21, 52, 52))

        yolo_output1 = yolo1(o1)
        yolo_output2 = yolo2(o2)
        yolo_output3 = yolo3(o3)

        detections = np.concatenate([yolo_output1, yolo_output2, yolo_output3], 1)
        # print(detections.shape)

        detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

        # print('org_size',org_size)
        detections = rescale_boxes(np.array(detections), 416, org_size)
        t3 = time.time()

        # print('detect res ', len(detections))
        # print(detections)

        print('raw_foward', t2 - t1)
        print('with nms', t3 - t1)


def main_yolov3tiny_test():
    anchors = [
        [(81, 82), (135, 169), (344, 319)],
        [(10, 14), (23, 27), (37, 58)]
    ]

    yolo1 = YOLO_NP(anchors[0], 2, 416)
    yolo2 = YOLO_NP(anchors[1], 2, 416)


    trt_engine = './weights/yolov3-mytiny_98_0.96_warehouse_3.trt'  # 128 float16 0.4s

    #
    engine = load_engine(trt_engine)
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)


    t1 = time.time()

    img, org_size = get_sample()
    #

    with engine.create_execution_context() as context:
        # case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        np.copyto(inputs[0].host, img)
        res = common.do_inference(context, bindings=bindings, inputs=inputs,
                                  outputs=outputs, stream=stream)
        t2 = time.time()
        # print(len(res))
        # print(res[0].shape,res[1].shape,res[2].shape)

        o1 = res[0].reshape((1, 21, 13, 13))
        o2 = res[1].reshape((1, 21, 26, 26))


        yolo_output1 = yolo1(o1)
        yolo_output2 = yolo2(o2)

        detections = np.concatenate([yolo_output1, yolo_output2], 1)
        # print(detections.shape)

        detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

        # print('org_size',org_size)
        detections = rescale_boxes(np.array(detections), 416, org_size)
        t3 = time.time()

        print('detect res ', len(detections))
        # print(detections)

        print('raw_foward', t2 - t1)
        print('with nms', t3 - t1)

if __name__ == '__main__':
    main_yolov3tiny_test()
    main_yolov3_test()
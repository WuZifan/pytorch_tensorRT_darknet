import cv2
from models.my_yolo import *
from utils.utils import *
import random
from matplotlib.ticker import NullLocator
import time
from models.yolov3_output import YOLO_NP

def display(detections,image):
    '''
    可视化
    :param detections:
    :param image:
    :return:
    '''
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # 可视化
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

        box_w = x2 - x1
        box_h = y2 - y1

        color = [random.random() for i in range(3)]
        color.append(1)
        color = tuple(color)
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)


    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()

    # 考虑是否保存
    # plt.savefig(f"./dog.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()

def loadcv2dnnNetONNX(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print('load successful')
    return net

def pad2square_cv2(image):
    h,w,c = image.shape
    dim_diff = np.abs(h-w)
    pad1,pad2= dim_diff//2 ,dim_diff-dim_diff//2

    if h<=w:
        image = cv2.copyMakeBorder(image,pad1,pad2,0,0,cv2.BORDER_CONSTANT,value=0)
    else:
        image = cv2.copyMakeBorder(image,0,0,pad1,pad2,cv2.BORDER_CONSTANT,value=0)

    return image

def preprocess4cv2dnn(image_path):
    image = cv2.imread(image_path)

    image = pad2square_cv2(image)

    blobImage = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), None, True, False)
    print()
    return blobImage,image

def testcv2nnNet_yolov3(img_path, model_path):
    blobImage,image = preprocess4cv2dnn(img_path)
    h,w = image.shape[:2]

    net = loadcv2dnnNetONNX(model_path)

    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)],  # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)],  # 13*13 上预测最小的
    ]
    yolo1 = YOLO_NP(anchors[0], 2, 416)
    yolo2 = YOLO_NP(anchors[1], 2, 416)
    yolo3 = YOLO_NP(anchors[2], 2, 416)

    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    start_time = time.time()
    outs = net.forward(outNames)

    print('o1',outs[0].shape)
    print('o2',outs[1].shape)
    print('o3',outs[2].shape)

    yolo_output1 = yolo1(outs[0])
    yolo_output2 = yolo2(outs[1])
    yolo_output3 = yolo3(outs[2])


    detections = np.concatenate([yolo_output1, yolo_output2, yolo_output3], 1)


    detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

    print('detect res ', len(detections))
    if detections is not None:
        detections = rescale_boxes(detections, 416, (h, w))
        end_time = time.time()
        print('cost time is ',end_time-start_time)

        display(detections, image)

def testcv2nnNet_yolov3tiny(img_path, model_path):
    blobImage,image = preprocess4cv2dnn(img_path)
    h,w = image.shape[:2]

    net = loadcv2dnnNetONNX(model_path)

    anchors = [
        [(81, 82), (135, 169), (344, 319)],
        [(10, 14), (23, 27), (37, 58)]
    ]

    yolo1 = YOLO_NP(anchors[0],2,416)
    yolo2 = YOLO_NP(anchors[1],2,416)

    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    start_time = time.time()
    outs = net.forward(outNames)

    print('o1', outs[0].shape)
    print('o2', outs[1].shape)


    yolo_output1 = yolo1(outs[0])
    yolo_output2 = yolo2(outs[1])

    detections = np.concatenate([yolo_output1, yolo_output2], 1)


    detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

    if detections is not None:
        detections = rescale_boxes(detections, 416, (h, w))
        end_time = time.time()
        print('cost time is ',end_time-start_time)

        display(detections, image)


if __name__ == '__main__':
    '''
        1、模型地址：
            './weights/yolov3-myyolov3_99_0.96_warehouse.pth'
        2、模型是基于./models/my_yolo.py中 MyYolov3进行训练的。
        3、由于cv2.dnn不能执行后处理，因此将原模型中的YOLO分支提出来，
           形成了MyYolov3_CV和./models/yolov3_output/YOLO_NP这两类
        4、将模型加载到MyYolov3_CV中，最终会输出3个output，然后通过3个YOLO头，
           可以得到相应结果。
        5、然后将加载好的模型保存成.onnx模型
        6、之后通过cv2.dnn模型加载.onnx模型，然后对图像进行前处理，对输出结果进行后处理，
           得到最后结果，
    '''
    onnx_path_yolov3tiny = "./weights/yolov3-mytiny_98_0.96_warehouse.onnx"
    onnx_path_yolov3 = './weights/yolov3-myyolov3_99_0.96_warehouse.onnx'
    img_path = './data/pics/warehouse1.jpg'
    # testcv2nnNet_yolov3tiny(img_path, onnx_path_yolov3tiny)
    testcv2nnNet_yolov3(img_path,onnx_path_yolov3)
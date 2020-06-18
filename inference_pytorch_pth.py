from __future__ import division

from models.my_yolo import *

import warnings
import cv2
from matplotlib.ticker import NullLocator
from utils.utils import *
from utils.datasets import *

warnings.filterwarnings('ignore')

def preprocess(image):
    '''
    预处理
    :param image:
    :return:
    '''

    img = transforms.ToTensor()(image)
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, 416)
    img.unsqueeze_(0)
    return img

def display(detections,image):
    '''
    可视化
    :param detections:
    :param image:
    :return:
    '''
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

def get_model_yolov3(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyYolov3(num_class=2, img_size=416).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def get_model_yolov3tiny(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyTinyYolov3(num_class=2,img_size=416,is_trained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def inference(model):
    img_path = './data/pics/warehouse1.jpg'

    model.eval()

    frame = cv2.imread(img_path)
    image = Image.open(img_path)
    img = preprocess(image)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(img.type(Tensor))

    with torch.no_grad():
        # 原始的输出坐标是(center x, center y, width, height)
        start_time = time.time()
        detections = model(input_imgs)
        # nms中会转换成(x1, y1, x2, y2)
        detections = non_max_suppression(detections, 0.5, 0.4)[0]

    print('detect res nums', len(detections))

    if detections is not None:
        org_h, org_w = frame.shape[:2]
        detections = rescale_boxes(detections, 416, (org_h, org_w))
        end_time = time.time()
        print('cost time is ', end_time-start_time)
        display(detections, image)


if __name__ == '__main__':

    model_path_yolov3= './weights/yolov3-myyolov3_99_0.96_warehouse.pth'
    model_path_yolov3tiny = './weights/yolov3-mytiny_98_0.96_warehouse.pth'


    # model = get_model_yolov3(model_path_yolov3)
    model = get_model_yolov3tiny(model_path_yolov3tiny)
    dets2 = inference(model)







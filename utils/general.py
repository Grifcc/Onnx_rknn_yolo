import torch
import time
import torchvision
import numpy as np
import cv2
import os
import random
import glob
from pathlib import Path
import yaml

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xyxy2xywhn(x, w, h, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def scale_coords(pad_par, gain, boxes, shape):

    if boxes.size == 0:
        return np.array([])
    boxes[:, :] *= gain

    if pad_par == None:
        pass
    else:
        boxes[:, [0, 2]] -= pad_par[1]
        boxes[:, [1, 3]] -= pad_par[0]
    for i, _ in enumerate(boxes):
        if _[0] < 0:
            boxes[i][0] = 0
        if _[1] < 0:
            boxes[i][1] = 0
        if _[2] > shape[1]:
            boxes[i][2] = shape[1]
        if _[3] > shape[0]:
            boxes[i][3] = shape[0]

    return boxes


def process(input, mask, anchors, imgz):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    input = sigmoid(input)
    box_confidence = input[..., 4]

    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]
    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(imgz/grid_h)

    # box_wh = pow(sigmoid(input[..., 2:4])*2, 2)

    box_wh = pow(input[..., 2:4]*2, 2)

    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return np.concatenate((box, box_confidence, box_class_probs), axis=3)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def yolov5_post_process(input, Order, anchors_name, imgz,nc,conf_thres=0.45, iou_thres=0.65,maxdets=100,maxbboxs=1024):
    input0_data = input[Order[2]]
    input1_data = input[Order[1]]
    input2_data = input[Order[0]]

    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    boxes, classes, scores = [], [], []
    z = []
    for input, mask in zip(input_data, masks):
        x = np.transpose(
            process(input, mask, anchors_name, imgz), (2, 0, 1, 3))
        z.append(np.reshape(x, (1, -1, 5+nc)))

    pred = torch.from_numpy(np.concatenate(z, axis=1))

    # #dump txt befor nms
    # with open("bbox.txt","w",encoding="utf-8") as f:
    #     for i in z:
    #         for j in i[0]:
    #             if j[4]>=conf_thres:
    #                 probs=j[4:]
    #                 id=np.argmax(probs)-1
    #                 f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.6f} {}\n".format(j[0],j[1],j[2],j[3],j[4],id))

    pred = non_max_suppression(pred, conf_thres, iou_thres,maxdets,maxbboxs)[0].numpy()

    boxes = pred[:, :4]
    classes = pred[:, -1]
    scores = pred[:, 4]

    return boxes, classes, scores


def draw(image, boxes, scores, classes, CLASSES, line_thickness=3):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        CLASSES: all classes name.
    """
    if boxes.size == 0:
        return
    for box, score, cl in zip(boxes, scores, classes):
       # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        label = "{} {:.2f}".format(CLASSES[int(cl)], score)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def save_txt(img0, boxes, scores, classes, f):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    if boxes.size == 0:
        return
    boxes = xyxy2xywhn(boxes, img0.shape[1], img0.shape[0])
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        f.write("{} {} {} {} {} {}\n".format(int(cl), x, y, w, h, score))


def non_max_suppression(prediction, conf_thres=0.45, iou_thres=0.65,maxdets=100,maxbboxs=1024,classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = maxdets  # maximum number of detections per image
    max_nms = maxbboxs  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            x[:, 5:] = x[:, 4:5]
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def pad_img(img):
    h = img.shape[0]
    w = img.shape[1]
    if h == w:
        pass
        return img, None
    elif h > w:
        img = cv2.copyMakeBorder(img, 0, 0, int(
            (h-w)/2), h-w-int((h-w)/2), cv2.BORDER_CONSTANT, 0)
        pad_par = [0, int((h-w)/2)]
    elif w > h:
        img = cv2.copyMakeBorder(
            img, int((w-h)/2), w-h-int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
        pad_par = [int((w-h)/2), 0]
    return img, pad_par


def scale_img(img, imgz):
    assert img.shape[0] == img.shape[1], "不是矩形"
    gain = float(img.shape[0])/float(imgz)
    img = cv2.resize(img, (imgz, imgz))

    return img, gain


def getOrder(inputs):
    cont = list()
    for i in inputs:
        cont.append(i.shape[-1])
    cont = np.array(cont)

    return np.argsort(cont).tolist()


def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def load_yaml(data):
    data = check_file(data)
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = int(data['nc'])
    CLASSES = data["names"]
    dataset = data["val"]
    annotations=data["annotations"]
    root_path=data["path"]
    def f(a): return map(lambda b: a[b:b+2], range(0, len(a), 2))
    ANCHOR = list()
    for i in data["anchors"]:
        ANCHOR.extend(list(f(i)))
    return root_path,nc, dataset,annotations,CLASSES, ANCHOR
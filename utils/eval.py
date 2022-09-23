#/usr/bin/python
"""
YOLO 格式的数据集转化为 COCO 格式的数据集并评估
"""

import os
import cv2
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



def yolo2coco(ImagesDir,LabelsDir,classes,pred=True):
    indexes = os.listdir(ImagesDir)
    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(ImagesDir,index))
        height, width, _ = im.shape
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(LabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(LabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])
                if pred:
                    score = float(label[5])
                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                if pred:
                    dataset['annotations'].append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': cls_id,
                        'id': ann_id_cnt,
                        'image_id': k,
                        'iscrowd': 0,
                        'score':score,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
                else :
                     dataset['annotations'].append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': cls_id,
                        'id': ann_id_cnt,
                        'image_id': k,
                        'iscrowd': 0,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
                ann_id_cnt += 1
    return dataset



def save_json(folder,json_name,dataset):
    # 保存结果
    if not os.path.exists(folder):
        os.touch(folder)
    json_name = os.path.join(folder, json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(json_name))
    return json_name

def evalByCoCo(anno_json,pred_json):
    anno = COCO(anno_json)
    pred=COCO(pred_json)
    cocoEval = COCOeval(anno,pred,"bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__=="__main__":
    classes=['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    pred=yolo2coco("/workspace/yolov7_rknn/VisDrone2019-DET-val/images","/workspace/yolov7_rknn/runs/yolov7-tiny-visdrone_rm_reshape/labels",classes)
    pred_json=save_json(".","test.json",pred)
    evalByCoCo("/workspace/yolov7_rknn/VisDrone2019-DET-val/gt_vis.json",pred_json)
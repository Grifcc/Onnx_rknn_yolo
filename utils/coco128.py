import os
import shutil

with open("/workspace/yolov7_rknn/coco128_gt/datasets.txt","r",encoding="utf-8") as f:
        datasets=f.readlines()

# f128=open("coco128/datasets.txt","w",encoding="utf-8")
for i in datasets:
    shutil.copyfile(i[:-4].replace("coco128/images","gt/labels") + "txt",i[:-4].replace("coco128/images","coco128_gt/labels") + "txt")
    # f128.write(i.replace("val2017","yolov7_rknn/coco128/images"))
# f128.close()
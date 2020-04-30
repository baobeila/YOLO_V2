
"""根据xml文件，绘出真实框在原图上检查标注是否出错"""

import xml.etree.ElementTree as ET
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt as sqrt
# 需要检查的数据
# 需要检查的类别
classes = ['aoxian', 'cahuashang', 'hanzha', 'huashang', 'liewen', 'shuizhayin',
           'zhapi']
# 输入分辨率，根据网络resize的尺寸来定
input_size = 416

if __name__ == '__main__':

    # GT框宽高统计
    width = []
    height = []
    path ='../data/data_set/defect_data/train.txt'
    for line in open(path):
        image_id, = line.split()
        img_path ='../data/data_set/defect_data/train_image/%s.jpg'%(image_id)

        label_file = open('../data/data_set/defect_data/train_annotations/%s.xml' % ( image_id))

        tree = ET.parse(label_file)
        root = tree.getroot()
        size = root.find('size')
        img_w = int(size.find('width').text)  # 原始图片的width
        img_h = int(size.find('height').text)  # 原始图片的height
        img = cv2.imread(img_path)
        print(img_path)
        for obj in root.iter('object'):

            cls = obj.find('name').text
            # 如果标注不是需要的类别或者标注为difficult，就忽略
            if cls not in classes :
                print(cls)
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            w_change = (w / img_w) * input_size
            h_change = (h / img_h) * input_size
            s = w_change * h_change  # 得到了GT框面积
            width.append(sqrt(s))
            height.append(w_change / (h_change))

        # cv2.imshow('result', img)
        # cv2.waitKey()
    plt.title('Data', size=14)
    plt.xlabel('Sqrt(S)', size=14)
    plt.ylabel('W/H', size=14)
    plt.plot(width, height, 'ro')
    plt.savefig('scale.png', format='png')
    plt.show()

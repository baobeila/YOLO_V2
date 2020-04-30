import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou





def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height#归一化后的坐标值
            if xmin >= xmax or ymin >= ymax:
                continue
            if abs(ymax - ymin) < ( 6/height):#异常值处理
                continue
            dataset.append([xmax - xmin, ymax - ymin])


    return np.array(dataset)
if __name__=="__main__":
    ANNOTATIONS_PATH = "../../data/data_set/defect_data/train_annotations"
    CLUSTERS = 9
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
    # Accuracy: 62.53%
    # Boxes:
    #  [[0.19607159 0.31486014]
    #  [0.05       0.0944435 ]
    #  [0.15176152 0.13196481]
    #  [0.56434572 0.81018875]
    #  [0.075      0.17646733]]
    # Ratios:
    #  [0.43, 0.53, 0.62, 0.7, 1.15]
    # a = np.array([[0.19607159, 0.31486014],
    #               [0.05, 0.0944435],
    #               [0.15176152, 0.13196481],
    #               [0.56434572, 0.81018875],
    #               [0.075, 0.17646733]]) * 13
    # print(a)
    # [0.08424908 0.22742947]
    # [0.31081081 0.78378378]
    # [0.20437483 0.42307692]#尺寸最小

    # [0.07176444 0.12254902]
    # [0.15039851 0.25949367]
    # [0.71910112 0.75806452]#中等尺寸

# [0.14344262 0.14285714]
# [0.46864686 0.39378238]
# [0.3740458  0.21259843]#大尺寸
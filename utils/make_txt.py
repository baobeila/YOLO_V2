"""生成train.txt与test.txt文件，文件中存放图像文件名索引"""
import os
def main(txt_path,txt_file):
    f1 = open(txt_path, 'w')
    for i in os.listdir(txt_file):
        f1.write(i.split('.')[0])
        f1.write('\n')
    f1.close()
if __name__ == '__main__':
    Txttrain_File='../data/data_set/defect_data/train_image'
    Txttrain_Path = '../data/data_set/defect_data/train.txt'
    Txttest_Path = '../data/data_set/defect_data/test.txt'
    main(Txttrain_Path,Txttrain_File)




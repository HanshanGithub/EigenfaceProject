import cv2
import os
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

import QT_GUI

import QT_GUI

IMAGE_SIZE = (200, 200)
# IMAGE_SIZE = (480, 640)
def draw(data,picname):
    fig, axis = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    # plt.title(picname)
    for r in range(5):
        for c in range(5):
            # 显示单通道的灰度图像, cmap='Greys_r'
            # axis[c, r].imshow(data[:,(5*c + r)].reshape((480, 640)))
            axis[r, c].imshow(data[:,(5*r + c)].reshape((200, 200)),cmap='gray') # cmap='gray', plt 默认b,g,r色彩
            axis[r, c].set_xticks([])
            axis[r, c].set_yticks([])

    plt.savefig(picname, dpi=300)

def drawDatas():
    path = r'E:\Python\faceData\Pro\TrainSourceImg'
    files = os.listdir(path)
    fig, axis = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    for c in range(5):
        for r in range(5):
            file = os.path.join(path, files[c*5+r])
            img = plt.imread(file)
            axis[c, r].imshow(img)
            axis[c, r].set_xticks([])
            axis[c, r].set_yticks([])
    plt.savefig('AData.png', dpi=300)

def getAllPath(dirpath, *suffix):
    # print('getAllPath')
    PathArray = []
    label = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
                label.append(fn[1:4])

    return PathArray, label

def LoadData(sourcePath, *suffix):
    ImgPaths, label = getAllPath(sourcePath, *suffix)
    imageMatrix = []
    count = 0
    for imgpath in ImgPaths:
        count += 1
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(imgpath)
        # 灰度图矩阵
        mats = np.array(img)
        # 将灰度矩阵转换为向量
        imageMatrix.append(mats.ravel())
    imageMatrix = np.array(imageMatrix)

    return count, imageMatrix, label

def PcaTest(TrainsourcePath, *suffix): #targetPath_Cas,
    # count是照片数目，imageMatrix是图片矩阵，label是照片的标签
    count, imageMatrix, label= LoadData(TrainsourcePath, *suffix)

    # print(type(imageMatrix))  numpy.ndarray
    # traib_lable = list(map(int, label))
    # np.savetxt('train_lable.csv', traib_lable, delimiter=',')
    # 数据矩阵，每一列都是一个图像
    # print('----运行位置-----')
    imageMatrix = np.transpose(imageMatrix)

    # print(type(imageMatrix)) numpy.ndarray
    rows, cols = imageMatrix.shape
    # print(type(imageMatrix))  numpy.ndarray
    imageMatrix = np.mat(imageMatrix)

    # imageMatrix = cv2.equalizeHist(imageMatrix)
    # print(type(imageMatrix)) numpy.matrix
    x_std = imageMatrix
    mean_img = np.mean(imageMatrix, axis=1)
    mean_img1 = np.reshape(mean_img, IMAGE_SIZE)  # 更改了mean_img的形状

    im = Image.fromarray(np.uint8(mean_img1))
    # TO dO something

    myImg = mean_img1  # 想要得到的数据信息
    # end
    # im.show()
    # 中心化处理
    imageMatrix = imageMatrix - mean_img
    # W是特征向量， V是特征向量组 (3458 X 3458)
    # 获得协方差矩阵：计算原始人脸样本数据的协方差矩阵，即将所有人脸数据组成的矩阵与其转置相乘
    imag_mat = (imageMatrix.T * imageMatrix) / float(count) # 100x100矩阵，数值特大
    # print(type(imageMatrix))  numpy.matrix
    # 特征值分解
    W, V = np.linalg.eig(imag_mat) # 特征值W:100x1矩阵，数值特大   特征向量V:100x100矩阵，数值较小

    """
    projmat = np.hstack((V[:, 0].reshape(-1,1),
                        V[:, 1].reshape(-1,1),V[:,2].reshape(-1,1))#V[:,0].reshape(-1,1)
                        )#V[:,0].reshape(-1,1)
    Y = x_std * projmat
    myImg = Y
    """
    # V_img是协方差矩阵的特征向量组
    # 映射矩阵
    V_img = imageMatrix * V  # 40000x100矩阵，数值正常
    imgMat = imageMatrix
    # draw(imgMat)
    # 降序排序后的索引值
    axis = W.argsort()[::-1]
    V_img = V_img[:, axis]

    # number = 0
    # x = sum(W)
    # for i in range(len(axis)):
    #     number += W[axis[i]]
    #     if float(number) / x > 0.9:
    #         print('累加有效值是：', i) # 累加有效值是62
    #         break

    # 取前62个最大特征值对应的特征向量
    V_img_finall = V_img[:, :25]  # 映射矩阵 40000x25 #对应draw c,r
    # V_img_finall = V_img[:, :20]  # 映射矩阵 40000x20 #对应draw c,r/



    return  myImg, V_img_finall, imageMatrix, mean_img, label, count

def recognize(TestsourthPath, V_img_finall, imageMatrix, mean_img, train_count, train_lable, *suffix):
    # 读取test矩阵
    test_count, test_imageMatrix, test_label = LoadData(TestsourthPath, *suffix)
    # test_label_ = list(map(int, test_label))
    # np.savetxt('test_label.csv', test_label_, delimiter=',')
    V_img_finall = np.mat(V_img_finall) # 特征向量
    # 训练样本空间
    test_imageMatrix = np.transpose(test_imageMatrix)
    test_imageMatrix = np.mat(test_imageMatrix)
    projectedImage = V_img_finall.T * imageMatrix # 数值巨大
    np.savetxt('pca_train_matrix.csv', projectedImage, delimiter=',')

    test_imageMatrix = test_imageMatrix - mean_img
    # 测试样本空间
    test_projectedImage = V_img_finall.T * test_imageMatrix # 数值巨大
    np.savetxt('pca_test_matrix.csv', test_projectedImage, delimiter=',')

    number = 0
    k = 30
    result = []
    for test in range(test_count):
        distance = []
        for train in range(train_count):
            temp = np.linalg.norm(test_projectedImage[:, test] - projectedImage[:, train])
            distance.append(temp)

        minDistance = min(distance)
        index = distance.index(minDistance)
        result.append(train_lable[index])

        if test_label[test] == train_lable[index]:
            number += 1

    return number/float(test_count), result



if __name__ == '__main__':
    start = time.time()
    TrainsourcePath = r'E:\Python\faceData\Pro\TrainFaceData'
    TestsourthPath = r'E:\Python\faceData\Pro\TestFaceData'

    # V_img_finall特征向量映射矩阵；imageMatrix 原始减去平均人像；mean-img平均人像
    myImg, V_img_finall, imageMatrix, mean_img, train_label, train_count = PcaTest(TrainsourcePath,  '.jpg', '.JPG', 'png', 'PNG') #targetPath_Cas,

    mean = np.reshape(mean_img, IMAGE_SIZE)
    plt.figure()
    plt.imshow(mean)
    plt.title('mean')
    plt.show()

    imgMat = imageMatrix
    # draw(imgMat)

    np.savetxt('V_img_finall.csv', V_img_finall, delimiter=',')
    np.savetxt('my_info.csv', myImg, delimiter=',')
    succsee, result = recognize(TestsourthPath,  V_img_finall, imageMatrix, mean_img, train_count, train_label, '.jpg', '.JPG', 'png', 'PNG')
    end = time.time()
    print('程序运行时间是：{}'.format(end - start))





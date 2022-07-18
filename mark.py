import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import pcatest


def mark():
    from watermarker.marker import add_mark
    add_mark(file=r"meanIMG.jpg", out=r"E:/_files", mark="mean", opacity=0.2, angle=0, space=30)

# E:\_Files
def removeJPG():
    file_name = sys.path[0]
    for root,dirs,files in os.walk(file_name):
        for name in files:
            if name.endswith(".jpg"):
                os.remove(os.path.join(root,name))
                print("delete file:",os.path.join(root,name))

def isOn():
    pic = 'mean.jpg'
    print(os.path.exists(pic))

def load():
    count, mats, lable = pcatest.LoadData(r'E:\Python\faceData\Pro\TrainSourceImg','.jpg')
    print(len(mats))
    try:
        imageMatrix = np.transpose(mats)
        imageMatrix = np.mat(imageMatrix)
        pcatest.draw(imageMatrix, './pic/Adata.png')
    except:
        print('error')

def saveABdata():
    isAB = os.path.exists('./pic/Adata.png') & os.path.exists('./pic/Bdata.png')
    if isAB == False:
        print("   error! 数据加载错误，请检查数据路径")
        return
    plt.figure()
    imgA = plt.imread('./pic/Adata.png')
    imgB = plt.imread('./pic/Bdata.png')
    plt.subplot(2, 2, 1)
    plt.imshow(imgA)
    plt.title('initial pictures')
    plt.subplot(2, 2, 2)
    plt.title('after cut')
    plt.imshow(imgB,cmap='gray')
    plt.show()
    plt.savefig('ABdata.png',dpi=300)


def drawAdata():
    path = r'E:\Python\faceData\Pro\TrainSourceImg'
    files = os.listdir(path)

    fig, axis = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    for c in range(5):
        for r in range(5):
            file = os.path.join(path, files[c * 5 + r])
            img = plt.imread(file)
            axis[c, r].imshow(img)
            axis[c, r].set_xticks([])
            axis[c, r].set_yticks([])
    plt.savefig('AData.png', dpi=300)

def location():
    isHere = os.path.exists('./pic/Aadata.png')
    print(isHere)

if __name__ == '__main__':
    # removeJPG()
    # isOn()
    # load()
    # drawAdata()
    # isAB = os.path.exists('Adata.png') & os.path.exists('data.jpg')
    # print(isAB)
    saveABdata()
    # location()
import cv2
import numpy as np
import config
import matplotlib.pyplot as plt


def cv_show(name, img):
    """
    images处理结果展示
    :param img:
    :param name:
    :return:
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_image():
    """
    把彩色图处理成灰度图
    :return:
    """
    img = cv2.imread('{}/images/cat.jpg'.format(config.BASE_DIR, ))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 展示处理完成的灰度图，
    cv_show("img_gray", img_gray)
    print(img_gray.shape)


def hsv_image():
    """
    将图片读取的BGR转化为HSV
    H - 色调（主波长）。
    S - 饱和度（纯度/颜色的阴影）。
    V值（强度）
    :return:
    """
    img = cv2.imread('{}/images/cat.jpg'.format(config.BASE_DIR, ))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv_show("hsv", hsv)


def image_threshold():
    """
    图像阈值的处理

    ret, dst = cv2.threshold(src, thresh, maxval, type)¶
    src： 输入图，只能输入单通道图像，通常来说为灰度图
    dst： 输出图
    thresh： 阈值
    maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV

    cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0

    cv2.THRESH_BINARY_INV THRESH_BINARY的反转
    cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
    cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
    cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转
    :return:
    """
    img = cv2.imread('{}/images/cat.jpg'.format(config.BASE_DIR, ))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO',
              'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def images_smooth():
    """
    去除图片中的噪点
    :return:
    """
    img = cv2.imread('{}/images/lenaNoise.png'.format(config.BASE_DIR, ))

    # 原图展示
    # cv2.imshow('img', img)
    # cv_show('img', img)

    # 均值滤波
    # 简单的平均卷积操作 对随机噪声有很好的去燥效果
    blur = cv2.blur(img, (3, 3))  # (3,3)表示生成的模糊内核是一个3*3的矩阵。
    # cv_show('blur', blur)

    # 方框滤波
    # 基本和均值一样，可以选择归一化
    # 如果均衡化（即normalize==ture，这也是默认值），则其本质是均值滤波
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    # cv_show('box', box)

    # 方框滤波
    # 基本和均值一样，可以选择归一化,容易越界
    box = cv2.boxFilter(img, -1, (3, 3), normalize=False)
    # cv_show('box', box)

    # 高斯滤波
    # 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
    aussian = cv2.GaussianBlur(img, (5, 5), 1)
    # cv_show('aussian', aussian)

    # 中值滤波
    # 相当于用中值代替
    median = cv2.medianBlur(img, 5)  # 中值滤波 5代表滤波模板的尺寸大小，必须是大于1的奇数
    # cv_show('median', median)

    # 展示所有的处理结果
    res = np.hstack((blur, box, aussian, median))
    # print (res)
    cv_show('median vs average', res)


def morphological_operation_1():
    """
    形态学操作：改变物体的形状

    形态学腐蚀瘦身操作

    去除图片主体内容边缘干扰
    :return:
    """
    img = cv2.imread('{}/images/dige.png'.format(config.BASE_DIR, ))

    # cv_show('img', img)
    # 腐蚀瘦身操作 kernel内(3,3)指定主体机构大小
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    # cv_show('erosion', erosion)

    pie = cv2.imread('{}/images/pie.png'.format(config.BASE_DIR, ))
    # cv_show('pie', pie)

    kernel = np.ones((30, 30), np.uint8)
    erosion_1 = cv2.erode(pie, kernel, iterations=1)
    erosion_2 = cv2.erode(pie, kernel, iterations=2)
    erosion_3 = cv2.erode(pie, kernel, iterations=3)
    res = np.hstack((erosion_1, erosion_2, erosion_3))
    cv_show('res', res)


def morphological_operation_2():
    """
    形态学膨胀操作
    :return:
    """
    img = cv2.imread('{}/images/dige.png'.format(config.BASE_DIR, ))
    # cv_show('img', img)

    kernel = np.ones((3, 3), np.uint8)
    dige_erosion = cv2.erode(img, kernel, iterations=1)
    # cv_show('erosion', dige_erosion)

    kernel = np.ones((3, 3), np.uint8)
    dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
    # cv_show('dilate', dige_dilate)

    pie = cv2.imread('{}/images/pie.png'.format(config.BASE_DIR, ))
    kernel = np.ones((30, 30), np.uint8)
    dilate_1 = cv2.dilate(pie, kernel, iterations=1)
    dilate_2 = cv2.dilate(pie, kernel, iterations=2)
    dilate_3 = cv2.dilate(pie, kernel, iterations=3)
    res = np.hstack((dilate_1, dilate_2, dilate_3))
    cv_show('res', res)


def opening_and_closing_operation():
    """
    开运算和闭运算

    开运算：先腐蚀，再膨胀
    闭运算：先膨胀，再腐蚀
    :return:
    """
    img = cv2.imread('{}/images/dige.png'.format(config.BASE_DIR, ))

    # 开运算
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv_show('opening', opening)

    # 闭运算
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv_show('closing', closing)


def gradient_operation():
    """
    梯度=膨胀-腐蚀
    :return:
    """
    pie = cv2.imread('{}/images/pie.png'.format(config.BASE_DIR, ))
    kernel = np.ones((7, 7), np.uint8)
    dilate = cv2.dilate(pie, kernel, iterations=5)
    erosion = cv2.erode(pie, kernel, iterations=5)
    res = np.hstack((dilate, erosion))
    # cv_show('res', res)

    gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
    cv_show('gradient', gradient)


def hat_and_black_hat():
    """
    礼帽 = 原始输入-开运算结果
    黑帽 = 闭运算-原始输入
    :return:
    """
    img = cv2.imread('{}/images/dige.png'.format(config.BASE_DIR, ))
    kernel = np.ones((7, 7), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # cv_show('tophat', tophat)

    # 黑帽
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv_show('blackhat ', blackhat)


def images_gradient_sobel():
    """
    sobel算子
    算子detail: images/sobel_1.png

    st = cv2.Sobel(src, ddepth, dx, dy, ksize)
    ddepth:图像的深度,-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
    dx和dy分别表示水平和竖直方向
    dx-x方向上的差分阶数，1或0
　　 dy-y方向上的差分阶数，1或0
    ksize是Sobel算子的大小
    :return:
    """
    # 图像梯度可以把图像看成二维离散函数，图像梯度其实就是这个二维离散函数的求导
    # OpenCV提供了三种不同的梯度滤波器，或者说高通滤波器:Sobel，Scharr和Lapacian。Sobel，Scharr
    # 其实就是求一阶或二阶导。Scharr是对Sobel的部分优化。Laplacian是求二阶导
    img = cv2.imread('{}/images/pie.png'.format(config.BASE_DIR, ),
                     cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", img)
    # cv_show('img', img)

    # dst = cv2.Sobel(src, ddepth, dx, dy, ksize)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # cv_show('sobelx', sobelx)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    # cv_show('sobelx', sobelx)

    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    # cv_show('sobely', sobely)

    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # cv_show('sobelxy', sobelxy)

    sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    sobelxy = cv2.convertScaleAbs(sobelxy)
    # cv_show('sobelxy', sobelxy)

    img = cv2.imread('{}/images/lena.jpg'.format(config.BASE_DIR, ), cv2.IMREAD_GRAYSCALE)
    # cv_show('img', img)

    sobelx_1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx_2 = cv2.convertScaleAbs(sobelx_1)
    sobely_3 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely_4 = cv2.convertScaleAbs(sobely_3)
    sobelxy = cv2.addWeighted(sobelx_2, 0.5, sobely_4, 0.5, 0)
    res = np.hstack((sobelx_1, sobelx_2, sobelxy))
    # cv_show('res', res)

    # img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    sobelxy = cv2.convertScaleAbs(sobelxy)
    cv_show('sobelxy', sobelxy)


def images_gradient_laplacian():
    """
    laplacian算子
    :return:
    """
    # 不同算子的差异比较
    img = cv2.imread('{}/images/lena.jpg'.format(config.BASE_DIR, ), cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    res = np.hstack((sobelxy, scharrxy, laplacian))
    cv_show('res', res)

    # cv_show(img, 'img')


if __name__ == '__main__':
    images_gradient_laplacian()

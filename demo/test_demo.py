import config
import cv2

from basic_operation import basic_operation as bo


def people_and_car_counter(image_name):
    """

    :param image_name:
    :return:
    """
    img = cv2.imread(config.image_dir.format(image_name))
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 140, 255, cv2.THRESH_BINARY_INV)[1]
    # bo.cv_show('ref', ref)
    refCnts, hierarchy = cv2.findContours(
        ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ref, refCnts, -1, (0, 0, 255), 3)
    # bo.cv_show('img', ref)

    cnts = refCnts
    cur_img = img.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    # bo.cv_show('img', cur_img)
    locs = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # ar = w / float(h)
        print((x, y, w, h))
        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        # if 2.5 < ar < 4.0:
        #     if 40 < w < 55 and 10 < h < 20:
        #         # 符合的留下来
        #         locs.append((x, y, w, h))

                # 将符合的轮廓从左到右排序


if __name__ == '__main__':
    image_name = 'picture_1.jpeg'
    people_and_car_counter(image_name)
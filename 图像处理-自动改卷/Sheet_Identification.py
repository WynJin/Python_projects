# -*- codeding = utf-8 -*-
# @Time : 2022/11/20 14:11
# @Author : 怀德
# @File : Sheet_Identification.py
# @Software: PyCharm

import cv2 as cv
import glob
import numpy as np
import imutils
import pyzbar.pyzbar as pyzbar


def detect_square(gray_img):
    '''
    # 检测试卷图片四周的4个正方形
    :return: 四个正方形的坐标
    '''
    copy = gray_img.copy()
    # a.预处理:
    decrease_noise = cv.fastNlMeansDenoising(gray_img, 10, 15, 7, 21)
    blurred_img = cv.GaussianBlur(decrease_noise, (3, 3), 0)
    canny_img = cv.Canny(blurred_img, 20, 40)
    thresh = cv.threshold(canny_img, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)  # 返回contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
    # 在二值图像（binary image）中寻找轮廓（contour）,RETR_EXTERNAL:只检测最外围轮廓,CV_CHAIN_APPROX_NONE 只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    #     # 每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数
    contour_list = []
    for c in contours:
        area = cv.contourArea(c)  # 计算轮廓内区域的面积
        perimeter = cv.arcLength(c, True)  # 计算轮廓周长
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)  # 获取轮廓角点坐标
        if len(approx) < 8:
            x, y, w, h = cv.boundingRect(c)  # 返回轮廓点的x，y坐标，宽和高
            # print("w - h = " + str(w - h))
            # print("x , y = " + str(x) + " , " + str(y))
            if abs(w - h) <= 2 and w > 12 and w < 17:  # 若轮廓的高和宽小于1，判断形似正方形
                # 用w 和 h 等设置阈值，相当于检测正方形的参数，调的不准可能出现干扰
                # cv.drawContours(copy, c, -1, (255, 0, 0), 3)  # 绘制正方形轮廓线 #36, 255, 12
                contour_list.append([x, y])  # 集合点作为集合
    # print(contour_list)  # 由列表可见，诶个像素y值递减findContour结果从从上到下检测轮廓，
    # b.通过4个边界左边划出边界
    x_coords = np.ravel([contour[0] for contour in contour_list])  # 使用ravel将角点的x坐标合成一维
    y_coords = np.ravel([contour[1] for contour in contour_list])
    if len(contour_list) > 4:  # 计算答题纸边界坐标
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        cv.rectangle(copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  #
        LB_coords = np.array([x_min, y_max], dtype='float64')  # 值和坐标相反
        LT_coords = np.array([x_min, y_min], dtype='float64')
        RB_coords = np.array([x_max, y_max], dtype='float64')
        RT_coords = np.array([x_max, y_min], dtype='float64')
        edge_list = np.array([LT_coords, RT_coords, RB_coords, LB_coords])
    else:
        cv.rectangle(copy, contour_list[3], contour_list[0], (0, 0, 255), 2)  #
        edge_list = np.array([contour_list[3], contour_list[2], contour_list[0], contour_list[1]])
    cv.imshow('square_detect', copy)
    cv.waitKey()
    print(edge_list)
    return edge_list


def Homography(img, four_point):
    '''
    # 应用图像集合变换
    :param four_point: 四个正方形边界坐标
    :return: 几何变换后的试卷图像
    '''
    size = img.shape
    pst_src = np.array(  # 取得原图四个顶点坐标
        [
            [0, 0], [size[1] - 1, 0],
            [size[1] - 1, size[0] - 1],
            [0, size[0] - 1]
        ], dtype=float
    )
    new_img = np.zeros_like(img)  # 创建空图片
    h, status = cv.findHomography(four_point, pst_src)  # 使用Homography函数计算转换矩阵，得出源点和目标点之间的意义映射关系
    # 对图片进行仿射变换
    im_temp = cv.warpPerspective(img, h, (new_img.shape[1], new_img.shape[0]))
    cv.fillConvexPoly(new_img, four_point.astype(int), 0, 16)
    new_img = new_img + im_temp  # 图片叠加
    cv.imshow('Answer Area', new_img)
    cv.waitKey()
    return new_img


def roi_mask(img):
    '''
    :return:  截取的两个感兴趣区域：选择区域和条形码
    '''
    size = img.shape
    mask = np.zeros_like(img)
    mask_poly1 = np.array(
        [[[0, (size[1] - 1) // 3 - 10], [(size[1] - 1) // 3 + 2, (size[1] - 1) // 3 - 10],
          [(size[1] - 1) // 3 + 2, (size[1] - 1) // 3 + 120], [0, (size[1] - 1) // 3 + 120]]])
    mask1 = cv.fillPoly(mask, pts=mask_poly1, color=255)  # 选项题掩膜
    roi_img1 = cv.bitwise_and(img, mask1)  # 掩膜和图像作布尔运算
    mask = np.zeros_like(img)
    mask_poly2 = np.array(
        [[[235, 100], [390, 100], [390, 160], [235, 160]]])
    mask2 = cv.fillPoly(mask, pts=mask_poly2, color=255)
    roi_img2 = cv.bitwise_and(img, mask2)  # 掩膜和图像作布尔运算
    # cv.imshow('mask1', mask1)
    # cv.imshow('roi_img1', roi_img1)
    # cv.imshow('mask2', mask2)
    # cv.imshow('roi_img2', roi_img2)
    cv.waitKey()
    return roi_img1, roi_img2

def barcode_detect(gray):
    copy = gray.copy()
    sharpen = sharpen_img(copy)  # 锐化
    # 从x 和 y 方向计算图像的梯度幅度表示
    gradX = cv.Sobel(sharpen, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv.Sobel(sharpen, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv.subtract(gradX, gradY)  # 从x梯度中减去y梯度
    gradient = cv.convertScaleAbs(gradient)

    blurred = cv.blur(gradient, (4, 4))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 7))  # 闭运算
    closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(closed.copy(), cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)  # 返回contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
    if len(contours) == 0:
        return None
    else:
        for i, c in enumerate(contours):
            cv.drawContours(copy, c, -1, (255, 0, 0), 4)  # 绘制轮廓线

    barcodes = pyzbar.decode(copy)  # 识别条形码
    for barcode in barcodes:
        barcodeData = barcode.data.decode("utf-8")
        cv.putText(copy, "code:" + str(barcodeData), (240, 90), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 40, 100), 1)

    # copy = sharpen_img(copy)
    cv.imshow("box_barcode", copy)
    cv.waitKey()

    return copy


def sharpen_img(gray):
    kernel = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="float32")  # 锐化核 若模板中心不是5，而是4，则与上面算子模板效果基本相同，仅方向相反）

    result = cv.filter2D(gray, -1, kernel)
    cv.imshow('result1 ', result)
    cv.waitKey(0)
    return result



# 检查空格是否被填充
def sure_if_fill(image):
    copy = image.copy()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(copy, cv.MORPH_CLOSE, kernel, iterations=1)  # 闭合运算 把填土
    canny_img = cv.Canny(dst, 250, 256)
    contours, hierarchy = cv.findContours(canny_img, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)  # 返回contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
    mem_x = 0
    answer_list = []
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        if area > 0.2 and abs(mem_x - x) > 15:
            cv.drawContours(copy, c, -1, (0, 255, 0), 2)  # 绘制轮廓线 方便后续查看
            answer_list.append(c)
    cv.imshow('fill_img', copy)
    cv.waitKey()
    # print("answer_list"+str(answer_list))
    # print("answer_list len "+str(len(answer_list)))
    return copy


def check_answer(image):# 检测填涂选框，识别标注其选项
    copy = image.copy()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) # 闭运算算子
    dst = cv.morphologyEx(copy, cv.MORPH_CLOSE, kernel, iterations=1)  # 闭运算
    canny_img = cv.Canny(dst, 250, 256) # 设定Canny算子阈值
    contours, hierarchy = cv.findContours(canny_img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    m = 30
    contours = tuple(reversed(list(contours)))# 将识别的边缘信息倒叙，方便从上到下阅卷
    mem_x = 0 # 记录上次选项横坐标位置
    answer_list = []
    for i, c in enumerate(contours):
        area = cv.contourArea(c) # 计算选项轮廓面积
        x, y, w, h = cv.boundingRect(c) # 找到角点坐标 高和宽
        if area > 2.5 and abs(mem_x - x) > 5:  # 筛选正确选项
            m += 1
            mem_x = x
            mm = cv.moments(c)  # 几何重心的获取
            cx = mm['m10'] / mm['m00']  # 获取几何中心的x
            cy = mm['m01'] / mm['m00']  # 获取几何重心的y
            cv.putText(copy, str(m), (np.int(cx) + 15, np.int(cy)), cv.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 1) #标注选项题号
            answer_r, color = option(cx, m - 31)  # 识别选项详细
            cv.drawContours(copy, c, -1, (255, 0, 0), 4)  # 绘制轮廓线
            cv.putText(copy, "  " + answer_r, (np.int(cx) + 20, np.int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1) # 标注选项
            answer_list.append([np.int(cx), np.int(cy)])  # 集合点作为集合
    return copy
    # print("answer_list" + str(answer_list))
    # print("answer_list len " + str(len(answer_list)))
    # cv.imshow('check_img', copy)
    # cv.waitKey()



def option(pos, count):
    """通过选项的边缘轮廓重心x坐标与正确x坐标比较，判断具体选项"""
    list_x = [54, 72, 90, 108, 127, 147, 166, 185, 204, 223, 241]
    answer = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    right_answer = ['C', 'F', 'K', 'B', 'A', 'E', 'J', 'G', 'D', 'K']
    color = (255, 0, 0)
    for i, x in enumerate(list_x):
        if pos > x - 10 and pos < x + 10:
            if answer[i] == right_answer[count]:
                color = (255, 0, 0)  # 正确为白色
                break
            else:
                color = (0, 255, 0)
                break
    return answer[i], color



def integrate_final(barcode, check, img):
    '''
  整合阅卷结果
    '''
    size = img.shape
    mask = np.zeros_like(img)
    mask_poly1 = np.array(
        [[[0, (size[1] - 1) // 3 - 10], [(size[1] - 1) // 3 + 2, (size[1] - 1) // 3 - 10],
          [(size[1] - 1) // 3 + 2, (size[1] - 1) // 3 + 120], [0, (size[1] - 1) // 3 + 120]]])
    mask1 = cv.fillPoly(mask, pts=mask_poly1, color=255)
    final = cv.bitwise_or(img, mask1)  # 掩膜和图像作布尔运算
    mask = np.zeros_like(img)
    mask_poly2 = np.array(
        [[[235, 100], [390, 100], [390, 160], [235, 160]]])
    mask2 = cv.fillPoly(mask, pts=mask_poly2, color=255)
    final = cv.bitwise_or(final, mask2)  # 掩膜和图像作布尔运算
    final = barcode + check + final
    return final


def entire_process(img):
    four_point = detect_square(img)  # 预处理检测试卷边缘后识别试卷四周的定位正方形
    img_dst = Homography(img, four_point)  # 根据定位正方形坐标进行仿射变换，矫正试卷图像倾斜
    roi_img1, roi_img2 = roi_mask(img_dst)  # 获得选项和条形码两个ROI
    barcode_img = barcode_detect(roi_img2)  # 检测条形码位置，并识别其具体值
    answer_img = sure_if_fill(roi_img1)  # 边缘检测配合闭运算填充空隙，强化填涂
    check_img = check_answer(answer_img)  # 识别填涂结果将批阅结果绘制到图像上
    final_img = integrate_final(barcode_img, check_img, img_dst)  # 整合图像，返回完整阅卷结果
    return final_img


if __name__ == "__main__":
    img = cv.imread('IMG_4.JPG')  # (4032,3024, 3) 4：3比例 原图为拍照获取
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)# Create window with freedom of dimensions 由于tu向比屏幕的分辨率大，使用
    # cv2.resizeWindow("output", 1440, 1080)# Resize window to specified dimensions
    img = cv.resize(img, (1120, 756))  ## Resize image缩小3.8倍率
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    final_img = entire_process(gray_img)
    cv.imshow('final_img', final_img)
    cv.waitKey()

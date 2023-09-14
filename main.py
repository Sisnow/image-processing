# -*- coding: utf-8 -*-
import sys
from functools import partial

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFileDialog

import image_trans as tr
import video, correct, change_color
from UI import setsize, demo, SelectStyle, ChangeColor
from UI.BaseAdjustDialog import Ui_baseAdjustDialog


# qt图片转化为opencv
def qtpixmap_to_cvimg(qtpixmap):
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    return result


# opencv图片转化为qt
def cvimg_to_qtimg(cvimg):
    rows, columns, channels = cvimg.shape
    bytesPerLine = channels * columns
    pix = QImage(cvimg.data.tobytes(), columns, rows, bytesPerLine, QImage.Format_BGR888)
    return QtGui.QPixmap(pix)


class MyWindow(QtWidgets.QMainWindow, demo.Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        # 子窗口
        self.ChildDialog = ChildWin()
        self.baseAdjustDialog = BaseAdjustDialog()
        self.selectStyleDialog = SelectStyleDialog()
        self.changeColorDialog = ChangeColorDialog()

        # 待处理图片
        self.Image = None
        # 处理后图片
        self.Dest_Image = None
        self.Tmp_Image = None

        # 定义触发信号
        self.actionopen.triggered.connect(self.openImage)
        self.actionsave.triggered.connect(self.saveImage)
        self.actiontansfer.triggered.connect(self.transferImage)
        self.actionvideo.triggered.connect(self.openTransferVideo)
        self.actionCorrect.triggered.connect(self.correct)
        self.actioncorrect_2.triggered.connect(self.correct2)
        self.actionChangeColor.triggered.connect(self.openChangeColor)
        self.pushButton.clicked.connect(self.onClicked)
        self.pushButton_2.clicked.connect(self.openBaseAdjustDialog)
        self.pushButton_3.clicked.connect(self.toGray)
        self.pushButton_4.clicked.connect(self.toBinary)
        self.pushButton_5.clicked.connect(self.invert)
        self.pushButton_6.clicked.connect(self.emboss)
        self.pushButton_7.clicked.connect(self.canny)
        self.pushButton_8.clicked.connect(self.blur)
        self.pushButton_9.clicked.connect(self.sharpen)
        self.pushButton_10.clicked.connect(self.rotate)
        self.pushButton_11.clicked.connect(self.refresh)

    # 工具函数
    def saveAndShow(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("result.jpg", img)
        self.Dest_Image = img
        tmp = cvimg_to_qtimg(img)
        tmp = QtGui.QPixmap(tmp).scaled(self.src_image.width(), self.src_image.height(), Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
        self.dest_image.setPixmap(tmp)

    # 定义槽函数
    # 打开图片
    def openImage(self):
        self.src_image.setStyleSheet("")
        img_name, img_type = QFileDialog.getOpenFileName(self,
                                                         "打开图片",
                                                         "",
                                                         " *.jpg *.png;;*.jpeg;;*.bmp;;All Files (*)")
        if img_type:
            # 利用qlabel(src_image)显示图片
            pix = QtGui.QPixmap(img_name).scaled(self.src_image.width(), self.src_image.height(), Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
            pix_tmp = QtGui.QPixmap(img_name)
            self.Image = qtpixmap_to_cvimg(pix_tmp)
            self.Dest_Image = self.Image
            self.src_image.setPixmap(pix)
            self.dest_image.setPixmap(pix)
        return

    # 保存图片
    def saveImage(self):
        if self.Dest_Image is not None:
            img_name, img_type = QFileDialog.getSaveFileName(self, "保存图片", "", "Images (*.png *.xpm *.jpg)")
            if img_name:
                cv2.imwrite(img_name, self.Dest_Image)
        return

    # 风格迁移
    def transferImage(self):
        if self.Dest_Image is None:
            return
        # 打开风格图片
        img_name, img_type = QFileDialog.getOpenFileName(self,
                                                         "选择风格图片",
                                                         "",
                                                         " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        if img_type:
            img = cv2.imread(img_name)
            # 保存图片，然后调用程序
            cv2.imwrite("image/tmp_style.jpg", img)
            if self.Dest_Image is not None:
                cv2.imwrite("image/tmp_content.jpg", self.Dest_Image)
                tr.run()
                print("风格迁移完成！")
                self.Dest_Image = cv2.imread("image/result.jpg")
                tmp = QtGui.QPixmap("image/result.jpg").scaled(self.src_image.width(), self.src_image.height(),
                                                               Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)
                self.dest_image.setPixmap(tmp)
        return

    # 打开视频风格迁移窗口
    def openTransferVideo(self):
        self.selectStyleDialog.signal.connect(self.transferVideo)
        self.selectStyleDialog.label.setPixmap(QtGui.QPixmap("image/1.jpg").scaled(256, 256,
                                                                                   Qt.KeepAspectRatio,
                                                                                   Qt.SmoothTransformation))
        self.selectStyleDialog.label_2.setPixmap(QtGui.QPixmap("image/2.jpg").scaled(256, 256,
                                                                                     Qt.KeepAspectRatio,
                                                                                     Qt.SmoothTransformation))
        self.selectStyleDialog.show()

    # 视频风格迁移
    def transferVideo(self, parameter):
        video_name, video_type = QFileDialog.getOpenFileName(self,
                                                             "选择视频",
                                                             "",
                                                             " *.mp4;;*.avi;;All Files (*)")
        if video_type:
            print("视频正在生成。。。")
            video.run(video_name, parameter)
            print("视频生成完成，请在当前目录下查看！")
        return

    # 打开图像缩放窗口
    def onClicked(self):
        self.ChildDialog.show()
        self.ChildDialog.signal.connect(self.getDataSetSize)

    # 获得子窗口数据并调整图片大小
    def getDataSetSize(self, parameter):
        if self.Dest_Image is not None:
            self.Dest_Image = cv2.resize(self.Dest_Image, (int(parameter[0]), int(parameter[1])),
                                         interpolation=cv2.INTER_LINEAR)
            img = cvimg_to_qtimg(self.Dest_Image)
            self.dest_image.setPixmap(img)

    # 打开颜色变化窗口
    def openChangeColor(self):
        self.changeColorDialog.dialogRejected.connect(self.openChangeColorRejected)
        self.changeColorDialog.dialogAccepted.connect(self.openChangeColorAccepted)
        self.changeColorDialog.signal.connect(self.changeColor)
        self.changeColorDialog.show()

    def openChangeColorAccepted(self):
        if self.Dest_Image is not None:
            self.saveAndShow(self.Tmp_Image)
        self.update()

    def openChangeColorRejected(self):
        self.update()

    # 颜色变化
    def changeColor(self, parameter):
        if self.Dest_Image is not None:
            self.Tmp_Image = change_color.run(self.Dest_Image, parameter[0], parameter[1])
        self.update()

    # 打开图像调节窗口
    def openBaseAdjustDialog(self):
        self.baseAdjustDialog.dialogRejected.connect(self.baseAdjustDialogRejected)
        self.baseAdjustDialog.dialogAccepted.connect(self.baseAdjustDialogAccepted)
        self.baseAdjustDialog.brightSliderReleased.connect(self.adjustBright)
        self.baseAdjustDialog.warmSliderReleased.connect(self.adjustWarm)
        self.baseAdjustDialog.saturabilitySliderReleased.connect(self.adjustSaturation)
        self.baseAdjustDialog.contrastSliderReleased.connect(self.adjustContrast)
        self.baseAdjustDialog.show()

    def baseAdjustDialogAccepted(self):
        self.saveAndShow(self.Tmp_Image)
        self.update()

    def baseAdjustDialogRejected(self):
        self.update()

    # 对比度
    # 参考：https://blog.csdn.net/zhaitianbao/article/details/120107171
    def adjustContrast(self, percent):
        if self.Dest_Image is None:
            return
        src = self.Dest_Image
        # 计算对比度因子
        alpha = percent / 100.0
        alpha = max(-1.0, min(1.0, alpha))
        # 创建一个新的图片
        temp = src.copy()
        row = src.shape[0]
        col = src.shape[1]
        thresh = 127
        for i in range(row):
            for j in range(col):
                b = src[i, j, 0]
                g = src[i, j, 1]
                r = src[i, j, 2]
                if alpha == 1:
                    if r > thresh:
                        temp[i, j, 2] = 255
                    else:
                        temp[i, j, 2] = 0
                    if g > thresh:
                        temp[i, j, 1] = 255
                    else:
                        temp[i, j, 1] = 0
                    if b > thresh:
                        temp[i, j, 0] = 255
                    else:
                        temp[i, j, 0] = 0
                    continue
                elif alpha >= 0:
                    newr = int(thresh + (r - thresh) / (1 - alpha))
                    newg = int(thresh + (g - thresh) / (1 - alpha))
                    newb = int(thresh + (b - thresh) / (1 - alpha))
                else:
                    newr = int(thresh + (r - thresh) * (1 + alpha))
                    newg = int(thresh + (g - thresh) * (1 + alpha))
                    newb = int(thresh + (b - thresh) * (1 + alpha))
                newr = max(0, min(255, newr))
                newg = max(0, min(255, newg))
                newb = max(0, min(255, newb))
                temp[i, j, 2] = newr
                temp[i, j, 1] = newg
                temp[i, j, 0] = newb
        self.Tmp_Image = temp
        self.update()

    def adjustBright(self, value):
        if self.Dest_Image is None:
            return
        # 亮度调整公式为：new_image = old_image + brightness_factor
        # brightness_factor = value * 2.55
        src = self.Dest_Image
        src = src.astype(np.float32)
        brightness_factor = value * 2.55
        new_image = src + brightness_factor
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        self.Tmp_Image = new_image
        self.update()

    def adjustWarm(self, value):
        if self.Dest_Image is None:
            return
        # 饱和度调整公式为：new_image = (old_image - gray_image) * saturation_factor + gray_image
        # saturation_factor = (value + 100) / 100
        # gray_image = 0.299 * R + 0.587 * G + 0.114 * B
        src = self.Dest_Image
        src = src.astype(np.float32)
        saturation_factor = (value + 100) / 100
        gray_image = 0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]
        new_image = (src - gray_image[:, :, np.newaxis]) * saturation_factor + gray_image[:, :, np.newaxis]
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        self.Tmp_Image = new_image
        self.update()

    def adjustSaturation(self, value):
        if self.Dest_Image is None:
            return
        src = self.Dest_Image
        # 饱和度调整公式为：new_image = (old_image - gray_image) * saturation_factor + gray_image
        # saturation_factor = (value + 100) / 100
        # gray_image = 0.299 * R + 0.587 * G + 0.114 * B
        src = src.astype(np.float32)
        saturation_factor = (value + 100) / 100
        gray_image = 0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]
        new_image = (src - gray_image[:, :, np.newaxis]) * saturation_factor + gray_image[:, :, np.newaxis]
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        self.Tmp_Image = new_image
        self.update()

    # 灰度化
    def toGray(self):
        if self.Dest_Image is None:
            return
        gray = cv2.cvtColor(self.Dest_Image, cv2.COLOR_BGR2GRAY)
        self.saveAndShow(gray)

    # 二值化
    def toBinary(self):
        if self.Dest_Image is None:
            return
        gray = cv2.cvtColor(self.Dest_Image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.saveAndShow(thresh1)

    # 反相
    def invert(self):
        if self.Dest_Image is None:
            return
        image = 255 - self.Dest_Image
        self.saveAndShow(image)

    # 浮雕
    def emboss(self):
        if self.Dest_Image is None:
            return
        src = self.Dest_Image
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        dst = cv2.filter2D(src, -1, kernel)
        self.saveAndShow(dst)

    # 模糊
    def blur(self):
        if self.Dest_Image is None:
            return
        src = self.Dest_Image
        dst = cv2.GaussianBlur(src, (0, 0), sigmaX=15)
        self.saveAndShow(dst)

    # 锐化
    def sharpen(self):
        if self.Dest_Image is None:
            return
        src = self.Dest_Image

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        dst = cv2.filter2D(src, -1, kernel)

        # blurImg = cv.GaussianBlur(src, (0, 0), 5)
        # usm = cv.addWeighted(src, 1.5, blurImg, -0.5, 0)
        self.saveAndShow(dst)

    # 边缘检测
    def canny(self):
        if self.Dest_Image is None:
            return
        blurred = cv2.GaussianBlur(self.Dest_Image, (3, 3), 0, 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.saveAndShow(edges)

    # 旋转
    def rotate(self):
        if self.Dest_Image is None:
            return
        image = self.Dest_Image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        self.saveAndShow(rotated_image)

    # 刷新
    def refresh(self):
        if self.Dest_Image is None:
            return
        self.saveAndShow(self.Image)

    # 证件矫正
    def correct(self):
        if self.Dest_Image is None:
            return
        img = correct.runCard(self.Dest_Image)
        self.saveAndShow(img)

    # 书页矫正
    def correct2(self):
        if self.Dest_Image is None:
            return
        img = correct.runPaper(self.Dest_Image)
        self.saveAndShow(img)


# 设置大小窗口
class ChildWin(QtWidgets.QMainWindow, setsize.Ui_Dialog):
    # 定义信号
    _signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super(ChildWin, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.pushButton.clicked.connect(self.slot1)

    def slot1(self):
        width = self.lineEdit.text()
        height = self.lineEdit_2.text()
        # 发送信号
        self._signal.emit([width, height])
        self.close()

    @property
    def signal(self):
        return self._signal


# 选择风格窗口
class SelectStyleDialog(QtWidgets.QMainWindow, SelectStyle.Ui_dialog):
    # 定义信号
    signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super(SelectStyleDialog, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.pushButton.clicked.connect(self.slot1)
        self.pushButton_2.clicked.connect(self.slot2)

    def slot1(self):
        self.signal.emit(1)
        self.close()

    def slot2(self):
        self.signal.emit(2)
        self.close()


# 选择颜色窗口
class ChangeColorDialog(QtWidgets.QDialog, ChangeColor.Ui_Dialog):
    # 定义信号
    signal_src = 'r'
    signal_dst = 'w'
    signal = QtCore.pyqtSignal(list)
    dialogRejected = QtCore.pyqtSignal()
    dialogAccepted = QtCore.pyqtSignal()

    def __init__(self):
        super(ChangeColorDialog, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.buttonBox.accepted.connect(self._dialogAccepted)
        self.buttonBox.rejected.connect(self._dialogRejected)

    def getValue(self):
        if self.checkBox.isChecked():
            self.signal_src = 'r'
        elif self.checkBox_2.isChecked():
            self.signal_src = 'b'
        elif self.checkBox_3.isChecked():
            self.signal_src = 'w'

        if self.checkBox_4.isChecked():
            self.signal_dst = 'r'
        elif self.checkBox_5.isChecked():
            self.signal_dst = 'b'
        elif self.checkBox_6.isChecked():
            self.signal_dst = 'w'
        self.signal.emit([self.signal_src, self.signal_dst])
        self.close()

    def _dialogAccepted(self):
        self.getValue()
        self.dialogAccepted.emit()

    def _dialogRejected(self):
        self.dialogRejected.emit()


# 图像调节窗口
class BaseAdjustDialog(QtWidgets.QDialog, Ui_baseAdjustDialog):
    brightSliderReleased = QtCore.pyqtSignal(object)
    warmSliderReleased = QtCore.pyqtSignal(object)
    saturabilitySliderReleased = QtCore.pyqtSignal(object)
    contrastSliderReleased = QtCore.pyqtSignal(object)

    dialogRejected = QtCore.pyqtSignal()
    dialogAccepted = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(BaseAdjustDialog, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.sliders = [self.brightSlider, self.saturabilitySlider, self.contrastSlider, self.warmSlider]
        self.sliderLabels = [self.brightLabel, self.saturabilityLabel, self.contrastLabel, self.warmLabel]
        self._establishConnections()

    def _establishConnections(self):
        self.dialogBtnBox.accepted.connect(self._dialogAccepted)
        self.dialogBtnBox.rejected.connect(self._dialogRejected)
        [self._buildSliderConnected(slider) for slider in self.sliders]
        self.brightSlider.sliderReleased.connect(self._brightSliderReleased)
        self.warmSlider.sliderReleased.connect(self._warmSliderReleased)
        self.saturabilitySlider.sliderReleased.connect(self._saturabilitySliderReleased)
        self.contrastSlider.sliderReleased.connect(self._contrastSliderReleased)

    def _buildSliderConnected(self, slider):
        slider.valueChanged.connect(partial(self._sliderValueChanged, slider))

    def _sliderValueChanged(self, slider):
        self.sliderLabels[self.sliders.index(slider)].setNum(slider.value())

    def _brightSliderReleased(self):
        brightValue = self.brightSlider.value()
        self.brightLabel.setNum(brightValue)
        self.brightSliderReleased.emit(brightValue)

    def _contrastSliderReleased(self):
        contrastValue = self.contrastSlider.value()
        self.contrastLabel.setNum(contrastValue)
        self.contrastSliderReleased.emit(contrastValue)

    def _warmSliderReleased(self):
        warmValue = self.warmSlider.value()
        self.warmLabel.setNum(warmValue)
        self.warmSliderReleased.emit(warmValue)

    def _saturabilitySliderReleased(self):
        saturationValue = self.saturabilitySlider.value()
        self.saturabilityLabel.setNum(saturationValue)
        self.saturabilitySliderReleased.emit(saturationValue)

    def _dialogAccepted(self):
        self.dialogAccepted.emit()

    def _dialogRejected(self):
        self.dialogRejected.emit()


app = QtWidgets.QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())

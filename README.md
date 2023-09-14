# 图像处理程序

这是一个基于opencv和pytorch的图像处理程序，实现了模糊、锐化、边缘检测等图像处理基本功能，提供了图像矫正等实际应用，以及基于深度学习的图像/视频风格迁移功能，采用pyqt5制作图形化用户界面。

## 项目背景

图像处理是计算机视觉领域的一个重要分支，它涉及到对图像进行各种操作，如滤波、增强、变换、分割、检测等，以达到改善图像质量、提取图像信息、增加图像效果等目的。图像处理的应用场景非常广泛，如医学影像、卫星遥感、人脸识别、数字水印等。

随着深度学习技术的发展，图像处理也出现了一些新的方法和应用，如风格迁移、超分辨率、图像生成等。这些方法利用深度神经网络来学习图像之间的映射关系或潜在特征，从而实现对图像的高级处理。

本项目是一个基于opencv和pytorch的图像处理程序，旨在结合传统的图像处理方法和深度学习方法，实现一些常见的图像处理功能和应用，并提供一个友好的图形化用户界面，方便用户使用。本项目可以作为一个学习和交流的平台，也可以作为一个实用的工具，为用户提供高效和高质量的图像处理服务。

## 功能特性

- 基本功能，包括图像缩放、图像调节、灰度化、二值化、反相、浮雕、模糊、锐化、边缘检测、旋转
- 实际应用，利用高斯滤波和开闭运算计算矩形顶点，通过透视变化进行图像矫正
- 高级功能，基于pytorch深度学习实现图像风格迁移，进一步改进后实现视频风格迁移
- 图形界面，集成了多种功能，可以方便地打开、处理、刷新、保存图片，并直观地展现原图和处理后图片的差别

## 环境依赖

- Python 3.9
- Anaconda

## 部署步骤

1. 克隆项目到本地

```bash
git clone ssh
```

2. 配置Anaconda及pytorch

参考以下博客：
https://blog.csdn.net/weixin_52836217/article/details/126674089

3. 安装依赖库

```bash
pip install -r requirements.txt
```

4. 运行主程序

```bash
python main.py
```

## 目录结构描述


- UI：存放QtDesigner设计的ui文件及转化后的py文件
- image：程序工作过程需要调用的图片
- original：原图
- sample：视频示例
- style：风格图片
- style.pth：训练后的风格模型
- main.py: 主程序入口文件
- README.md: 项目说明文件

## 贡献
- 华东师范大学 张宜添
- 华东师范大学 马鼎
- 华东师范大学 张国帅
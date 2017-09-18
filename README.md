# CrowdNet
本po是复现 [CrowdNet: A Deep Convolutional Network for Dense Crowd Counting](https://arxiv.org/abs/1608.06197) 所描述的CrowdNet在ShanghaiTech数据集上的训练结果。



## 数据集准备
下载地址：[百度云盘](http://pan.baidu.com/s/1gfyNBTh) 密码：p1rv

本实验只用了里面的part\_B的数据，预处理脚本是scripts/generate\_gt\_images.py，预处理后原图会被resize到256x256并保存在full\_crop\_images中，对应的ground truth图则保存在full\_gt\_images中。sample中有一个处理好的示例。


## 网络定义
train、inference和solver都在prototxt中，重新训练的话学习率需要自己调整

## 训练好的模型
在model下的CrowdNet_50000.caffemodel是一个训练好的模型


## 有用的脚本
一些可能有用的脚本都在scripts下，包括运行视频分析的run\_video.py、画loss曲线的plot\_info.py等。可能需要根据实际自行修改。

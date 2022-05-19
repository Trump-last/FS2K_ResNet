# 基于ResNet的素描人脸属性识别

## 数据集

链接：[FS2K 数据集](https://github.com/DengPingFan/FS2K)

## 模型简要描述

本项目分别使用了ResNet的三种变形结构，ResNet18，ResNet34，ResNet50，对数据集中的素描图像进行了属性识别。同时比较了最后分类的全连接层，使用softmax与sigmiod函数的差异，分别对两者做了实验。所以本项目总过进行了6次不同模型，或者相同模型不同函数的测试。

## 文件结构

### 初始

```
├─FS2K
│  ├─photo             # 源图片数据集
│  ├─sketch            # 源素描数据集
│  │  anno_test.json   # 测试json
│  │  anno_train.json  # 训练json
│  │  README.pdf
├─tools                # 数据集工具
│     ├─check.py
│     ├─split_train_test.py
│     ├─vis.py
└─Train                # 分类模型
      ├─config.py      # 配置文件
      ├─data.py        # 数据集处理
      ├─FS2K.py        # 运行主文件
      ├─model.py       # 模型文件
      ├─trainer.py     # 训练&测试
```

### 运行结束后

```
├─FS2K
│  ├─photo             # 源图片数据集
│  ├─sketch            # 源素描数据集
│  │  anno_test.json   # 测试json
│  │  anno_train.json  # 训练json
│  │  README.pdf
│  ├─test              # 测试集
│  │  ├─photo
│  │  └─sketch
│  └─train             # 训练集
│      ├─photo
│      └─sketch
├─result               # 结果文件夹
│  ├─resnet18
│  ├─resnet34
│  └─resnet50
├─tools                # 数据集工具
│     ├─check.py
│     ├─split_train_test.py
│     ├─vis.py
└─Train                # 分类模型
      ├─config.py      # 配置文件
      ├─data.py        # 数据集处理
      ├─FS2K.py        # 运行主文件
      ├─model.py       # 模型文件
      ├─trainer.py     # 训练&测试
```



## 快速开始

```python
python ./tools/split_train_test.py       #划分测试集与训练集
python ./Train/FS2K.py --model resnet50  #请输入resnet50, resnet34, resnet18其中一种模型
```

## 结果展示

### 不同模型的准确度

![](https://github.com/Trump-last/FS2K_ResNet/blob/main/picture/radar.png)

### 不同模型的每个epoch的损失变化

![](https://github.com/Trump-last/FS2K_ResNet/blob/main/picture/resnet.png)

### 相同模型不同分类函数损失变化

![](https://github.com/Trump-last/FS2K_ResNet/blob/main/picture/sig_sof.png)

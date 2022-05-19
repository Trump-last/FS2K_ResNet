'''
Autor: XLF
Date: 2022-05-16 21:57:54
LastEditors: XLF
LastEditTime: 2022-05-19 09:24:00
Description: 
    采用ResNet50进行特征提取，利用网络的卷积层的参数共享，通过不同的全连接层输出6种属性
'''
from unittest import case
import torch
import torch.nn as nn
from torchvision import models

class fc_make(nn.Module):
    '''
    全连接层，用于最后的输出判断
    '''
    def __init__(self, input_dim=2048, output_dim=2):
        super(fc_make, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        res = self.fc(x)
        res = torch.softmax(res, dim=-1)
        return res


class ResNet(nn.Module):
    def __init__(self,model_type = 'resnet50'):
        super(ResNet, self).__init__()
        if model_type == 'resnet50':
            self.FeatureNet = models.resnet50(pretrained=True)
        elif model_type == 'resnet34':
            self.FeatureNet = models.resnet34(pretrained=True)
        elif model_type == 'resnet18':
            self.FeatureNet = models.resnet18(pretrained=True)
        else:
            raise Exception('模型选择错误，请选择resnet50, resnet34, resnet18')
        self.FeatureNet = nn.Sequential(*list(self.FeatureNet.children())[:-1])
    def forward(self, x):
        return self.FeatureNet(x)


class MyNet(nn.Module):
    def __init__(self, model_type='resnet50'):
        super(MyNet, self).__init__()
        self.FeatureNet = ResNet(model_type)
        if model_type == 'resnet50':
            dim = 2048
        elif model_type == 'resnet34':
            dim = 512
        elif model_type == 'resnet18':
            dim = 512
        else:
            raise Exception('模型选择错误，请选择resnet50, resnet34, resnet18')
        self.hair_attr = fc_make(dim)
        self.gender_attr = fc_make(dim)
        self.earring_attr = fc_make(dim)
        self.smile_attr = fc_make(dim)
        self.face_attr = fc_make(dim)
        self.style_attr = fc_make(dim,output_dim=3)
    
    def forward(self, img):
        # 特征提取
        features = self.FeatureNet(img)
        # FC分类
        hair = self.hair_attr(features)
        gender = self.gender_attr(features)
        earring = self.earring_attr(features)
        smile = self.smile_attr(features)
        frontal = self.face_attr(features)
        style = self.style_attr(features)
        return hair, gender, earring, smile, frontal, style
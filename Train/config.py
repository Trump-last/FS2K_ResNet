'''
Autor: XLF
Date: 2022-05-17 09:19:18
LastEditors: XLF
LastEditTime: 2022-05-18 10:49:47
Description: 
    配置文件
'''
import os
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = r'../FS2K'
json_file_train = os.path.join(root_dir, 'anno_train.json')
json_file_test = os.path.join(root_dir, 'anno_test.json')
test_file = os.path.join(root_dir,'test')
train_file = os.path.join(root_dir,'train')

#选取图片与素描共有的属性，这样简单一点
feature_attr = ['hair', 'gender', 'earring', 'smile', 'frontal_face', 'style']

epoches = 50
batch_size = 10
lr = 1e-5

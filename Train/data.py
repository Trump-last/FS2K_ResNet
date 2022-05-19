'''
Autor: XLF
Date: 2022-05-16 21:57:50
LastEditors: XLF
LastEditTime: 2022-05-19 08:16:39
Description: 
'''
import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import torch.nn.functional as F

import config as cfg



def load_lable_img(tag = 0):
    '''
    返回标签集合
    tag = 0, 返回训练集与标签
    tag = 1, 返回测试集与标签
    默认是0，训练集与标签
    attrs: 特征集合列表
    photo_path：照片路径列表
    sketch_path：素描图片路径列表
    '''
    attrs = []
    photo_path = []
    sketch_path = []
    photo_root = ''
    sketch_root = ''
    if tag == 0:
        with open(cfg.json_file_train, 'r') as f:
            json_data = json.loads(f.read())
        photo_root = os.path.join(cfg.train_file,'photo')
        sketch_root = os.path.join(cfg.train_file,'sketch')
    else:
        with open(cfg.json_file_test, 'r') as f:
            json_data = json.loads(f.read())
        photo_root = os.path.join(cfg.test_file,'photo')
        sketch_root = os.path.join(cfg.test_file,'sketch') 
    for _, fs in enumerate(json_data):
        fearture = []
        single_photo = ''
        single_sketch = ''
        pho_str = fs['image_name'].replace('/', '_')+'.png'
        ske_str = fs['image_name'].replace('photo', 'sketch').replace('image', 'sketch').replace('/', '_')+'.png'
        single_photo = os.path.join(photo_root, pho_str)
        single_sketch = os.path.join(sketch_root, ske_str)
        for attr in cfg.feature_attr:
            fearture.append(fs[attr])
        attrs.append(fearture)
        photo_path.append(single_photo)
        sketch_path.append(single_sketch)
    return attrs, photo_path, sketch_path



def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))
        
class myDataset(Data.DataLoader):
    def __init__(self,transform=None,loader=default_loader,mode='train'):
        if mode == 'train':
            self.attrs, self.photo_path, self.sketch_path = load_lable_img(0)
        elif mode == 'test':
            self.attrs, self.photo_path, self.sketch_path = load_lable_img(1)
        else:
            raise Exception('模式选择错误，请选择train或者test')
        self.transform = transform
        self.loader = loader 

    def __len__(self):
        return len(self.attrs)
    
    def __getitem__(self,index):
        '''
        这里就先用素描测试
        '''
        img_path = self.sketch_path[index]
        label = torch.from_numpy(np.array(self.attrs[index], dtype=np.int64)) 
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))
        return img,label

transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                     std = [0.5,0.5,0.5])
                                ])

def get_data_load(mode = 'train'):
    dataset = myDataset(transform, mode=mode)
    data_loader = Data.DataLoader(dataset=dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=(mode == 'train'),
                                  drop_last=True)
    return data_loader

if __name__ == '__main__':
    k = get_data_load('test')
    print(len(k))
    for idx, data in enumerate(get_data_load('test')):
        q = torch.rand(10,3).to(cfg.device)
        img, labels = data
        if idx == 0:
            p = torch.tensor([[1,0] for i in range(10)]).to(cfg.device)
            _, l = torch.max(p,1)
            print(l)
            labels = labels.to(cfg.device)
            lk = labels[:,5] == l
            lsd = lk.cpu().numpy().tolist()
            num = {True:0, False:0}
            for key in lsd:
                num[key]= num.get(key,0) + 1
            print(num[True])
            z = F.cross_entropy(q, labels[:,5])
            print(z)
            mi = labels[:,5]
            o = [] 
            o += mi.cpu().numpy().tolist()
            o += mi.cpu().numpy().tolist()
            print(o)
'''
Autor: XLF
Date: 2022-05-17 19:59:05
LastEditors: XLF
LastEditTime: 2022-05-19 09:37:59
Description: 
'''


from trainer import FS2K_Trainer
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',help='请输入resnet50, resnet34, resnet18 其中一种模型')
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = FS2K_Trainer(args.model)
    trainer.FS2K_train()

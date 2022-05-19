'''
Autor: XLF
Date: 2022-05-18 10:11:09
LastEditors: XLF
LastEditTime: 2022-05-19 11:43:33
Description:
    训练与测试模块
'''


import torch,json,copy
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path


import config as cfg
import data
from model import MyNet

def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

class FS2K_Trainer():
    def __init__(self, model_type='resnet50') -> None:
        if model_type not in ['resnet50','resnet34','resnet18']:
            raise Exception('模型选择错误，请选择resnet50, resnet34, resnet18')
        self.epoches = cfg.epoches  # 迭代次数
        self.batch_size = cfg.batch_size  # 批处理大小
        self.lr = cfg.lr  # 学习率
        self.selected_attrs = cfg.feature_attr # 需要分类的特征集
        self.device = cfg.device # GPU or CPU
        self.model_type = model_type
        self.train_loader = data.get_data_load('train') # 加载训练数据
        self.test_loader = data.get_data_load('test')  # 加载测试数据
        self.model = MyNet(self.model_type).to(self.device) # 在GPU上建立模型
        self.optimer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimer, [20,40], gamma=0.1)


    def count(self, count_list):
        num = {True:0, False:0}
        count_num = count_list.cpu().numpy().tolist()
        for key in count_num:
            num[key] = num.get(key, 0) + 1
        return num
    
    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        batch_num = 0
        for batch_idx, train_data in enumerate(self.train_loader):
            images, labels = train_data
            images = Variable(images)
            labels = Variable(labels)
            images = images.to(self.device) # 将图片放到显卡上
            labels = labels.to(self.device) # 将标签放到显卡上
            hairs, genders, earrings, smiles, frontals, styles = self.model(images) #返回值应该是5个10*2的tensor，1个10*3的tensor
            hair_loss = F.cross_entropy(hairs, labels[:,0])
            gender_loss = F.cross_entropy(genders, labels[:,1])
            earring_loss = F.cross_entropy(earrings, labels[:,2])
            smile_loss = F.cross_entropy(smiles, labels[:,3])
            frontal_loss = F.cross_entropy(frontals, labels[:,4])
            style_loss = F.cross_entropy(styles, labels[:,5])
            total_loss =  hair_loss + gender_loss + earring_loss + smile_loss + frontal_loss + style_loss
            total_loss.backward()
            self.optimer.step()
            self.optimer.zero_grad()
            epoch_loss += total_loss.item()
            batch_num = batch_idx + 1
            if (batch_idx + 1) % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  in Train'.format(epoch, (batch_idx+1)*len(images), len(self.train_loader.dataset), 100.*(batch_idx+1)/len(self.train_loader), total_loss.item()))
        return epoch_loss/(batch_num)

    def test(self, epoch):
        self.model.eval()

        correct_count = {} # 每个属性的预测正确的数量
        pred_list = {} # 保存每个属性的预测值，以属性为分割，不是以图片做分割
        label_list = {} # 保存每个属性的标签值
        batch_num = 0
        
        for attr in self.selected_attrs:
            correct_count[attr] = 0
            pred_list[attr] = []
            label_list[attr] = []
        
        with torch.no_grad():
            for batch_idx, test_data in enumerate(self.test_loader):
                images, labels = test_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                hairs, genders, earrings, smiles, frontals, styles = self.model(images)
                out_list = {'hair':hairs,'gender':genders,'earring':earrings,'smile':smiles,'frontal_face':frontals,'style':styles}
                batch_num = batch_idx+1
                for idx,attr in enumerate(self.selected_attrs):
                    _, pred = torch.max(out_list[attr].data,1)
                    true_lable = labels[:,idx]
                    num = self.count(pred == true_lable)
                    correct_count[attr] += num[True]
                    pred_list[attr] += pred.cpu().numpy().tolist()
                    label_list[attr] += true_lable.cpu().numpy().tolist()
        mAP = 0
        for attr in self.selected_attrs:
            correct_count[attr] = 100. * correct_count[attr] / (self.batch_size * batch_num)
            mAP += correct_count[attr]
        mAP /= len(self.selected_attrs)
        return pred_list, label_list, correct_count,mAP

    def FS2K_train(self, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print("加载模型参数")
        best_ap = 0.0
        train_loss = []
        eval_acc_per_epoch = {}
        best_pred_list = {}
        best_label_list = {}
        best_correct_count = {}
        bset_wt = None
        make_dir('../result')
        if self.model_type == 'resnet50':
            make_dir('../result/resnet50')
        elif self.model_type == 'resnet34':
            make_dir('../result/resnet34')
        elif self.model_type == 'resnet18':
            make_dir('../result/resnet18')
        else:
            raise Exception('模型选择错误，请选择resnet50, resnet34, resnet18')

        for attr in self.selected_attrs:
            eval_acc_per_epoch[attr] = []

        for epoch in range(self.epoches):
            # 训练
            per_train_loss = self.train(epoch=epoch)
            print("Train Epoch: {}  loss: {:.6f}  lr: {:.7f}".format(epoch, per_train_loss, self.optimer.param_groups[0]['lr']))
            # 测试
            pred_list, label_list, correct_count,mAP = self.test(epoch=epoch)
            print("Test Epoch: {} accuracy:{}".format(epoch, correct_count))
            print("Test Epoch: {} mAP: {}".format(epoch, mAP))

            train_loss.append(per_train_loss)
            for attr in self.selected_attrs:
                eval_acc_per_epoch[attr].append(correct_count[attr])
            
            if mAP > best_ap:
                best_ap = mAP
                best_pred_list = pred_list
                best_label_list = label_list
                best_correct_count = correct_count
                bset_wt = copy.deepcopy(self.model.state_dict())
            self.scheduler.step()
        
        # 保存每个epoch的各属性正确率
        eval_acc_csv = pd.DataFrame(eval_acc_per_epoch, index=[i for i in range(self.epoches)])
        eval_acc_csv.to_csv(r"../result/" + self.model_type + "/eval_acc_per_epoch" + ".csv")
        # 保存训练过程的loss
        train_losses_csv = pd.DataFrame(train_loss)
        train_losses_csv.to_csv(r"../result/" + self.model_type + "/loss" + ".csv")
        # 保存best model
        model_save_path = r"../result/" + self.model_type + "/best_model" + ".pth"
        torch.save(bset_wt, model_save_path)
        print("The model has saved in {}".format(model_save_path))
        # 保存预测值
        pred_csv = pd.DataFrame(best_pred_list)
        pred_csv.to_csv(r"../result/"+ self.model_type + "/predict" + ".csv")
        # 保存真实值
        label_csv = pd.DataFrame(best_label_list)
        label_csv.to_csv(r"../result/"+ self.model_type + "/label" + ".csv")
        # 保存最优的预测
        label_csv = pd.DataFrame(best_correct_count, index=['0'])
        label_csv.to_csv(r"../result/"+ self.model_type + "/best_correct" + ".csv")
        # 保存这次训练的简要信息
        report_dict = {}
        report_dict["best_mAP"] = best_ap
        report_dict["lr"] = self.lr
        report_dict["optim"] = 'Adam'
        report_dict['Batch_size'] = self.batch_size
        report_json = json.dumps(report_dict)
        with open(r"../result/"+ self.model_type + "/report.json", 'w') as f:
             f.write(report_json)
        print("完成")

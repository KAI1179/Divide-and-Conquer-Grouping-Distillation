import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torchvision import models
from resnet_56 import *
import numpy as np
from models import *
import scipy.io as scio
import pandas as pd
import copy


# 分组算法 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

dataFile1 = './result/correct_test_vgg16.mat'
# dataFile1 = './result/correct_test_resnet56.mat'



data = scio.loadmat(dataFile1)
train = data['train']

super_class_1 = [4, 30, 55, 72, 95]
super_class_2 = [1, 32, 67, 73, 91]
super_class_3 = [54, 62, 70, 82, 92]
super_class_4 = [9, 10, 16, 28, 61]
super_class_5 = [0, 51, 53, 57, 83]
super_class_6 = [22, 39, 40, 86, 87]
super_class_7 = [5, 20, 25, 84, 94]
super_class_8 = [6, 7, 14, 18, 24]
super_class_9 = [3, 42, 43, 88, 97]
super_class_10 = [12, 17, 37, 68, 76]
super_class_11 = [23, 33, 49, 60, 71]
super_class_12 = [15, 19, 21, 31, 38]
super_class_13 = [34, 63, 64, 66, 75]
super_class_14 = [26, 45, 77, 79, 99]
super_class_15 = [2, 11, 35, 46, 98]
super_class_16 = [27, 29, 44, 78, 93]
super_class_17 = [36, 50, 65, 74, 80]
super_class_18 = [47, 52, 56, 59, 96]
super_class_19 = [8, 13, 48, 58, 90]
super_class_20 = [41, 69, 81, 85, 89]
# super_class = super_class_1 +  super_class_2 +  super_class_3 +  super_class_4 +  super_class_5 +  super_class_6 +\
# super_class_7 +  super_class_8 +  super_class_9 +  super_class_10 +  super_class_11 +  super_class_12 +  super_class_13 +\
# super_class_14 +  super_class_15 +  super_class_16 +  super_class_17 +  super_class_18 +  super_class_19 +  super_class_20

# print(train[0])
# print(len(train[0]))

for i in range(10000):
    if train[0][i] in super_class_1:
        train[0][i] = 100
    elif train[0][i] in super_class_2:
        train[0][i] = 101
    elif train[0][i] in super_class_3:
        train[0][i] = 102
    elif train[0][i] in super_class_4:
        train[0][i] = 103
    elif train[0][i] in super_class_5:
        train[0][i] = 104
    elif train[0][i] in super_class_6:
        train[0][i] = 105
    elif train[0][i] in super_class_7:
        train[0][i] = 106
    elif train[0][i] in super_class_8:
        train[0][i] = 107
    elif train[0][i] in super_class_9:
        train[0][i] = 108
    elif train[0][i] in super_class_10:
        train[0][i] = 109
    elif train[0][i] in super_class_11:
        train[0][i] = 110
    elif train[0][i] in super_class_12:
        train[0][i] = 111
    elif train[0][i] in super_class_13:
        train[0][i] = 112
    elif train[0][i] in super_class_14:
        train[0][i] = 113
    elif train[0][i] in super_class_15:
        train[0][i] = 114
    elif train[0][i] in super_class_16:
        train[0][i] = 115
    elif train[0][i] in super_class_17:
        train[0][i] = 116
    elif train[0][i] in super_class_18:
        train[0][i] = 117
    elif train[0][i] in super_class_19:
        train[0][i] = 118
    else:
        train[0][i] = 119
train[0] = train[0] - 100
# print(train[0])
for i in range(10000):
    if train[1][i] in super_class_1:
        train[1][i] = 100
    elif train[1][i] in super_class_2:
        train[1][i] = 101
    elif train[1][i] in super_class_3:
        train[1][i] = 102
    elif train[1][i] in super_class_4:
        train[1][i] = 103
    elif train[1][i] in super_class_5:
        train[1][i] = 104
    elif train[1][i] in super_class_6:
        train[1][i] = 105
    elif train[1][i] in super_class_7:
        train[1][i] = 106
    elif train[1][i] in super_class_8:
        train[1][i] = 107
    elif train[1][i] in super_class_9:
        train[1][i] = 108
    elif train[1][i] in super_class_10:
        train[1][i] = 109
    elif train[1][i] in super_class_11:
        train[1][i] = 110
    elif train[1][i] in super_class_12:
        train[1][i] = 111
    elif train[1][i] in super_class_13:
        train[1][i] = 112
    elif train[1][i] in super_class_14:
        train[1][i] = 113
    elif train[1][i] in super_class_15:
        train[1][i] = 114
    elif train[1][i] in super_class_16:
        train[1][i] = 115
    elif train[1][i] in super_class_17:
        train[1][i] = 116
    elif train[1][i] in super_class_18:
        train[1][i] = 117
    elif train[1][i] in super_class_19:
        train[1][i] = 118
    else:
        train[1][i] = 119
train[1] = train[1] - 100
# print(train)


tmp = [[], []]
# print(tmp.shape)
# tmp = tmp.tolist()
# print(tmp)

for i in range(20):
    for j in range(10000):
        if train[0][j] == i:
            tmp[0].append(train[0][j])
            tmp[1].append(train[1][j])
train = np.array(tmp)


x = np.zeros((20, 20), dtype=float)
y = np.zeros((20, 20), dtype=float)

for i in range(20):
    j = i + 1
    split_count = train[:, i*500:j*500][1, :]
    split_count = pd.Series(split_count)
    split_count = split_count.value_counts()
    split_count.sort_index(inplace=True)


    sc = dict(split_count)
    sum = 500 - sc[i]
    _ = sc.pop(i)
    for k in sc:
        sc[k] = sc[k] / sum

    for j in range(20):
        if sc.get(j) != None:
            x[i][j] = sc[j]


# print(x)
for i in range(20):
    for j in range(20):
        if i <= j:
            y[j][i] = x[i][j] + x[j][i]
y_tmp = copy.deepcopy(y) ## copy，y用来后续查表，y_tmp用来找初始分组.（这里需要注意深拷贝和浅拷贝）


## grouping
n = 20
m = 5
# s = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
s = [0] * n
for i in range(n):
    s[i] = i
out = {'class_1': [], 'class_2': [], 'class_3': [], 'class_4': [], 'class_5': []}
h = 0
for i in range(100):
    if h == 5:
        break
    aa = y_tmp.argmax()
    # print(aa)
    aa_x = aa // 20
    aa_y = aa % 20
    if (aa_x not in s) or (aa_y not in s):
        y_tmp[aa_x][aa_y] = 0.0
        continue

    class_name = 'class_' + str(h+1)
    out[class_name].append(aa_x)
    out[class_name].append(aa_y)
    s.remove(aa_x)
    s.remove(aa_y)
    y_tmp[aa_x][aa_y] = 0.0
    h = h + 1

for i in s:
    # print(i)
    # print(type(i))

    Xij = [0.0] * m

    for j in range(m):
        sum = 0.0
        count = 0
        class_name = 'class_' + str(j + 1)

        for k in out[class_name]:
            count += 1
            if k > i:
                tmp = y[k][i]
            else:
                tmp = y[i][k]  ## 行号必须大于列号（因为正三角矩阵）
            sum += tmp

        Xij[j] = sum / count
    class_name = 'class_' + str((Xij.index(max(Xij)))+1)
    # print(class_name)
    out[class_name].append(i)

for kk in range(m):
    class_name = 'class_' + str(kk+1)
    out[class_name].sort()

# print(out)
out_class = {'class_1': [], 'class_2': [], 'class_3': [], 'class_4': [], 'class_5': []}
for i in range(5):
    class_name = 'class_' + str(i + 1)
    for k in out[class_name]:
        if k == 0:
            out_class[class_name] += super_class_1
        elif k == 1:
            out_class[class_name] += super_class_2
        elif k == 2:
            out_class[class_name] += super_class_3
        elif k == 3:
            out_class[class_name] += super_class_4
        elif k == 4:
            out_class[class_name] += super_class_5
        elif k == 5:
            out_class[class_name] += super_class_6
        elif k == 6:
            out_class[class_name] += super_class_7
        elif k == 7:
            out_class[class_name] += super_class_8
        elif k == 8:
            out_class[class_name] += super_class_9
        elif k == 9:
            out_class[class_name] += super_class_10
        elif k == 10:
            out_class[class_name] += super_class_11
        elif k == 11:
            out_class[class_name] += super_class_12
        elif k == 12:
            out_class[class_name] += super_class_13
        elif k == 13:
            out_class[class_name] += super_class_14
        elif k == 14:
            out_class[class_name] += super_class_15
        elif k == 15:
            out_class[class_name] += super_class_16
        elif k == 16:
            out_class[class_name] += super_class_17
        elif k == 17:
            out_class[class_name] += super_class_18
        elif k == 18:
            out_class[class_name] += super_class_19
        elif k == 19:
            out_class[class_name] += super_class_20
# print(out_class)
out = out_class
for kk in range(m):
    class_name = 'class_' + str(kk+1)
    out[class_name].sort()

## 根据分组结果，确定学生网络的设计
# print(len(out['class_1']))
# print(len(out['class_2']))
# print(len(out['class_3']))
# print(len(out['class_4']))
# print(len(out['class_5']))
#
# exit()
## 分组算法完↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):
    def __init__(self, in_features, class_num):
        super(classifier, self).__init__()

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=in_features, out_features=class_num),
        )

        for m in self.classifier.children():
            m.apply(weights_init_classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


class small_network(nn.Module):
    def __init__(self, class_num=10):
        super(small_network, self).__init__()
        self.class_num = class_num

        self.base = resnet20()

        self.outblock = resnet20()
        self.outblock.conv1 = nn.Sequential()
        self.outblock.bn1 = nn.Sequential()
        self.outblock.layer1 = nn.Sequential()
        self.outblock.layer2 = nn.Sequential()
        self.outblock.linear = nn.Linear(64, 100)

        self.res_1 = resnet20()
        self.res_1.conv1 = nn.Sequential()
        self.res_1.bn1 = nn.Sequential()
        self.res_1.layer1 = nn.Sequential()
        self.res_1.layer2 = nn.Sequential()
        self.res_1.linear = nn.Sequential()

        self.res_2 = resnet20()
        self.res_2.conv1 = nn.Sequential()
        self.res_2.bn1 = nn.Sequential()
        self.res_2.layer1 = nn.Sequential()
        self.res_2.layer2 = nn.Sequential()
        self.res_2.linear = nn.Sequential()

        self.res_3 = resnet20()
        self.res_3.conv1 = nn.Sequential()
        self.res_3.bn1 = nn.Sequential()
        self.res_3.layer1 = nn.Sequential()
        self.res_3.layer2 = nn.Sequential()
        self.res_3.linear = nn.Sequential()

        self.res_4 = resnet20()
        self.res_4.conv1 = nn.Sequential()
        self.res_4.bn1 = nn.Sequential()
        self.res_4.layer1 = nn.Sequential()
        self.res_4.layer2 = nn.Sequential()
        self.res_4.linear = nn.Sequential()

        self.res_5 = resnet20()
        self.res_5.conv1 = nn.Sequential()
        self.res_5.bn1 = nn.Sequential()
        self.res_5.layer1 = nn.Sequential()
        self.res_5.layer2 = nn.Sequential()
        self.res_5.linear = nn.Sequential()


        self.block_group_1_classifier = nn.Sequential(  # group_1
            classifier(in_features=64, class_num=15),
        )
        self.block_group_2_classifier = nn.Sequential(  # group_2
            classifier(in_features=64, class_num=10),
        )
        self.block_group_3_classifier = nn.Sequential(  # group_3
            classifier(in_features=64, class_num=25),
        )
        self.block_group_4_classifier = nn.Sequential(  # group_4
            classifier(in_features=64, class_num=25),
        )
        self.block_group_5_classifier = nn.Sequential(  # group_5
            classifier(in_features=64, class_num=25),
        )

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)

        x_1 = self.res_1(x)
        x_1_out = self.block_group_1_classifier(x_1)

        x_2 = self.res_2(x)
        x_2_out = self.block_group_2_classifier(x_2)

        x_3 = self.res_3(x)
        x_3_out = self.block_group_3_classifier(x_3)

        x_4 = self.res_4(x)
        x_4_out = self.block_group_4_classifier(x_4)

        x_5 = self.res_5(x)
        x_5_out = self.block_group_5_classifier(x_5)

        preds = []

        class_1 = out['class_1']
        class_2 = out['class_2']
        class_3 = out['class_3']
        class_4 = out['class_4']
        class_5 = out['class_5']

        ## 计数各list下标情况
        num_1 = 0
        num_2 = 0
        num_3 = 0
        num_4 = 0
        num_5 = 0

        for i in range(100):
            if (i in class_1):
                preds.append(x_1_out[:, num_1].reshape([-1, 1]))
                num_1 += 1

            elif (i in class_2):
                preds.append(x_2_out[:, num_2].reshape([-1, 1]))
                num_2 += 1

            elif (i in class_3):
                preds.append(x_3_out[:, num_3].reshape([-1, 1]))
                num_3 += 1

            elif (i in class_4):
                preds.append(x_4_out[:, num_4].reshape([-1, 1]))
                num_4 += 1

            elif (i in class_5):
                preds.append(x_5_out[:, num_5].reshape([-1, 1]))
                num_5 += 1

        out_multi = torch.cat(preds, dim=1)

        out_sigle = self.outblock(x)

        return out_multi, out_sigle



if __name__ == '__main__':
    model = small_network()
    print(model)
    x = torch.randn([1, 3, 32, 32])
    y, _ = model(x)
    print(y.shape)
















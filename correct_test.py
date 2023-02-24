import torch
import torchvision
import torch.nn as nn
import numpy as np
from models import *
from resnet_56 import *
import torchvision.transforms as transforms
import os
import scipy.io as scio


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model...')
net = VGG('VGG16')
# net = VGG('VGG19')
# net = ResNet50()
# net = ResNet18()
# net = WRN40_2(100)
# net = resnet44()
# net.linear = nn.Linear(64, 100)


net.eval()
save_path = './checkpoint/vgg16/ckpt.pth'
# save_path = './checkpoint/resnet56/ckpt.pth'
# save_path = './checkpoint/WRN-40-2_1/ckpt.pth'
net.load_state_dict(torch.load(save_path)['net'])


ff = torch.FloatTensor(4, 11).zero_().cuda()
ff_flag = True


total = 0
correct = 0

error_logits = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # outputs = net(inputs)

        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)


        correct += predicted.eq(targets).sum().item()

    # epoch_loss = test_loss / (batch_idx + 1)
    epoch_acc = correct / total
    print('Test Acc: {:.4f}'.format( epoch_acc))

# exit()



with torch.no_grad():
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        # outputs = net(inputs)
        outputs = net(inputs)

        targets = targets.reshape([-1, 1])
        targets = targets.float()
        outputs = outputs.float()

        cat_mid = []
        cat_mid.append(targets)
        cat_mid.append(outputs)
        mid = torch.cat(cat_mid, dim=1)

        if ff_flag == True:
            cat_ff = []
            cat_ff.append(mid)
            ff_flag = False
        else:
            cat_ff = []
            cat_ff.append(mid)
            cat_ff.append(ff)

        ff = torch.cat(cat_ff, dim=0)



# print(ff.shape)

ff_numpy = ff.cpu().numpy()
print(ff_numpy.shape)
ff_numpy = ff_numpy[ff_numpy[:,0].argsort()]  # 排序
ff_numpy_predict = np.argmax(ff_numpy[:, 1:], axis=1)
tar_pred = np.vstack((ff_numpy[:,0], ff_numpy_predict))

# correct_test = './result/correct_test_resnet56.mat'
# correct_test = './result/correct_test_resnet50.mat'
correct_test = './result/correct_test_vgg16.mat'
scio.savemat(correct_test, {'train':tar_pred})





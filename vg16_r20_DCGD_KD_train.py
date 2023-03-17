'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from resnet_56 import *
from models import *
from vg16_r20_model import small_network


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--alpha', default=0.3, type=float, help='KD loss alpha')
parser.add_argument('--temperature', default=20, type=int, help='KD loss temperature')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = './checkpoint/vg16_r20_model'
save_path_pth = os.path.join(save_path, 'ckpt.pth')
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
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



# Model
print('==> Building model..')
# net = VGG('VGG16')
net = small_network()
net_teacher = VGG('VGG16')
# net_teacher = resnet56()
# net_teacher.linear = nn.Linear(64, 100)

net = net.cuda()
net_teacher = net_teacher.cuda()

# checkpoint = torch.load('./checkpoint/resnet56/ckpt.pth')
checkpoint = torch.load('./checkpoint/vgg16/ckpt.pth')
net_teacher.load_state_dict(checkpoint['net'])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    checkpoint = torch.load(save_path_pth)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
bce_WL = nn.BCEWithLogitsLoss()

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T) + \
              F.cross_entropy(outputs, labels) * alpha

    return KD_loss


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_sigle = 0
    correct_multi = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs_student_multi, outputs_student_sigle = net(inputs)
        outputs_teacher = net_teacher(inputs)

        loss_1 = loss_fn_kd(outputs_student_multi, targets, outputs_teacher, args.alpha, args.temperature)
        # loss_2 = loss_fn_kd(outputs_student_sigle, targets, outputs_teacher, args.alpha, args.temperature) ## for DCGD+KD
        loss_2 = criterion(outputs_student_sigle, targets) ## FOR dcgd+ce
        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted_sigle = outputs_student_sigle.max(1)
        total += targets.size(0)
        correct_sigle += predicted_sigle.eq(targets).sum().item()

        _, predicted_multi = outputs_student_multi.max(1)
        correct_multi += predicted_multi.eq(targets).sum().item()

    epoch_loss = train_loss / (batch_idx + 1)
    epoch_acc_sigle = correct_sigle / total
    epoch_acc_multi = correct_multi / total
    print('Train Loss: {:.4f}  single Acc: {:.4f}  multi Acc: {:.4f}'.format(epoch_loss, epoch_acc_sigle, epoch_acc_multi))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct_sigle = 0
    correct_multi = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # outputs = net(inputs)
            outputs_student_multi, outputs_student_sigle = net(inputs)
            loss = criterion(outputs_student_multi, targets)

            test_loss += loss.item()
            _, predicted_sigle = outputs_student_sigle.max(1)
            total += targets.size(0)
            correct_sigle += predicted_sigle.eq(targets).sum().item()

            _, predicted_multi = outputs_student_multi.max(1)
            correct_multi += predicted_multi.eq(targets).sum().item()



        epoch_loss = test_loss / (batch_idx + 1)
        epoch_acc_sigle = correct_sigle / total
        epoch_acc_multi = correct_multi / total
        print('Test Loss: {:.4f} single Acc: {:.4f}  multi Acc: {:.4f}'.format(epoch_loss, epoch_acc_sigle, epoch_acc_multi))



    # Save checkpoint.
    acc = 100.*epoch_acc_sigle/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path_pth)
        best_acc = acc


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+240):
        train(epoch)
        test(epoch)
        scheduler.step()

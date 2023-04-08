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
from vgg16_r20_model import small_network
# from _958_1_model import small_network_1
##！！！！！ 下面的也需要改 save_path = './checkpoint/_xxx_model'


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--alpha', default
=0.9, type=float, help='KD loss alpha')
parser.add_argument('--temperature', default=20, type=int, help='KD loss temperature')
# parser.add_argument('--model_name', action='_5_model',
#                     help='model name for save path')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = './checkpoint/dcgd_dkd_1_2_model_dkd'
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
# net = resnet56()
# net.linear = nn.Linear(64, 100)
net_teacher = VGG('VGG16')

net = net.cuda()
net_teacher = net_teacher.cuda()

checkpoint = torch.load('./checkpoint/resnet56/ckpt.pth')
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
              F.cross_entropy(outputs, labels)

    return KD_loss

def loss_fn_kd_crd(outputs, labels, teacher_outputs, alpha, temperature):
    p_s = F.log_softmax(outputs/temperature, dim=1)
    p_t = F.softmax(teacher_outputs/temperature, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temperature**2) / outputs.shape[0] + F.cross_entropy(outputs, labels) *0.3

    return loss
## DKD ↓
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

## DKD ↑

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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

        # loss = bce_WL(outputs_student, outputs_teacher)
        loss_1 = loss_fn_kd_crd(outputs_student_multi, targets, outputs_teacher, args.alpha, args.temperature)
        loss_2 = dkd_loss(outputs_student_sigle, outputs_teacher, targets, alpha=1.0, beta=4.0, temperature=args.temperature)
        loss_3 = criterion(outputs_student_sigle, targets)

        # loss = criterion(outputs_student, targets)
        loss = min((epoch + 1) / 20, 1.0) * (loss_1 + loss_2 + loss_3)
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

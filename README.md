# Divide-and-Conquer-Grouping-Distillation
This is a simple implementation of DCGD.

## Requirements

This is my experiment eviroument
- python3.7
- pytorch1.8.0+cu111
- tensorboard 2.9.0
- numpy 1.21.5
- pands 1.3.5


## Usage

### 1. dataset
I will use cifar100 dataset from torchvision.

### 2. train teacher network
```bash
$ python train.py 
```
Defult is vgg16.
Or you can down our pre-trained model at [here](https://1drv.ms/u/s!At1wX8TPqaH9fxSXr1TjIV4soAw?e=EkYWIM). Then unzip it to ```/checkpoint```

### 3. calculate MED matrix
You need to calculate MED matrix by 

```bash
$ python correct_test.py
```

Or you can use the file ```/result/correct_test_vgg16.mat``` provided in advance.

### 4. construct student model according to MED matrix

```bash
$ python vg16_r20_model.py
```

If you use another network (MED matrix), you need to change the file path loaded in the code.

### 5. train student

```bash
# for instance, DCGD+KD method.
$ python vg16_r20_DCGD_KD_train.py 

# for instance, DCGD+DKD method.
$ python vg16_r20_DCGD_DKD_train.py 
```

We provide two implementations in the file, ```DCGD+CE``` and ```DCGD+KD```. 


#

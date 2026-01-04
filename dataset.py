import os
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class OCTADataset(Dataset):
    
    def __init__(self, configs, type):
        super(OCTADataset, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(configs.data_dir, type)
        self.channel = configs.channel
        self.type = type
        self.name = ""
        self.aug = configs.if_aug
        self.patch_size = configs.patch_size
        self.rotate = 10
        self.flip_probability = 0.8
        self.resize = configs.resize
        self.centercrop = configs.centercrop
        
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3
    
    def __getitem__(self, index):
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        
        if self.resize is not None:
            simple_transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor()
            ])  
        else:
            simple_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
        
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        faz_gt = np.array(gt)
        faz_gt[faz_gt == 127] = 0
        faz_gt[faz_gt == 50] = 255
        faz_gt = Image.fromarray(faz_gt)

        rv_gt = np.array(gt)
        rv_gt[rv_gt == 50] = 0
        rv_gt[rv_gt == 127] = 255
        rv_gt = Image.fromarray(rv_gt)
        
        if self.type == 'train':
            if self.aug == True:
                if self.rotate is not None:
                    # rotate augumentation
                    angel = random.randint(-self.rotate, self.rotate)
                    img = img.rotate(angel)
                    faz_gt = faz_gt.rotate(angel)
                    rv_gt = rv_gt.rotate(angel)

                if random.random() < self.flip_probability:
                    # random horizontal flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    faz_gt = faz_gt.transpose(Image.FLIP_LEFT_RIGHT)
                    rv_gt = rv_gt.transpose(Image.FLIP_LEFT_RIGHT)

        img = simple_transform(img)
        faz_gt = simple_transform(faz_gt)
        rv_gt = simple_transform(rv_gt)
        
        return img, faz_gt, rv_gt
    
    def __len__(self):
        return len(self.img_lst)
    
    def get_dataPath(self, root, type):
        if type == 'train':
            img_dir = os.path.join(root + "/train/img")
            gt_dir = os.path.join(root + "/train/gt")
        elif type == 'valid':
            img_dir = os.path.join(root + "/val/img")
            gt_dir = os.path.join(root + "/val/gt")
        elif type == 'test':
            img_dir = os.path.join(root + "/test/img")
            gt_dir = os.path.join(root + "/test/gt")
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        
        return img_lst, gt_lst
    
    def getFileName(self):
        return self.name


def load_data(configs):
    train_dataset = OCTADataset(configs, type='train')
    valid_dataset = OCTADataset(configs, type='valid')
    test_dataset  = OCTADataset(configs, type='test')
    trainSet = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    validSet = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    testSet  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return trainSet, validSet, testSet

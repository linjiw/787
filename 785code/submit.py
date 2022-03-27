import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as ttf
from torch import Tensor

import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.init as init
from datetime import datetime
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models

import torch
# import torchvision.transforms as T
import wandb
from torch.nn.modules.loss import CrossEntropyLoss


class resnet18(models.ResNet):
    
    def _forward_impl(self, x: Tensor, return_feats=False) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        out = self.fc(feats)

        # if return_feats:
        #     return feats
        # else:
        #     return out
        return feats, out

        # return x
    def forward(self, x: Tensor,return_feats=False) -> Tensor:
        return self._forward_impl(x,return_feats=False)

# BasicBlock = models.resnet.BasicBlock

# Network = resnet18(BasicBlock, [2,2,2,2])
# Network.fc = nn.Linear(512,7000)


class resnet32(models.ResNet):
    
    def _forward_impl(self, x: Tensor, return_feats=False) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        out = self.fc(feats)

        if return_feats:
            return feats
        else:
            return out

        # return x
    def forward(self, x: Tensor,return_feats=False) -> Tensor:
        return self._forward_impl(x,return_feats=False)
# BasicBlock = models.resnet.BasicBlock

# Network = resnet32(BasicBlock, [3, 4, 6, 3])
# Network.fc = nn.Linear(512,7000)


# mobilenet = models.mobilenet_v3_large(pretrained=True)

class mobilenetv3(models.MobileNetV3):
    
    def _forward_impl(self, x: Tensor,return_feats= False ) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        if return_feats:
            return feats
        
        x = self.classifier(feats)

        return x

        # return x
    def forward(self, x: Tensor,return_feats=False) -> Tensor:
        return self._forward_impl(x,return_feats=False)
# # BasicBlock = models.resnet.BasicBlock
# arch = "mobilenet_v3_large"
# inverted_residual_setting, last_channel = models.mobilenetv3._mobilenet_v3_conf(arch)
# # mobilenet_v3_large
# model = mobilenetv3(inverted_residual_setting, last_channel)
# model.classifier = nn.Sequential(
#             nn.Linear(960, 1280),
#             nn.Hardswish(inplace=True),
#             nn.Dropout(p=0.3, inplace=True),
#             nn.Linear(1280, num_classes),
#         )


def getmodel(name='resnet18',num_classes = 7000):
    if name=='resnet18':
        BasicBlock = models.resnet.BasicBlock

        model = resnet18(BasicBlock, [2,2,2,2])
        model.fc = nn.Linear(512,7000)
    elif name=='resnet32':
        BasicBlock = models.resnet.BasicBlock

        model = resnet32(BasicBlock, [3, 4, 6, 3])
        model.fc = nn.Linear(512,7000)
    elif name=='mobilenetv3':
        # mobilenet = models.mobilenet_v3_large(pretrained=True)
        arch = "mobilenet_v3_large"
        inverted_residual_setting, last_channel = models.mobilenetv3._mobilenet_v3_conf(arch)
        # mobilenet_v3_large
        model = mobilenetv3(inverted_residual_setting, last_channel)
        model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, num_classes),
        )
    return model


if __name__=='__main__':
    print('start training')
    wandb.init(project="hw2pw", entity="linjiw")


    """
    The well-accepted SGD batch_size & lr combination for CNN classification is 256 batch size for 0.1 learning rate.
    When changing batch size for SGD, follow the linear scaling rule- halving batch size -> halve learning rate, etc.
    This is less theoretically supported for Adam, but in my experience, it's a decent ballpark estimate.
    """
    batch_size = 32
    lr = 0.03
    epochs = 200 # Just for the early submission. We'd want you to train like 50 epochs for your main submissions.
    num_classes = 7000
    wandb.config = {
    "learning_rate": 0.03,
    "epochs": 100,
    "batch_size": 32
    }


    
    """
    Transforms (data augmentation) is quite important for this task.
    Go explore https://pytorch.org/vision/stable/transforms.html for more details
    """
    DATA_DIR = "./"
    # TRAIN_DIR = osp.join(DATA_DIR, "train_subset/train_subset") # This is a smaller subset of the data. Should change this to classification/classification/train
    TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train")
    VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
    TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

    # train_transforms = [ttf.ToTensor(),ttf.RandomRotation((-90,90)),ttf.RandomAdjustSharpness(1.5),ttf.RandomHorizontalFlip(p=0.5),ttf.RandomVerticalFlip(p=0.5),ttf.RandomPerspective(distortion_scale=0.5,p=0.5) ] ,ttf.RandomVerticalFlip(p=0.2),
    # val_transforms = [ttf.ToTensor()],ttf.RandomRotation((-5,5),fill=20)
    train_transforms = [   ttf.ToTensor(),ttf.RandomHorizontalFlip(p=0.5),ttf.RandomInvert(p=0.01),ttf.RandomAffine(degrees=(-5,5),scale=(1,1.05)),ttf.ColorJitter(brightness=0.3,hue=0.2,contrast=0.3,saturation=0.2),ttf.Resize((224,224)),ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    val_transforms = [   ttf.ToTensor(),ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ]


    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                 transform=ttf.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                               transform=ttf.Compose(val_transforms))




    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, num_workers=4)

    model = getmodel(name='resnet18')
    # model = mobilenet
    # model.load_state_dict(torch.load("/content/drive/MyDrive/hw2p2/model/156_27_02_2022_21_08.pth")) best
    # model.load_state_dict(torch.load("/content/drive/MyDrive/hw2p2/model/best_mobilev3_onlyflip.pth"))
    # 23_09_03_2022_06_25 81
    # 9_10_03_2022_00_39 82
    # model.load_state_dict(torch.load("/content/drive/MyDrive/hw2p2/model/54_08_03_2022_22_45.pth")) #79.6
    # model.load_state_dict(torch.load("./model/23_09_03_2022_06_25.pth")) #81
    # model.load_state_dict(torch.load("./model/model1.pth"))
    # model.load_state_dict(torch.load("./787/model/0.8428%_84_10_03_2022_19_09.pth"))
    # model.load_state_dict(torch.load("./787/model/8542.pth"))
    # model.load_state_dict(torch.load("model/0.8239714285714286%_23_14_03_2022_21_34.pth"))
    # model/0.8209142857142857%_41_14_03_2022_23_39.pth
    # model.load_state_dict(torch.load("model/0.8209142857142857%_41_14_03_2022_23_39.pth"))
    # model.load_state_dict(torch.load("model/0.7978%_8_15_03_2022_01_42.pth")) #0.95841
    # model/0.8069428571428572%_17_15_03_2022_02_45.pth
    model.load_state_dict(torch.load("model/0.8172857142857143%_0_15_03_2022_05_52.pth")) #0.95841
    
    model.cuda()

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    # TODO: What criterion do we use for this task?
    # criterion = 
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
    # T_max is "how many times will i call scheduler.step() until it reaches 0 lr?"

    # For this homework, we strongly strongly recommend using FP16 to speed up training.
    # It helps more for larger models.
    # Go to https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
    # and compare "Single precision training" section with "Mixed precision training" section


    scaler = torch.cuda.amp.GradScaler()


    # wandb.watch(model, log_freq=100)
    
    class ClassificationTestSet(Dataset):
        # It's possible to load test set data using ImageFolder without making a custom class.
    # See if you can think it through!

        def __init__(self, data_dir, transforms):
            self.data_dir = data_dir
            self.transforms = transforms

            # This one-liner basically generates a sorted list of full paths to each image in data_dir
            self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

        def __len__(self):
            return len(self.img_paths)
    
    
        def __getitem__(self, idx):
            return self.transforms(Image.open(self.img_paths[idx]))
test_dataset = ClassificationTestSet(TEST_DIR, ttf.Compose(val_transforms))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=1)
model.eval()
batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

res = []
for i, (x) in enumerate(test_loader):
    x = x.cuda()
    # TODO: Finish predicting on the test set.
    with torch.no_grad():
        _,outputs = model(x)
    pred_y = torch.argmax(outputs, axis=1)
    res.extend(pred_y.tolist())

    batch_bar.update()
    
batch_bar.close()

with open("classification_early_submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))
        
        
# !kaggle competitions submit -c 11-785-s22-hw2p2-classification -f classification_early_submission.csv -m "hello"

class VerificationDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # We return the image, as well as the path to that image (relative path)
        return self.transforms(Image.open(self.img_paths[idx])), osp.relpath(self.img_paths[idx], self.data_dir)
    
    
test_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/test"),
                                        ttf.Compose(val_transforms))
test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=1)

model.eval()

feats_dict = dict()
for batch_idx, (imgs, path_names) in tqdm(enumerate(test_ver_loader), total=len(test_ver_loader), position=0, leave=False):
    imgs = imgs.cuda()

    with torch.no_grad():
        # Note that we return the feats here, not the final outputs
        # Feel free to try to final outputs too!
        feats,_ = model(imgs) 
        # feats = model(imgs, return_feats=True) 
        for i,j in zip(feats,path_names):
            feats_dict.update({j: i})
    # TODO: Now we have features and the image path names. What to do with them?
    # Hint: use the feats_dict somehow.
    
# We use cosine similarity between feature embeddings.
# TODO: Find the relevant function in pytorch and read its documentation.
# similarity_metric = 
from scipy.spatial.distance import cosine
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


similarity_metric = cosine
similarity_metric = cos
val_veri_csv = osp.join(DATA_DIR, "verification/verification/verification_test.csv")


# Now, loop through the csv and compare each pair, getting the similarity between them
pred_similarities = []
for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=False): # skip header
    img_path1, img_path2 = line.split(",")
    feats1 = feats_dict[img_path1[5:]]
    feats2 = feats_dict[img_path2[5:]]

    pred = float(similarity_metric(feats1,feats2))
    pred_similarities.append(pred)
    # TODO: Finish up verification testing.
    # How to use these img_paths? What to do with the features?
pred_similarities = np.array(pred_similarities)

with open("verification_early_submission.csv", "w+") as f:
    f.write("id,match\n")
    for i in range(len(pred_similarities)):
        f.write("{},{}\n".format(i, pred_similarities[i]))
        
# !kaggle competitions submit -c 11-785-s22-hw2p2-verification -f verification_early_submission.csv -m "test"
# kaggle competitions submit -c 11-785-s22-hw2p2-classification -f classification_early_submission.csv -m "hello"
    
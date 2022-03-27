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
from centerloss import CenterLoss
import torch
# import torchvision.transforms as T
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from scipy.spatial.distance import cosine

# wandb.init(project="hw2pw", entity="linjiw")


# """
# The well-accepted SGD batch_size & lr combination for CNN classification is 256 batch size for 0.1 learning rate.
# When changing batch size for SGD, follow the linear scaling rule - halving batch size -> halve learning rate, etc.
# This is less theoretically supported for Adam, but in my experience, it's a decent ballpark estimate.
# """
# batch_size = 256
# lr = 0.05
# epochs = 200 # Just for the early submission. We'd want you to train like 50 epochs for your main submissions.
# num_classes = 7000
# wandb.config = {
#   "learning_rate": 0.05,
#   "epochs": 100,
#   "batch_size": 256
# }


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

def calauc(feats_dict):
    # We use cosine similarity between feature embeddings.
# TODO: Find the relevant function in pytorch and read its documentation.
# similarity_metric = 
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)


    similarity_metric = cosine
    similarity_metric = cos


    val_veri_csv = osp.join("./", "verification/verification/verification_dev.csv")


    # Now, loop through the csv and compare each pair, getting the similarity between them
    pred_similarities = []
    gt_similarities = []
    for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=False): # skip header
        img_path1, img_path2, gt = line.split(",")

        # TODO: Use the similarity metric
        # How to use these img_paths? What to do with the features?
        # similarity = similarity_metric(...)
        # feats1 = feats_dict[img_path1[4:]].cpu().numpy().reshape((-1,1))
        # feats2 = feats_dict[img_path2[4:]].cpu().numpy().reshape((-1,1))
        # feats1 = 
        # print(feats1.shape)
        feats1 = feats_dict[img_path1[4:]]
        feats2 = feats_dict[img_path2[4:]]

        pred = float(similarity_metric(feats1,feats2))
    # TODO: Use the similarity metric
    # How to use these img_paths? What to do with the features?
    # similarity = similarity_metric(...)
  
    # pred_score = 0
    # if pred>0.5:
    #     pred_score = 0
    # else:
    #     pred_score = 1

        gt_similarities.append(int(gt))
        pred_similarities.append(pred)

    pred_similarities = np.array(pred_similarities)
    gt_similarities = np.array(gt_similarities)
    auc = roc_auc_score(gt_similarities, pred_similarities)
    print("AUC:", auc)
    return auc


if __name__=='__main__':
    print('start training')
    wandb.init(project="hw2pw", entity="linjiw")


    """
    The well-accepted SGD batch_size & lr combination for CNN classification is 256 batch size for 0.1 learning rate.
    When changing batch size for SGD, follow the linear scaling rule- halving batch size -> halve learning rate, etc.
    This is less theoretically supported for Adam, but in my experience, it's a decent ballpark estimate.
    """
    batch_size = 512
    lr = 0.05
    epochs = 100 # Just for the early submission. We'd want you to train like 50 epochs for your main submissions.
    num_classes = 7000
    wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size
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
    train_transforms = [   ttf.ToTensor(),ttf.RandomHorizontalFlip(p=0.5),ttf.RandomInvert(p=0.01),ttf.RandomAffine(degrees=(-10,10),scale=(1,1.05)),ttf.ColorJitter(brightness=0.3,hue=0.2,contrast=0.3,saturation=0.2),ttf.Resize((224,224)),ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    val_transforms = [   ttf.ToTensor(),ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ]


    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                 transform=ttf.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                               transform=ttf.Compose(val_transforms))

    val_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/dev"),
                                       ttf.Compose(val_transforms))
    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=1)


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
    model.load_state_dict(torch.load("model/0.7852857142857143%_6_15_03_2022_05_26.pth"))

    
    model.cuda()

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    # TODO: What criterion do we use for this task?
    # criterion = 
    criterion = CrossEntropyLoss()
    
    center_loss = CenterLoss(7000, 512,True)
    loss_weight = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimzer_center =optim.SGD(center_loss.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
    # T_max is "how many times will i call scheduler.step() until it reaches 0 lr?"

    # For this homework, we strongly strongly recommend using FP16 to speed up training.
    # It helps more for larger models.
    # Go to https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
    # and compare "Single precision training" section with "Mixed precision training" section


    scaler = torch.cuda.amp.GradScaler()


    wandb.watch(model, log_freq=100)

    for epoch in range(epochs):
        # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

        num_correct = 0
        total_loss = 0
        model.train()
        centerloss_sum = 0
        softloss_sum = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.cuda()

        # Don't be surprised - we just wrap these two lines to make it work for FP16
            with torch.cuda.amp.autocast():     
                feats,outputs = model(x)
                softloss = criterion(outputs, y)
                centerloss = loss_weight*center_loss(feats.float(),y.float())
                # val_loss = softloss+centerloss
                loss = centerloss+softloss
                # loss = criterion(outputs, y) +loss_weight*center_loss(feats.float(),y.float())
            # loss+= loss_weight*center_loss(feats,y)
        # Update # correct & loss as we go
            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            total_loss += float(loss)
            centerloss_sum +=centerloss
            softloss_sum += softloss
        # tqdm lets you add some details so you can monitor training as you train.
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
                centerloss_sum= "{:.04f}".format(float(centerloss_sum / (i + 1))),
                softloss_sum= "{:.04f}".format(float(softloss_sum / (i + 1))))
        
        # Another couple things you need for FP16. 
            scaler.scale(loss).backward() # This is a replacement for loss.backward()
            scaler.step(optimizer) # This is a replacement for optimizer.step()
            scaler.update() # This is something added just for FP16

            scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.
            optimzer_center.zero_grad()
            optimzer_center.step()
            batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar
        now = datetime.now()
        # pth = "/content/drive/MyDrive/11785/model/"
        name = now.strftime("%d_%m_%Y_%H_%M")
        
        # You can add validation per-epoch here if you would like
 
        model.eval()
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
        val_num_correct = 0
        val_loss_sum = 0
        for i, (x, y) in enumerate(val_loader):

            x = x.cuda()
            y = y.cuda()

            with torch.no_grad():
                feats,outputs = model(x)
                softloss = criterion(outputs, y)
                centerloss = loss_weight*center_loss(feats,y)
                val_loss = softloss+centerloss
            
            val_loss_sum+=val_loss
            val_num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            batch_bar.set_postfix(acc="{:.04f}%".format(100 * val_num_correct / ((i + 1) * batch_size)))

            batch_bar.update()
    
        batch_bar.close()
        print("Validation: {:.04f}%, centerloss: {:.04f}, softloss: {:.04f}".format(100 * val_num_correct / len(val_dataset),centerloss,softloss))
        val_acc = val_num_correct / len(val_dataset)
        wandb.log({"val_loss": float(val_loss_sum / len(train_loader))})
        wandb.log({"val acc": val_acc})
        wandb.log({"softloss": softloss})
        wandb.log({"centerloss": centerloss})

        wandb.log({"lr": float(optimizer.param_groups[0]['lr'])})
        
        torch.save(model.state_dict(), f"./model/{val_acc}%_{epoch}_{name}.pth")
        
        feats_dict = dict()
        
        for batch_idx, (imgs, path_names) in tqdm(enumerate(val_ver_loader), total=len(val_ver_loader), position=0, leave=False):
            imgs = imgs.cuda()

            with torch.no_grad():
            # Note that we return the feats here, not the final outputs
            # Feel free to try the final outputs too!
                feats,_ = model(imgs) 
                for i,j in zip(feats,path_names):
                    feats_dict.update({j: i})
                    
        auc = calauc(feats_dict)
        wandb.log({"auc": auc})
    
        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            epochs,
            100 * num_correct / (len(train_loader) * batch_size),
        
            float(total_loss / len(train_loader)),
            float(optimizer.param_groups[0]['lr'])))

import torch
from torch import nn

class DistillModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cat_feature = nn.Conv2d(6, 3, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(3)
        self.act_1 = nn.ReLU(True)

        self.conv_feature = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(3)
        self.act_2 = nn.ReLU(True)

        self.loss = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x_deep, x_shallow):
        x_cat = torch.cat([x_deep, x_shallow], dim=1)
        feature_cat = self.conv_cat_feature(x_cat)
        feature_cat = self.bn_1(feature_cat)
        feature_cat = self.act_1(feature_cat)

        feature_shallow = self.conv_feature(x_shallow)
        feature_shallow = self.bn_2(feature_shallow)
        feature_shallow = self.act_2(feature_shallow)

        loss = self.loss(feature_cat, feature_shallow)
        loss = loss.to(self.device)
        
        return loss, feature_cat
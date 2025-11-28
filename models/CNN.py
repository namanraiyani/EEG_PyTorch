import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=4, keep_prob=0.75):
        super(CNN, self).__init__()
        self.keep_prob = keep_prob
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(15 * 15 * 128, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        
        h_conv1 = self.conv1(x)
        h_conv1_acti = F.leaky_relu(h_conv1)
        h_conv1_drop = F.dropout(h_conv1_acti, p=1-self.keep_prob, training=self.training)
        
        h_conv2 = self.conv2(h_conv1_drop)
        h_conv2_bn = self.bn2(h_conv2)
        h_conv2_acti = F.leaky_relu(h_conv2_bn)
        
        h_conv3_res = torch.cat([h_conv2_acti, h_conv1_drop], dim=1)
        h_conv3 = self.conv3(h_conv3_res)
        h_conv3_acti = F.leaky_relu(h_conv3)
        h_conv3_drop = F.dropout(h_conv3_acti, p=1-self.keep_prob, training=self.training)
        
        h_pool3 = F.max_pool2d(h_conv3_drop, kernel_size=2, stride=2, padding=0)
        
        h_conv4 = self.conv4(h_pool3)
        h_conv4_bn = self.bn4(h_conv4)
        h_conv4_acti = F.leaky_relu(h_conv4_bn)
        h_conv4_drop = F.dropout(h_conv4_acti, p=1-self.keep_prob, training=self.training)
        
        h_conv5 = self.conv5(h_conv4_drop)
        h_conv5_bn = self.bn5(h_conv5)
        h_conv5_acti = F.leaky_relu(h_conv5_bn)
        
        h_conv6_res = torch.cat([h_conv5_acti, h_conv4_drop], dim=1)
        h_conv6 = self.conv6(h_conv6_res)
        h_conv6_acti = F.leaky_relu(h_conv6)
        h_conv6_drop = F.dropout(h_conv6_acti, p=1-self.keep_prob, training=self.training)
        
        h_pool6 = F.max_pool2d(h_conv6_drop, kernel_size=2, stride=2, padding=0)
        
        h_pool6_flat = h_pool6.view(-1, 15 * 15 * 128)
        
        h_fc1 = self.fc1(h_pool6_flat)
        h_fc1_bn = self.bn_fc1(h_fc1)
        h_fc1_acti = F.leaky_relu(h_fc1_bn)
        h_fc1_drop = F.dropout(h_fc1_acti, p=1-self.keep_prob, training=self.training)
        
        prediction = self.fc2(h_fc1_drop)
        
        return prediction

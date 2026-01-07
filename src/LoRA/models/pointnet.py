import torch
from torch import nn
from torch.nn import functional as F
from src.LoRA.models.utils import *

# set precision to what lightning suggests for this gpu
torch.set_float32_matmul_precision('high')
# make results reproducible


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim, device, h_dim=1024):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim
        self.device = device
        self.h_dim=h_dim

        self.conv_ft1 = nn.Conv1d(input_dim, 64, 1, bias=False)
        self.conv_ft2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_ft3 = nn.Conv1d(128, h_dim, 1, bias=False)

        self.bn_ft1 = nn.BatchNorm1d(64)
        self.bn_ft2 = nn.BatchNorm1d(128)
        self.bn_ft3 = nn.BatchNorm1d(h_dim)
        self.bn_ft4 = nn.BatchNorm1d(512)
        self.bn_ft5 = nn.BatchNorm1d(256)

        self.fc_ft1 = nn.Linear(h_dim, 512)
        self.fc_ft2 = nn.Linear(512, 256)
        self.fc_ft3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_ft1(self.conv_ft1(x)))
        x = F.relu(self.bn_ft2(self.conv_ft2(x)))
        x = F.relu(self.bn_ft3(self.conv_ft3(x)))  # [batch, 1024, 4096]

        x = nn.MaxPool1d(num_points)(x)  # [batch, 1024, 1] kernel_size = num_points (4096)
        x = x.view(-1, self.h_dim)
        x = F.relu(self.bn_ft4(self.fc_ft1(x)))
        x = F.relu(self.bn_ft5(self.fc_ft2(x)))
        x = self.fc_ft3(x)

        identity_matrix = torch.eye(self.output_dim)
        identity_matrix = identity_matrix.to(self.device)

        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        
        return x


class BasePointNet(nn.Module):

    def __init__(self, 
                 point_dimension=3,
                 return_local_features=False,
                 device='cpu',
                 channels_in=3):
        super(BasePointNet, self).__init__()
        
        self.point_dimension = point_dimension
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension, device=device)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)

        self.conv_1 = nn.Conv1d(channels_in, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, 1024, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        num_points = x.shape[1]  # torch.Size([BATCH, N_POINTS, DIMS])
        
        x_tnet = x[:, :, :self.point_dimension]  # [32, 4096, 3]
        input_transform = self.input_transform(x_tnet)  # [32, 3, 3]
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product
        
        x_tnet = torch.cat([x_tnet, x[:, :, 3:]], dim=2)  # concat other features if any
        x_tnet = x_tnet.transpose(2, 1)  # [batch, dims, n_points]

        x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        # weights conv_1 shape (64, 3, 1)
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 64, N_POINTS] # todo visualitzar weights son tots iguals?
        # weights conv_2 shape (64, 64, 1)
        
        # Visualize the weights of self.conv_2
        # self.visualize_conv2_weights()
        
        x = x.transpose(2, 1)  # [batch, 2000, 64]
        
        # Apply T-Net
        feature_transform = self.feature_transform(x)  # [batch, 64, 64]

        x = torch.bmm(x, feature_transform)
        local_point_features = x  # [batch, 2000, 64]

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, 1024)  # [ batch, 1024, 1]

        if self.return_local_features:
            global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform
    

class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3, device=''):
        super(ClassificationPointNet, self).__init__()

        self.base_pointnet = BasePointNet(return_local_features=False, point_dimension=point_dimension,
                                          device=device)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

        self.bn_21 = nn.BatchNorm1d(512)
        self.bn_22 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        global_feature, feature_transform = self.base_pointnet(x)

        x = F.relu(self.bn_21(self.fc_1(global_feature)))
        x = F.relu(self.bn_22(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.classifier(x), dim=1), feature_transform


class SegmentationPointNet(nn.Module):

    def __init__(self, num_classes,
                 point_dimension=3,
                 device='cuda',
                 return_local_features=True,
                 is_barlow=False,
                 channels_in=9):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_local_features=return_local_features,
                                          point_dimension=point_dimension,
                                          device=device,
                                          is_barlow=is_barlow,
                                          channels_in=channels_in)

        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        local_global_features, feature_transform = self.base_pointnet(x)
        x = local_global_features
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.conv_4(x)
        # x = x.transpose(2, 1)
        # return F.log_softmax(x, dim=-1), feature_transform
        return x, feature_transform

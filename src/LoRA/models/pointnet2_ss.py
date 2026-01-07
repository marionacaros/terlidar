import torch.nn as nn
import torch.nn.functional as F
from src.LoRA.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

EMBED_DIM = 512 


class PointNetAbstractionEncoder(nn.Module):
    """
    Features of points (HAG,I,G,B,NDVI) + coordinates x, y, z after FPS are used in each setabstraction network
    """
    def __init__(self, group_all, num_feat, rad):
        super(PointNetAbstractionEncoder, self).__init__()
        self.num_feat = num_feat
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=rad[0], nsample=32, in_channel= self.num_feat + 3, mlp=[32, 32, 64], 
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(256, rad[1], 32, 64 + 3, [64, 64, 128], False)  # concat x,y,z
        self.sa3 = PointNetSetAbstraction(64, rad[2], 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, rad[3], 32, 256 + 3, [256, 256, EMBED_DIM], group_all)

    def forward(self, pc):
        l0_points =  pc
        l0_xyz = pc[:, :3, :]
        # TODO PLOT RESULTING POINTS!!
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [b, 3, 1024] [b, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [b, 3 ,256] [b, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [b, 3, 64] [b, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [b, 3, 16] [b, 512, 16]

        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points


class PointNet2(nn.Module):

    def __init__(self, num_classes, group_all=False, num_feat=5, radius=[0.1, 0.2, 0.4, 0.8]):
        super(PointNet2, self).__init__()

        self.encoder = PointNetAbstractionEncoder(group_all=group_all, num_feat=num_feat, rad=radius)

        self.fp4 = PointNetFeaturePropagation(256 + EMBED_DIM, [256, 256])  # 768 channels
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.classifier = nn.Conv1d(128, num_classes, 1)

    def forward(self, pc):
        l0_xyz = pc[:, :3, :]
        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points = self.encoder(pc)

        l4_points = l4_points.view(-1, EMBED_DIM, 16)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [b, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [b, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [b, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [b, 128, 4096]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # [b, 128, 4096]
        x = self.classifier(x)  # [b, n_classes, 4096]
        x = F.log_softmax(x, dim=1)  # [b, n_classes, 4096]
        x = x.permute(0, 2, 1)
        
        # l4 -> transf feature # changed l4_points for None
        return x, None


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

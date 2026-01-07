
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F
import math
from src.LoRA.models.pointnet2_utils import *
from src.LoRA.models.utils import *


class LoraPointNet2(nn.Module):
    """
    Lora applied to PointNet++
    """

    def __init__(self,
                 num_classes=5,
                 num_feat=5,
                 lora_fix_rank=False,
                 lora_max_rank=64,
                 lora_min_rank=16,
                 alpha=1,
                 radius=[0.1, 0.2, 0.4, 0.8] 
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_feat = num_feat
        self.fixed_rank = lora_fix_rank
        self.radius = radius

        # Define rank constraints
        self.min_rank = lora_min_rank
        self.max_rank = lora_max_rank
        self.lora_alpha = alpha # lora scaling factor    

        min_rank = self.min_rank
        max_rank = self.max_rank

        # --------------------------------- PointNetAbstractionEncoder --------------------------------------------------
        
        # PointNetSetAbstraction 1 (npoint=1024, radius=0.1, nsample=32, in_channel= self.num_feat + 3, mlp=[32, 32, 64])
        self.mlp_convs_1 = nn.ModuleList()
        self.mlp_bns_1 = nn.ModuleList()
        last_channel = self.num_feat + 3 # CHNAGED 6 TO 3
        for out_channel in [32, 32, 64]: 
            self.mlp_convs_1.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns_1.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # PointNetSetAbstraction 2 (256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.mlp_convs_2 = nn.ModuleList()
        self.mlp_bns_2 = nn.ModuleList()
        last_channel = 64 + 3
        for out_channel in [64, 64, 128]:
            self.mlp_convs_2.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns_2.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        # PointNetSetAbstraction 3 (64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.mlp_convs_3 = nn.ModuleList()
        self.mlp_bns_3 = nn.ModuleList()
        last_channel = 128 + 3
        for out_channel in [128, 128, 256]:
            self.mlp_convs_3.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns_3.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        # PointNetSetAbstraction 4 (16, 0.8, 32, 256 + 3, [256, 256, 512], group_all)
        self.mlp_convs_4 = nn.ModuleList()
        self.mlp_bns_4 = nn.ModuleList()
        last_channel = 256 + 3
        for out_channel in [256, 256, 512]:
            self.mlp_convs_4.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns_4.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        # --------------------------------- Feature Propagation ------------------------------------
        # self.fp4 = PointNetFeaturePropagation(256 + 512, [256, 256])  # 768 channels
        self.mlp_convs_fp4 = nn.ModuleList()
        self.mlp_bns_fp4 = nn.ModuleList()
        last_channel = 256 + 512
        for out_channel in [256, 256]:
            self.mlp_convs_fp4.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns_fp4.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.mlp_convs_fp3 = nn.ModuleList()
        self.mlp_bns_fp3 = nn.ModuleList()
        last_channel = 384
        for out_channel in [256, 256]:
            self.mlp_convs_fp3.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns_fp3.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.mlp_convs_fp2 = nn.ModuleList()
        self.mlp_bns_fp2 = nn.ModuleList()
        last_channel = 320
        for out_channel in [256, 128]:
            self.mlp_convs_fp2.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns_fp2.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.mlp_convs_fp1 = nn.ModuleList()
        self.mlp_bns_fp1 = nn.ModuleList()
        last_channel = 128
        for out_channel in [128, 128, 128]:
            self.mlp_convs_fp1.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns_fp1.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel    

        # Classification
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.lora_classifier = nn.Conv1d(128, num_classes, 1)


        # --------------------------- Define lora hyperparameters ------------------------------
  
        # Layers sizes
        layers=[(self.num_feat + 3, 32), (32, 32), (32, 64), (67, 64), (64, 64), (64, 128), (128+3, 128), (128, 128), 
                (128, 256), (256+3, 256), (256, 256), (256, 512)]
        
        if self.fixed_rank:
            lora_rank = self.max_rank
        
        for i, (dim1, dim2) in enumerate(layers, 1):

            if not self.fixed_rank:
                # Compute proportional rank based on layer size and apply rank constraints
                layer_size = dim1 * dim2
                lora_rank = self.nearest_power_of_2(int(layer_size / 1000), min_rank, max_rank) 

            self.__setattr__(f'lora_sa{i}_A', nn.Parameter(torch.empty(dim1, lora_rank)))
            self.__setattr__(f'lora_sa{i}_B', nn.Parameter(torch.empty(lora_rank, dim2)))

            # Print the computed LoRA rank for this layer
            print(f'Layer {i}: dim1={dim1}, dim2={dim2}, lora_rank={lora_rank}')

        layers=[(768, 256),(256, 256),(384, 256),(256, 256),(320, 256), (256, 128), (128, 128), (128, 128), (128, 128)]

        for i, (dim1, dim2) in enumerate(layers, 1):

            if not self.fixed_rank:
                # Compute proportional rank based on layer size and apply rank constraints
                layer_size = dim1 * dim2
                lora_rank = self.nearest_power_of_2(int(layer_size / 1000), min_rank, max_rank) 
            
            # Print the computed LoRA rank for this layer
            print(f'FP Layer {i}: dim1={dim1}, dim2={dim2}, lora_rank={lora_rank}')
            
            self.__setattr__(f'lora_fp{i}_A', nn.Parameter(torch.empty(dim1, lora_rank)))
            self.__setattr__(f'lora_fp{i}_B', nn.Parameter(torch.empty(lora_rank, dim2)))
        
        # Classification LoRA
        if not self.fixed_rank:
            dim1, dim2 = 128, 128
            layer_size = dim1 * dim2
            lora_rank = self.nearest_power_of_2(int(layer_size / 1000), min_rank, max_rank) 
            print(f'Classification Layer: dim1={dim1}, dim2={dim2}, lora_rank={lora_rank}')

        self.lora_l1_A = nn.Parameter(torch.empty(128, lora_rank))
        self.lora_l1_B = nn.Parameter(torch.empty(lora_rank, 128))
     
        trainable_params = 0
        all_param = 0
        for n, p in self.named_parameters():
            all_param += p.numel()
            if 'lora' in n:
                trainable_params += p.numel()
                if n[-1]=='A':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif n[-1]=='B':
                    nn.init.zeros_(p)
            else:
                p.requires_grad = False
        print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )   
    
    # Helper function to get the nearest power of 2 within constraints
    def nearest_power_of_2(self, n, min_rank, max_rank):
        if n <= 0:
            return min_rank  # Default to min_rank if n is 0 or negative
        power_of_2 = 2 ** math.ceil(math.log2(n))
        # Clamp the result within the rank constraints
        return max(min_rank, min(max_rank, power_of_2))


    def forward(self, pc):
        """
        Input:
            pc: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        radius=self.radius

        l0_xyz = pc[:, :3, :]  #[64, 3, 6000]
        l0_points = pc         #[64, 8, 6000]
        # ------------------------------- PointNetSetAbstraction 1 -------------------------------
        l0_xyz = l0_xyz.permute(0, 2, 1)
        l0_points = l0_points.permute(0, 2, 1)

        l1_xyz, l1_points = sample_and_group(1024, radius[0], 32, l0_xyz, l0_points)        
        # l1_xyz: sampled points position data, [B, npoint, C] [64, 1024, 3]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        l1_points = l1_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs_1):
            bn = self.mlp_bns_1[i]
            lora_A = getattr(self, f'lora_sa{i+1}_A')
            lora_B = getattr(self, f'lora_sa{i+1}_B')
            l1_points = F.relu(bn(self.lora_bmm4d(l1_points, conv, lora_A, lora_B, alpha=self.lora_alpha)))  # *lora_A.shape[1]
        l1_points = torch.max(l1_points, 2)[0]

        # ------------------------------- PointNetSetAbstraction 2 -------------------------------
        
        l2_points = l1_points.permute(0, 2, 1)
        l2_xyz, l2_points = sample_and_group(256, radius[1], 32, l1_xyz, l2_points)  # [64, 256, 3]      
        l2_points = l2_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs_2):
            bn = self.mlp_bns_2[i]
            lora_A = getattr(self, f'lora_sa{i+4}_A')
            lora_B = getattr(self, f'lora_sa{i+4}_B')
            l2_points = F.relu(bn(self.lora_bmm4d(l2_points, conv, lora_A, lora_B, self.lora_alpha))) 
        l2_points = torch.max(l2_points, 2)[0]
        
        # ------------------------------- PointNetSetAbstraction 3 -------------------------------
        
        l3_points = l2_points.permute(0, 2, 1)
        l3_xyz, l3_points = sample_and_group(64, radius[2], 32, l2_xyz, l3_points)   # [64, 64, 3]       
        l3_points = l3_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs_3):
            bn = self.mlp_bns_3[i]
            lora_A = getattr(self, f'lora_sa{i+7}_A')
            lora_B = getattr(self, f'lora_sa{i+7}_B')
            l3_points = F.relu(bn(self.lora_bmm4d(l3_points, conv, lora_A, lora_B, self.lora_alpha))) 
        l3_points = torch.max(l3_points, 2)[0]
        
        # ------------------------------- PointNetSetAbstraction 4 -------------------------------
        
        l4_points = l3_points.permute(0, 2, 1)
        l4_xyz, l4_points = sample_and_group(16, radius[3], 32, l3_xyz, l4_points)  # [64, 16, 3]     
        l4_points = l4_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs_4):
            bn = self.mlp_bns_4[i]
            lora_A = getattr(self, f'lora_sa{i+10}_A')
            lora_B = getattr(self, f'lora_sa{i+10}_B')
            l4_points = F.relu(bn(self.lora_bmm4d(l4_points, conv, lora_A, lora_B, self.lora_alpha))) 

        l4_points = torch.max(l4_points, 2)[0]
        l4_points = l4_points.view(-1, 512, 16)

        # ------------------------------------------------------------------------------------------------
        # --------------------------------- Feature Propagation ------------------------------------------
        # ------------------------------------------------------------------------------------------------
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # ------------------------------ PointNetFeaturePropagation 4 ------------------------------------
        # ------- l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [b, 256, 64] -------------
        # ------------------------------------------------------------------------------------------------

        l4_points = l4_points.permute(0, 2, 1) # [b, 16, 512]
        B, N, C = l3_xyz.shape

        dists = square_distance(l3_xyz, l4_xyz) # [64,64,16]
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(l4_points, idx) * weight.view(B, N, 3, 1), dim=2) # [64,64,512]

        l3_points = l3_points.permute(0, 2, 1) # [64,64,254]
        l3_points = torch.cat([l3_points, interpolated_points], dim=-1)  # [64,64,768]

        l3_points = l3_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs_fp4):
            bn = self.mlp_bns_fp4[i]
            lora_A = getattr(self, f'lora_fp{i+1}_A')
            lora_B = getattr(self, f'lora_fp{i+1}_B')
            l3_points = F.relu(bn(self.lora_bmm3d(l3_points, conv, lora_A, lora_B, self.lora_alpha))) 
        
        # ------------------------------ PointNetFeaturePropagation 3 ------------------------------------
        # ------- l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [b, 256, 256]  -----------
        # ------------------------------------------------------------------------------------------------

        l3_points = l3_points.permute(0, 2, 1)
        B, N, C = l2_xyz.shape

        dists = square_distance(l2_xyz, l3_xyz)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(l3_points, idx) * weight.view(B, N, 3, 1), dim=2)

        l2_points = l2_points.permute(0, 2, 1)
        l2_points = torch.cat([l2_points, interpolated_points], dim=-1)

        l2_points = l2_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs_fp3):
            bn = self.mlp_bns_fp3[i]
            lora_A = getattr(self, f'lora_fp{i+3}_A')
            lora_B = getattr(self, f'lora_fp{i+3}_B')
            l2_points = F.relu(bn(self.lora_bmm3d(l2_points, conv, lora_A, lora_B, self.lora_alpha))) 

        # ------------------------------ PointNetFeaturePropagation 2 ------------------------------------
        # ------- l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [b, 128, 1024] -----------
        # ------------------------------------------------------------------------------------------------

        l2_points = l2_points.permute(0, 2, 1)  # [64,256,256]
        B, N, C = l1_xyz.shape

        dists = square_distance(l1_xyz, l2_xyz)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(l2_points, idx) * weight.view(B, N, 3, 1), dim=2) # [64,1024,256]

        l1_points = l1_points.permute(0, 2, 1)
        l1_points = torch.cat([l1_points, interpolated_points], dim=-1)  # [64,1024,64]

        l1_points = l1_points.permute(0, 2, 1)  # [64,320,1024]
        for i, conv in enumerate(self.mlp_convs_fp2):
            bn = self.mlp_bns_fp2[i]
            lora_A = getattr(self, f'lora_fp{i+5}_A')
            lora_B = getattr(self, f'lora_fp{i+5}_B')
            l1_points = F.relu(bn(self.lora_bmm3d(l1_points, conv, lora_A, lora_B, self.lora_alpha))) 


        # ------------------------------ PointNetFeaturePropagation 1 ------------------------------------
        # --------  l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [b, 128, 4096]    -----------
        # ------------------------------------------------------------------------------------------------

        l1_points = l1_points.permute(0, 2, 1)
        B, N, C = l0_xyz.shape

        dists = square_distance(l0_xyz, l1_xyz)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(l1_points, idx) * weight.view(B, N, 3, 1), dim=2)

        l0_points = interpolated_points

        l0_points = l0_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs_fp1):
            bn = self.mlp_bns_fp1[i]
            lora_A = getattr(self, f'lora_fp{i+7}_A')
            lora_B = getattr(self, f'lora_fp{i+7}_B')
            l0_points = F.relu(bn(self.lora_bmm3d(l0_points, conv, lora_A, lora_B, self.lora_alpha))) 

        x = self.drop1(F.relu(self.bn1(self.lora_bmm3d(l0_points, self.conv1, self.lora_l1_A, self.lora_l1_B, self.lora_alpha))))   # [b, 128, 4096]

        # ------------------------------------------------------------------------------------------------
        # --------------------------------- Classification layer -----------------------------------------
        # ------------------------------------------------------------------------------------------------

        x = self.lora_classifier(x)  # [b, n_classes, 4096]

        x = F.log_softmax(x, dim=1)  # [b, n_classes, 4096]
        x = x.permute(0, 2, 1)

        return x


    def _compute_lora_weight(self, A, B):
        # Computes scaled LoRA weight once
        return (self.lora_alpha / A.shape[1]) * (A @ B)

    def lora_bmm4d(self, x, layer, lora_A, lora_B, alpha=1):
        """
        Applies Low-Rank Adaptation (LoRA) to the output of a given layer using
        batch matrix multiplication.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, features, sampled_points, points]
        - layer (torch.nn.Module): The layer to which the input tensor `x` is passed
        - lora_A (torch.Tensor): The first low-rank matrix of shape [input_features, rank]
        - lora_B (torch.Tensor): The second low-rank matrix of shape [rank, output_features]
          
          A shape: torch.Size([11, 16])
          B shape: torch.Size([16, 32])

        Returns:
        - h (torch.Tensor): Output tensor after applying the LoRA transformation,
          with the same shape as the layer output
         """
        
        h = layer(x)  # [B, Cout, S, N]

        B, Cin, S, N = x.shape

        # Precompute AB once
        AB = (alpha / lora_A.shape[1]) * (lora_A @ lora_B)  # [Cin, Cout]

        # x: [B, Cin, S, N] -> [B, N, S, Cin]
        x_t = x.permute(0, 3, 2, 1)

        # Einsum:
        # [B, N, S, Cin] x [Cin, Cout] -> [B, N, S, Cout]
        lora_res = torch.einsum('bnsc,co->bnso', x_t, AB)

        # Back to [B, Cout, S, N]
        lora_res = lora_res.permute(0, 3, 2, 1)

        return h + lora_res
    

    def lora_bmm3d(self, x, layer, lora_A, lora_B, alpha=1):

        """
        Applies Low-Rank Adaptation (LoRA) to the output of a given layer using
        batch matrix multiplication.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, input_features, sequence_length]
        - layer (torch.nn.Module): The layer to which the input tensor `x` is passed
        - lora_A (torch.Tensor): The first low-rank matrix of shape [input_features, rank]
        - lora_B (torch.Tensor): The second low-rank matrix of shape [rank, output_features]

        Returns:
        - h (torch.Tensor): Output tensor after applying the LoRA transformation,
          with the same shape as the layer output
        """
        """ does the work of combining outputs from conv1D layer and lora layer for x
         notice that h is the sum of two separate operations on x
          
          A shape: torch.Size([3, 1])
          B shape: torch.Size([1, 64])

         """
        h = layer(x)  # [B, Cout, N]

        # Precompute AB once
        AB = (alpha / lora_A.shape[1]) * (lora_A @ lora_B)  # [Cin, Cout]

        # x: [B, Cin, N] -> [B, N, Cin]
        x_t = x.transpose(1, 2)

        # Einsum instead of repeat + bmm
        # [B, N, Cin] x [Cin, Cout] -> [B, N, Cout]
        lora_res = torch.einsum('bnc,co->bno', x_t, AB)

        # Back to [B, Cout, N]
        lora_res = lora_res.transpose(1, 2)

        return h + lora_res
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import cv2
from torchvision.ops import roi_align, nms
from Network.utils import FeatureSelectionModule
from Network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ==============================================================================
# Basic Blocks
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# ==============================================================================
# CBAM (Channel and Spatial Attention)
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==============================================================================
# MLE (Formerly DecoderBlock / HWconv)
# ==============================================================================

class MLE(nn.Module):
    def __init__(self, in_channels, n_filters, inp=False, ksize=7):
        super(MLE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp
        
        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, ksize), padding=(0,(ksize - 1) // 2)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (ksize, 1), padding=((ksize - 1) // 2, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (ksize, 1), padding=((ksize - 1) // 2, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, ksize), padding=(0, (ksize - 1) // 2)
        )

        self.bn2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

# ==============================================================================
# Transformer Helpers
# ==============================================================================

class FixCNN(nn.Module):
    def __init__(self, win_size=16):
        super(FixCNN, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, win_size, win_size))
    def forward(self, x):
        out = F.conv2d(x, self.weight, bias=None, stride=1, padding=0)
        return out

def Shifted_Windows(height, width, win_size, stride=2):
    shift_y = torch.arange(0, height-win_size+1, stride)
    shift_x = torch.arange(0, width-win_size+1, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)
    M = shift.shape[0]
    window = shift.reshape(M, 4)
    window[:, 2] = window[:, 0] + win_size-1
    window[:, 3] = window[:, 1] + win_size-1
    return window

def make_gridsx(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_x)

def make_gridsy(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_y)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm3p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm5 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x5, x4, x3, **kwargs):
        return self.fn(self.norm5(x5), self.norm4(x4), self.norm3(x3), **kwargs)

class PreNorm2p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, mask, y, **kwargs):
        return self.fn(self.norm(x), mask, y, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention_global(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim*2, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x5, x4, x3):
        q1 = self.to_q1(x3)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        q2 = self.to_q2(x3)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)

        k1 = self.to_k1(x4)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.heads)
        v1 = self.to_v1(x4)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads)

        k2 = self.to_k1(x5)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)
        v2 = self.to_v1(x5)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        out = torch.cat((out1, out2), dim=-1)

        return self.to_out(out)

class Attention_global2(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x5, x3):
        q2 = self.to_q2(x3)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)

        k2 = self.to_k1(x5)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)
        v2 = self.to_v1(x5)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        return self.to_out(out2)

class Attention_local(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., win_size=16, img_height=256, img_width=256):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.win_size = win_size

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # --------------------- new sub-networks ---------------------
        self.fixcnn = FixCNN(win_size=win_size//2)
        self.window = Shifted_Windows(img_height, img_width, win_size)
        self.window4 = Shifted_Windows(img_height, img_width // 2, win_size // 2)
        self.shifty = make_gridsy(win_size).cuda()
        self.shiftx = make_gridsx(win_size).cuda()

        self.conv1 = DoubleConv(dim * 2, dim)
        self.conv1_0 = DoubleConv(dim, dim)
        self.conv1_1 = MLE(dim, dim, ksize=7) # Renamed from HWconv
        self.fsm = FeatureSelectionModule(dim, dim)
        self.conv2 = DoubleConv(dim * 2, dim)
        
    def get_win(self, keep_windowi, entropy):
        n, _ = keep_windowi.shape
        keep_windowi = keep_windowi.int()
        # Initialize wins_score to avoid potential UnboundLocalError
        wins_score = None 
        for i in range(n):
            x1, y1, x2, y2 = keep_windowi[i] // 2
            win_score = self.fixcnn(entropy[None, None, x1:x2 + 1, y1:y2 + 1]) / (self.win_size // 2 * self.win_size // 2)  # b 1 h w
            if i == 0:
                wins_score = win_score
            else:
                wins_score = torch.cat((wins_score, win_score), dim=0)

        wins_score = wins_score.view(1, -1)
        keep_windowi = keep_windowi.float()
        indexi = nms(boxes=keep_windowi, scores=wins_score[0], iou_threshold=0.2)
        keep_num = n // 4
        indexi = indexi[:keep_num]
        keep_windowi = keep_windowi[indexi, :]
        return keep_windowi

    def forward(self, x, prob, y, epoch):
        b, c, h, w = prob.shape
        log_prob = torch.log2(prob + 1e-10)
        entropy = -1 * torch.sum(prob * log_prob, dim=1)  # b h w
        
        log_y = torch.log2(y + 1e-10)
        entropy_y = -1 * torch.sum(y * log_y, dim=1)  # b h w
        
        outx_2d = x * 0  # b d h w
        win_cunt = x[:, 0, :, :] * 0 # b h w

        # compute the score of each window, achieve by fix the filters
        win_score = self.fixcnn(entropy[:, None, :, :])/(self.win_size//2*self.win_size//2) # b 1 h w
        win_score = win_score.view(b, -1)
        window = torch.from_numpy(self.window).cuda().float() # N 4
        keep_num = min(int(0.7*(2 * h // self.win_size)**2), 30) * 4 #120
        for i in range(b):
            scorei = win_score[i]  # N
            indexi = nms(boxes=window, scores=scorei, iou_threshold=0.4)
            indexi = indexi[:keep_num]
            keep_windowi = window[indexi, :]
            # 同步CUDA流
            torch.cuda.synchronize()
            keep_windowi = self.get_win(keep_windowi, entropy_y[i])

            window_batch_indexi = torch.zeros(keep_windowi.shape[0]) + i
            window_batch_indexi = window_batch_indexi.cuda().float()
            index_windowi = torch.cat([window_batch_indexi[:, None], keep_windowi], dim=1)
            window_featurei = roi_align(x, index_windowi, (self.win_size, self.win_size))  # b d h w
            
            conv_features1 = self.conv1_0(window_featurei)
            conv_features11 = self.conv1_1(window_featurei)
            
            x_0_1 = torch.cat([conv_features1, conv_features11], dim=1)
            x_2 = self.conv1(x_0_1)
            
            m = x_2.shape[0]
            for j in range(m):
                sy = int(keep_windowi[j, 1])
                sx = int(keep_windowi[j, 0])
                outx_2d[i, :, sy:sy+self.win_size, sx:sx+self.win_size] += x_2[j, :, :, :]
                win_cunt[i, sy:sy+self.win_size, sx:sx+self.win_size] += 1
        
        outx = outx_2d / (win_cunt[:, None, :, :] + 1e-10)
        x = self.conv2(torch.cat([x,outx],dim=1))
        return x

class Transformer_global(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm3p(dim, Attention_global(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x5, x4, x):
        for attn, ff in self.layers:
            x = attn(x5, x4, x) + x
            x = ff(x) + x
        return x

class Transformer_global2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2p(dim, Attention_global2(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x5, x):
        for attn, ff in self.layers:
            x = attn(x5,  x) + x
            x = ff(x) + x
        return x

class Transformer_local(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128, win_size=16, img_height=256, img_width=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        # Note: The original code iterated depth but didn't use the layers in forward properly or just used the last one?
        # In original Transformer_block_local forward, it called self.transformer(x, x2, y, epoch)
        # And Transformer_local.forward called self.atten(x, fore_score, y, epoch)
        # The loop in __init__ created layers but forward didn't use them in the original code I read!
        # Original code:
        # self.layers = ...
        # self.atten = Attention_local(...)
        # def forward(...): return self.atten(...)
        # So the layers created in loop were unused?
        # I will keep it as is to maintain behavior.
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, Attention_local(dim, heads=heads, dim_head=dim_head, dropout=dropout, win_size=win_size, img_height=img_height, img_width=img_width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.atten = Attention_local(dim, heads=heads, dim_head=dim_head, dropout=dropout, win_size=win_size,
                                     img_height=img_height, img_width=img_width)

    def forward(self, x, fore_score, y, epoch):
        x1 = self.atten(x, fore_score, y, epoch)
        return x1

# ==============================================================================
# MGE (Formerly Transformer_block_global)
# ==============================================================================

class MGE(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding_x5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear(in_channels*4, self.dmodel),
        )
        self.to_patch_embedding_x4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
            nn.Linear(in_channels*2*4, self.dmodel), # 2 means the channel of x4 is double of channel of x3, 4 is p1*p2
        )
        self.to_patch_embedding_x3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )
        self.to_patch_embedding_y = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(64, self.dmodel),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_global(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x5, x4, x3):
        x5 = self.to_patch_embedding_x5(x5)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        x4 = self.to_patch_embedding_x4(x4)
        x3 = self.to_patch_embedding_x3(x3)

        _, n5, _ = x5.shape
        _, n4, _ = x4.shape
        _, n3, _ = x3.shape

        x5 += self.pos_embedding[:, :n5]
        x4 += self.pos_embedding[:, :n4]
        x3 += self.pos_embedding[:, :n3]

        x5 = self.dropout(x5)
        x4 = self.dropout(x4)
        x3 = self.dropout(x3)
        # transformer layer
        ax = self.transformer(x5, x4, x3)
        out = self.recover_patch_embedding(ax)
        return out

class MGE_v2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding_x5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear(in_channels*2, self.dmodel),
        )
        self.to_patch_embedding_x3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_global2(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x5,x3):
        x5 = self.to_patch_embedding_x5(x5)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        x3 = self.to_patch_embedding_x3(x3)

        _, n5, _ = x5.shape
        _, n3, _ = x3.shape

        x5 += self.pos_embedding[:, :n5]
        x3 += self.pos_embedding[:, :n3]

        x5 = self.dropout(x5)
        x3 = self.dropout(x3)
        # transformer layer
        ax = self.transformer(x5, x3)
        out = self.recover_patch_embedding(ax)
        return out

# ==============================================================================
# LGBM (Formerly Transformer_block_local)
# ==============================================================================

class LGBM(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, n_classes=9, patch_size=1, win_size=16, heads=6, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_local(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches, win_size, image_height, image_width)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x1, x2, y, epoch):
        x = self.to_patch_embedding(x1)  # (b, c, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # transformer layer
        ax = self.transformer(x, x2, y, epoch)
        out = self.recover_patch_embedding(ax)
        return out

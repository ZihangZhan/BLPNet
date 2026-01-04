import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.DCN.dcn_v2 import DeformConv2D


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        # 定义一个卷积层，用于生成注意力权重
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        # 定义一个 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        # 定义一个卷积层，用于生成最终特征
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)

    def forward(self, x):
        # 通过卷积生成注意力权重，并经过 Sigmoid 激活函数
        # print(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        # exit(0)
        # F.avg_pool2d(x, x.size()[2:]) 获得1x1的权重信息
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        # 将输入特征图与注意力权重相乘，得到加权特征
        feat = torch.mul(x, atten)
        # 将加权特征与原始特征相加
        x = x + feat
        # 通过卷积生成最终特征
        feat = self.conv(x)
        return feat



class FeatureAlignModule(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc):
        super(FeatureAlignModule, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.dcpack_L2 = DeformConv2D(out_nc, out_nc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_l, feat_s):
        # feat_l, feat_s = x, x
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        # print('68=======',feat_arm.shape,'-----------',feat_up.shape,'=-=--=-=---=-=-',(torch.cat([feat_arm, feat_up * 2], dim=1).shape))
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2(offset))
        return feat_align + feat_arm


class SpatialAttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


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


class MultiscaleSoftFusion(nn.Module):

    def __init__(self, channels):
        super(MultiscaleSoftFusion, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)

        self.sam1 = SpatialAttentionBlock(channels)
        self.sam2 = SpatialAttentionBlock(channels)
        self.sam3 = SpatialAttentionBlock(channels)

        self.fusion1_1 = SingleConv(channels * 2, channels)
        self.fusion1_2 = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.fusion2_1 = SingleConv(channels * 2, channels)
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        sam_out_1 = self.sam1(x)
        sam_out_2 = self.sam2(x)
        sam_out_3 = self.sam3(x)

        cat_out_1 = self.fusion1_1(torch.cat([x1, x2], dim=1))
        out_1 = self.fusion1_2(torch.cat([cat_out_1, sam_out_1, sam_out_2], dim=1))

        cat_out_2 = self.fusion2_1(torch.cat([x2, x3], dim=1))
        out_2 = self.fusion2_2(torch.cat([cat_out_2, sam_out_2, sam_out_3], dim=1))

        return (out_1 + out_2) * x + x


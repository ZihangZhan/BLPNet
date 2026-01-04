# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.utils import FeatureSelectionModule, FeatureAlignModule, MultiscaleSoftFusion
from Network.blocks import DoubleConv, SingleConv, up_conv, MLE, MGE, MGE_v2, LGBM, CBAM

class BLPNet(nn.Module):

    def __init__(self, num_classes=1):
        super(BLPNet, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.conv1 = DoubleConv(3, filters[0])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(filters[3], filters[4])

        # FAZ path decoder
        self.faz_fsm5 = FeatureSelectionModule(in_chan=filters[4], out_chan=filters[3])

        self.faz_up6 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.faz_fsm6 = FeatureSelectionModule(in_chan=filters[3], out_chan=filters[2])
        self.faz_fam6 = FeatureAlignModule(in_nc=filters[2], out_nc=filters[2])
        self.faz_conv6 = SingleConv(ch_in=filters[3], ch_out=filters[3])
        self.faz_conv62 = SingleConv(ch_in=filters[3] * 2, ch_out=filters[3])

        self.faz_up7 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.faz_fsm7 = FeatureSelectionModule(in_chan=filters[2], out_chan=filters[1])
        self.faz_fam7 = FeatureAlignModule(in_nc=filters[1], out_nc=filters[1])
        self.faz_conv7 = SingleConv(ch_in=filters[2], ch_out=filters[2])
        self.faz_conv72 = SingleConv(ch_in=filters[2] * 2, ch_out=filters[2])

        self.faz_up8 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.faz_fsm8 = FeatureSelectionModule(in_chan=filters[1], out_chan=filters[0])
        self.faz_fam8 = FeatureAlignModule(in_nc=filters[0], out_nc=filters[0])
        self.faz_conv8 = SingleConv(ch_in=filters[1], ch_out=filters[1])

        self.faz_up9 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.faz_fsm9 = FeatureSelectionModule(in_chan=filters[0], out_chan=filters[0] // 2)
        self.faz_fam9 = FeatureAlignModule(in_nc=filters[0] // 2, out_nc=filters[0] // 2)
        self.faz_conv9 = SingleConv(ch_in=filters[0], ch_out=filters[0])
        #-----------GLFA------------
        self.softmax = nn.Softmax(dim=1)
        for p in self.parameters():
            p.requires_grad = True  # set "True" manually in the first 350 epochs, then load the best model and set "False" manually in the following 50 epochs.
        imgsize = 400
        filters = [32, 64, 128, 256, 512]
        # MLE
        self.mle7 = MLE(filters[2], filters[2], ksize=7)
        self.mle6 = MLE(filters[3], filters[3], ksize=3)

        # MGE
        self.mge4 = MGE_v2(filters[3], filters[3], imgsize // 8, 1,
                           heads=4, patch_size=1, emb_dropout=0.2)
        self.mge3 = MGE(filters[2], filters[2], imgsize // 4, 1,
                        heads=4, patch_size=1, emb_dropout=0.2)
        # CBAM
        self.cbam6 = CBAM(filters[3], kernel_size=3)
        self.cbam7 = CBAM(filters[2], kernel_size=3)

        # LGBM
        self.lgbm2 = LGBM(filters[1], filters[1], imgsize // 2, 1,
                          heads=6, patch_size=1, n_classes=num_classes, win_size=16, emb_dropout=0.2)

        # RV path decoder
        self.rv_fsm6 = FeatureSelectionModule(in_chan=filters[3], out_chan=filters[2])
        self.rv_attention = MultiscaleSoftFusion(filters[2])

        self.rv_up7 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.rv_fsm7 = FeatureSelectionModule(in_chan=filters[2], out_chan=filters[1])
        self.rv_conv7 = DoubleConv(ch_in=filters[1] * 2, ch_out=filters[1])

        self.rv_up8 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.rv_fsm8 = FeatureSelectionModule(in_chan=filters[1], out_chan=filters[0])
        self.rv_conv8 = DoubleConv(ch_in=filters[0] * 2, ch_out=filters[0])

        self.rv_up9 = up_conv(ch_in=filters[0], ch_out=filters[0] // 2)
        self.rv_fsm9 = FeatureSelectionModule(in_chan=filters[0], out_chan=filters[0] // 2)
        self.rv_conv9 = DoubleConv(ch_in=filters[0], ch_out=filters[0])

        self.out1 = nn.Conv2d(filters[2], num_classes, 1, 1, 0)
        self.out = nn.Conv2d(filters[0], num_classes, 3, 1, 1)

    def forward(self, x, epoch):
        '''encode'''
        conv_out_1 = self.conv1(x)
        pool_out_1 = self.pool1(conv_out_1)
        conv_out_2 = self.conv2(pool_out_1)
        pool_out_2 = self.pool2(conv_out_2)
        conv_out_3 = self.conv3(pool_out_2)
        pool_out_3 = self.pool3(conv_out_3)
        conv_out_4 = self.conv4(pool_out_3)
        pool_out_4 = self.pool4(conv_out_4)
        conv_out_5 = self.conv5(pool_out_4)

        '''decoder path for RV branch'''
        msf = self.rv_fsm6(conv_out_4) + self.rv_attention(self.rv_fsm6(conv_out_4))

        rv_d7 = self.rv_up7(msf)
        rv_cat7 = torch.cat([rv_d7, self.rv_fsm7(conv_out_3)], dim=1)
        rv_d7 = self.rv_conv7(rv_cat7)

        rv_d8 = self.rv_up8(rv_d7)
        rv_cat8 = torch.cat([rv_d8, self.rv_fsm8(conv_out_2)], dim=1)
        rv_d8 = self.rv_conv8(rv_cat8)

        rv_d9 = self.rv_up9(rv_d8)
        rv_cat9 = torch.cat([rv_d9, self.rv_fsm9(conv_out_1)], dim=1)
        rv_d9 = self.rv_conv9(rv_cat9)
        
        '''decoder path for FAZ branch'''

        '''MLE1'''
        y6 = self.mle6(conv_out_4)
        '''MLE2'''
        y7 = self.mle7(conv_out_3)

        faz_d6 = self.faz_up6(conv_out_5)
        faz_conv_out_4 = self.faz_fsm6(conv_out_4)
        faz_cat6 = torch.cat([self.faz_fam6(faz_conv_out_4, faz_d6), faz_conv_out_4], dim=1)
        faz_d6 = self.faz_conv6(faz_cat6)

        '''GLFA1'''

        '''MGE1 + MLE1'''
        faz_d6 = torch.cat([self.mge4(conv_out_5, faz_d6), y6], dim=1)
        faz_d6 = self.faz_conv62(faz_d6)
        '''CBAM1'''
        faz_d6 = self.cbam6(faz_d6)

        faz_d7 = self.faz_up7(faz_d6)
        faz_conv_out_3 = self.faz_fsm7(conv_out_3)
        faz_cat7 = torch.cat([self.faz_fam7(faz_conv_out_3, faz_d7), faz_conv_out_3], dim=1)
        faz_d7 = self.faz_conv7(faz_cat7)

        '''GLFA2'''
        '''MGE2 + MLE2'''
        faz_d7 = torch.cat([self.mge3(conv_out_5, faz_d6, faz_d7), y7], dim=1)
        faz_d7 = self.faz_conv72(faz_d7)
        '''CBAM2'''
        faz_d7 = self.cbam7(faz_d7)

        faz_d8 = self.faz_up8(faz_d7)
        faz_conv_out_2 = self.faz_fsm8(conv_out_2)
        faz_cat8 = torch.cat([self.faz_fam8(faz_conv_out_2, faz_d8), faz_conv_out_2], dim=1)
        faz_d8 = self.faz_conv8(faz_cat8)

        '''LGBM'''
        pred3_p = self.softmax(faz_d7)
        prv_d7 = self.softmax(rv_d7)
        faz_d8 = self.lgbm2(faz_d8, pred3_p, prv_d7, epoch)

        faz_d9 = self.faz_up9(faz_d8)
        faz_conv_out_1 = self.faz_fsm9(conv_out_1)
        faz_cat9 = torch.cat([self.faz_fam9(faz_conv_out_1, faz_d9), faz_conv_out_1], dim=1)
        faz_d9 = self.faz_conv9(faz_cat9)
        
        pred2 = F.interpolate(faz_d7, scale_factor=4, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(self.out1(pred2)), torch.sigmoid(self.out(faz_d9)), torch.sigmoid(self.out(rv_d9))

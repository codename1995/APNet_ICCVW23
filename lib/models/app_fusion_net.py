from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
import lib.models.pytorch_utils as pt_utils

from .seg_hrnet_ocr import get_seg_model
from lib.models.kpconv.blocks import KPConv
from lib.models.PartialConv import BlurConv


logger = logging.getLogger(__name__)

class APNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.hrnet_seg_ocr = get_seg_model(cfg)

        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.d_out = cfg.MODEL.D_OUT
        self.src_pts = cfg.MODEL.SRC_PTS
        self.IMG_BLUR = cfg.MODEL.IMG_BLUR

        if self.IMG_BLUR:
            self.blur_conv = BlurConv(in_channels=3, out_channels=3, kernel_size=3, stride=1,
                                      padding=1, bias=False)

        self.fc0 = pt_utils.Conv1d(6, 16, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 16
        for i in range(self.num_layers):
            d_out = self.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.d_out[-j-2]
                d_out = 2 * self.d_out[-j-2]
            else:
                d_in = 4 * self.d_out[-4]
                d_out = 2 * self.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.hrnet_seg_ocr_comp = pt_utils.Conv2d(cfg.MODEL.OCR.MID_CHANNELS, self.d_out[0]*2, kernel_size=(1, 1),
                                                      name="hrnet_seg_ocr_comp")  # Channel compression layer

        d_pts_fea_in = 2*self.d_out[0]
        if cfg.MODEL.FUSION == "KPCONV2":
            self.kpconv = KPConv(
                kernel_size=15,
                p_dim=3,
                in_channels=d_pts_fea_in*2,
                out_channels=d_pts_fea_in,
                KP_extent=0.024,  # 1.2 is a default value
                radius=0.05,  # 0.1 = (0.2m/10) x 6.0 = (grid_size/normalized_factor) x rho (rho is 6.0 for SemanticKitti)
                # deformable=True,
            )
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)


        self.final_decoder = nn.Sequential(
            nn.Dropout(0.1),
            pt_utils.Conv2d(d_out, self.num_classes, kernel_size=(1, 1), bn=False, activation=None),
        )

    def forward(self, end_points):
        if self.IMG_BLUR:
            for i in range(3):
                end_points['imgs'], end_points['mask'] = self.blur_conv(end_points['imgs'], end_points['mask'])

        out_aux, out, af0, af1, af2, af3 = self.hrnet_seg_ocr(end_points['imgs'])
        end_points['ocr_out'] = out_aux

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list1 = []
        f_encoder_list2 = []
        features1 = features
        features2 = features
        for i in range(self.num_layers):
            f_encoder_i1 = self.dilated_res_blocks[i](features1, end_points['xyz'][i], end_points['neigh_idx'][i])
            f_encoder_i2 = self.dilated_res_blocks[i](features2, end_points['xyz2'][i], end_points['neigh_idx2'][i])

            f_sampled_i1 = self.random_sample(f_encoder_i1, end_points['sub_idx'][i])
            f_sampled_i2 = self.random_sample(f_encoder_i2, end_points['sub_idx2'][i])
            features1 = f_sampled_i1
            features2 = f_sampled_i2
            if i == 0:
                f_encoder_list1.append(f_encoder_i1)
                f_encoder_list2.append(f_encoder_i2)
            f_encoder_list1.append(f_sampled_i1)
            f_encoder_list2.append(f_sampled_i2)
        # ###########################Encoder############################

        features1 = self.decoder_0(f_encoder_list1[-1])
        features2 = self.decoder_0(f_encoder_list2[-1])

        # ###########################Decoder############################
        f_decoder_list1 = []
        f_decoder_list2 = []
        for j in range(self.num_layers):
            f_interp_i1 = self.nearest_interpolation(features1, end_points['interp_idx'][-j - 1])
            f_interp_i2 = self.nearest_interpolation(features2, end_points['interp_idx2'][-j - 1])
            f_decoder_i1 = self.decoder_blocks[j](torch.cat([f_encoder_list1[-j - 2], f_interp_i1], dim=1))
            f_decoder_i2 = self.decoder_blocks[j](torch.cat([f_encoder_list2[-j - 2], f_interp_i2], dim=1))

            features1 = f_decoder_i1
            features2 = f_decoder_i2
            f_decoder_list1.append(f_decoder_i1)
            f_decoder_list2.append(f_decoder_i2)
        # ###########################Decoder############################
        p_fea = features1+features2  # BxCxNx1

        af3 = self.hrnet_seg_ocr_comp(af3)  # BxCxWxH
        a_out = self.final_decoder(af3).squeeze(3)
        a_fea = F.grid_sample(af3, end_points['pts_img_idx0'], mode="bilinear", align_corners=False, padding_mode='zeros')  # BxCx1xN
        a_fea = a_fea.permute(0, 1, 3, 2)  # BxCxNx1
        p_out = self.final_decoder(p_fea).squeeze(3)

        # P-branch KPConv
        features = torch.cat((a_fea, p_fea), axis=1)
        knns = end_points['knns']
        res = []
        for i in range(p_fea.shape[0]):
            xyz = end_points['xyz'][0][i, ...]  # xyz for 0-layer is the original point
            feats = features[i, ...].transpose(0, 1).squeeze()  # Bx2CxNx1 --> 2CxNx1 --> Nx2C
            if self.src_pts:
                query_pts = end_points["src_xyz"][i, ...]
                feats = self.kpconv(query_pts, xyz, knns[i], feats)
            else:
                feats = self.kpconv(xyz, xyz, knns[i], feats)  # NxC
            res.append(feats.unsqueeze(2).transpose(0, 1).unsqueeze(0))  # NxC --> CxNx1 --> 1xCxNx1
        features = torch.cat(res, axis=0).squeeze(3)  # b_sxCxN
        features = self.activation(features)
        features = features.unsqueeze(3)
        f_out = self.final_decoder(features).squeeze(3)

        end_points['logits'] = f_out
        end_points['a_out'] = a_out
        end_points['p_out'] = p_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def init_pp_weights(self, pretrained=None):
        if pretrained is None:
            raise NotImplemented
        logger.info('=> loading pretrained model {}'.format(pretrained))

        pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k:v for k,v in pretrained_dict.items()
                           if 'final_decoder' not in k
                           or 'blur_conv' not in k}

        model_dict=self.state_dict()
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logger.info('=> Successfully load pretrained model {}'.format(pretrained))




class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        # batch*npoint*nsamples*10
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        # f_agg = torch.mean(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


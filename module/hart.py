import torch
import torch.nn as nn
import torch.nn.functional as F
from module.extractor import Feature
from module.formerblock import Hadamard_Attention_Transformer
from module.geometry import IGEV_Volume
from module.submodule import *

from module.feat_extractor_backbone import PretrainedBackbone
from module.decoder import LSTMDecoder

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        chans = [128, 192, 448, 384]
        # chans = [48,64,192,160]

        self.feature_att_8 = FeatureAtt(in_channels * 2, chans[1])
        self.feature_att_16 = FeatureAtt(in_channels * 4, chans[2])
        self.feature_att_32 = FeatureAtt(in_channels * 6, chans[3])
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, chans[2])
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, chans[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class HART(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.backbone = PretrainedBackbone(args)

        self.transformer_encoder = Hadamard_Attention_Transformer()

        self.transformer_decoder = LSTMDecoder(args)


        self.bias_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 4, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        chans = [128, 192, 448, 384]
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, chans[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(chans[0], chans[0], 3, 1, 1, bias=False),
            nn.InstanceNorm2d(chans[0]), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(chans[0] * 2, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )


        self.conv = BasicConv_IN(chans[0] * 2, self.args.max_disp // 2, kernel_size=3, padding=1, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, chans[0] * 2)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, image1, image2, iters=32, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            stem_2x, features_left, features_right, match_left, match_right = self.backbone(image1, image2)

            transformer_list = self.transformer_encoder(image1)

            net_list = [torch.tanh(x[0]) for x in transformer_list]
            inp_list = [torch.relu(x[1]) for x in transformer_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//4, dim=1)) for i,conv in zip(inp_list, self.bias_convs)]

            gwc_volume = build_gwc_volume(match_left, match_right, 192//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(encoding_volume).squeeze(1), dim=1)

            init_disp = disparity_regression(prob, self.args.max_disp//4)   # feat

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

        corr_block = IGEV_Volume
        corr_fn = corr_block(match_left.float(), match_right.float(), encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp

        if test_mode:
            disp_up = self.transformer_decoder(iters, stem_2x, net_list, inp_list, disp, corr_fn, coords, test_mode)
            return disp_up

        disp_preds = self.transformer_decoder(iters, stem_2x, net_list, inp_list, disp, corr_fn, coords, test_mode)

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds

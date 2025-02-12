import torch
import torch.nn as nn
import torch.nn.functional as F

from module.update import LSTMMultiUpdateBlock
from module.submodule import *

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


class LSTMDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args =args
        self.update_block = LSTMMultiUpdateBlock(args, hidden_dims=args.hidden_dims)

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

    def forward(self, iters, stem_2x, net_list, inp_list, disp, corr_fn, coords, test_mode):
        disp_preds = []

        netC = net_list
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = corr_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):

                netC, net_list, mask_feat_4, delta_disp = self.update_block(netC, net_list, inp_list, geo_feat, disp,
                                                                            iter16=self.args.n_gru_layers == 3,
                                                                            iter08=self.args.n_gru_layers >= 2)

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

            if test_mode:
                return disp_up

        return disp_preds

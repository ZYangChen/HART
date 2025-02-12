import torch
import torch.nn as nn
import torch.nn.functional as F


from module.extractor import Feature
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



class PretrainedBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature = Feature()
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

        self.conv = BasicConv_IN(chans[0] * 2, self.args.max_disp // 2, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(self.args.max_disp // 2, self.args.max_disp // 2, kernel_size=1, padding=0, stride=1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, image1, image2):
        features_left = self.feature(image1)
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))


        return stem_2x, features_left, features_right, match_left, match_right
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS


class AdaptiveAngleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, angle_list=45, bias=None):
        super(AdaptiveAngleConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.angle_list = angle_list
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.randn((out_channel, in_channel) + kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channel, 1, 1))
        self.zero_pad_input = torch.nn.ZeroPad2d(stride)

    def forward(self, x):
        x = self.zero_pad_input(x)
        return self._conv2d_multi_in_out(x, self.weight, self.angle_list) + self.bias

    def _rotate_kernel(self, original_kernel, angle):  # only 3*3 kernel
        n = angle // 45
        new_kernel = torch.zeros_like(original_kernel)
        if n == 1 :  # 45 degree
            new_kernel[0][1:] = original_kernel[0][:2]
            new_kernel[1][2] = original_kernel[0][2]
            new_kernel[2][2] = original_kernel[1][2]
            new_kernel[2][:2] = original_kernel[2][1:]
            new_kernel[0][0] = original_kernel[1][0]
            new_kernel[1][0] = original_kernel[2][0]
            new_kernel[1][1] = original_kernel[1][1]

        if n == 2:  # 90 degree
            #new_kernel = original_kernel.reshape(original_kernel.size())
            #new_kernel = new_kernel[::-1]
            #new_kernel = new_kernel.reshape(original_kernel.shape)
            #new_kernel = np.transpose(new_kernel)[::-1]
            l = len(original_kernel)
            for i in range(l):
                for j in range(l):
                    new_kernel[j][l - 1 - i] = original_kernel[i][j]


        if n == 3:  #  135 degree
            new_kernel[1][2] = original_kernel[0][0]
            new_kernel[2][2] = original_kernel[0][1]
            new_kernel[2][0] = original_kernel[1][2]
            new_kernel[2][1] = original_kernel[0][2]
            new_kernel[0][0] = original_kernel[2][1]
            new_kernel[1][0] = original_kernel[2][2]
            new_kernel[0][1] = original_kernel[2][0]
            new_kernel[0][2] = original_kernel[1][0]
            new_kernel[1][1] = original_kernel[1][1]

        if n == 4:  # 180 degree
            l = len(original_kernel)
            for i in range(l):
                for j in range(l):
                    new_kernel[i][j] = original_kernel[l - 1 - i][l - 1 - j]

        return new_kernel


    def _my_rotate_conv2d(self, input, kernel, angle_list):  # angle_list must contains at least 0 degree
        b, h, w = input.shape
        k_h, k_w = kernel.shape
        # rotate convolutional kernel
        nums_rotate_kernel = len(angle_list)
        kernels = [kernel]  # 0 degree
        for i in range(1, nums_rotate_kernel):
            kernels.append(self._rotate_kernel(kernel, angle_list[i]))

        Y =  []
        for kernel in kernels:
            y = torch.zeros((b, h - k_h + 1, w - k_w + 1))
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    y[:, i, j] = (input[:, i:i+k_h, j:j+k_w] * kernel).sum(dim=2).sum(dim=1) # stride=1
            Y.append(y)
        return Y

    def _conv2d_multi_in(self, X, K, angle_list):
        res = self._my_rotate_conv2d(X[:, 0, :, :], K[0, :, :], angle_list)
        for i in range(1, X.shape[1]):
            res_temp = self._my_rotate_conv2d(X[:, i, :, :], K[i, :, :], angle_list)
            for j in range(len(angle_list)):
                res[j] += res_temp[j]
        return res

    def _conv2d_multi_in_out(self, X, K, angle_list):  # maybe it's wrong
       res = self._conv2d_multi_in(X, K[0], angle_list)
       for i, k in enumerate(K):
            if i > 0:
                temp = self._conv2d_multi_in(X, k, angle_list)
                for j in range(len(angle_list)):
                    res[j] = torch.stack([res[j], temp[j]], dim=1)
       return res
        # return torch.stack([self._conv2d_multi_in(X, k, angle_list) for k in K], dim=1)


class SKConv(nn.Module):
    def __init__(self, features, M, r, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.features = features
        self.bn_relus = nn.ModuleList()
        for i in range(M):
            self.bn_relus.append(nn.Sequential(
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, branch):
        feas = []
        for i in range(branch):
            feas.append(self.bn_relus[i](X[i]).unsqueeze_(dim=1))
        feas = torch.cat(feas, dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


@NECKS.register_module()
class FPN_AAR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN_AAR, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_rotate_convs = nn.ModuleList()  # rotate_convolution , M angle branch
        self.fusions = nn.ModuleList()  # add fusion module SK-Net

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            """
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            """

            fpn_rotate_conv = AdaptiveAngleConv(
                out_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 45, 90, 135, 180],
                bias=None
            )
            fusion = SKConv(256, 5, 2, 32)

            self.lateral_convs.append(l_conv)
            self.fpn_rotate_convs.append(fpn_rotate_conv)
            self.fusions.append(fusion)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                """
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                """
                extra_fpn_rotate_conv = AdaptiveAngleConv(
                in_channels,
                out_channels,
                3,
                1,
                1,
                angle_list=[0, 45, 90, 135, 180],
                bias=None
            )
                fusion = SKConv(256, 5, 2, 32)

                self.fpn_rotate_convs.append(extra_fpn_rotate_conv)
                self.fusions.append(fusion)


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        # convolution: local
        conv_outs = [
            self.fpn_rotate_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # local-global fusion
        outs = [
            self.fusions[i](conv_outs[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fusions[used_backbone_levels](self.fpn_rotate_convs[used_backbone_levels](extra_source)))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fusions[i](self.fpn_rotate_convs[i](F.relu(outs[-1]))))
                    else:
                        outs.append(self.fusions[i](self.fpn_rotate_convs[i](outs[-1])))
        return tuple(outs)



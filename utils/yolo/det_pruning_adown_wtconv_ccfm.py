from ultralytics import YOLO
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from ultralytics.nn.modules import Conv, SPPF, Detect, ADown, C2PSA, C3k2
from ultralytics.nn.Extramodule.WTConv import C3k2_WTConv, Bottleneck_WTConv


class PRUNE:
    def __init__(self):
        self.threshold = None


    def get_threshold(self, model, factor=0.8):
        ws = []
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                w = m.weight.abs().detach()
                ws.append(w)
                print(name, w.max().item(), w.min().item())
        ws = torch.cat(ws)
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]

    def _get_keep_idxs(self, gamma, min_keep=8):
        keep_idxs = []
        local_threshold = self.threshold
        while len(keep_idxs) < min_keep:
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold *= 0.5
        return keep_idxs

    def _prune_conv_out(self, conv_module, keep_idxs):
        gamma = conv_module.bn.weight.data.detach()
        beta = conv_module.bn.bias.data.detach()
        n = len(keep_idxs)

        conv_module.bn.weight.data = gamma[keep_idxs]
        conv_module.bn.bias.data = beta[keep_idxs]
        conv_module.bn.running_var.data = conv_module.bn.running_var.data[keep_idxs]
        conv_module.bn.running_mean.data = conv_module.bn.running_mean.data[keep_idxs]
        conv_module.bn.num_features = n

        conv_module.conv.weight.data = conv_module.conv.weight.data[keep_idxs]
        conv_module.conv.out_channels = n
        if conv_module.conv.bias is not None:
            conv_module.conv.bias.data = conv_module.conv.bias.data[keep_idxs]

    def _prune_conv_in(self, module, keep_idxs):
        if module is None:
            return

        # 1. 普通 Conv
        if self._is_conv_wrapper(module):
            conv = module.conv

            # depthwise conv: [C, 1, k, k]
            if conv.groups == conv.in_channels == conv.out_channels:
                conv.in_channels = len(keep_idxs)
                conv.out_channels = len(keep_idxs)
                conv.groups = len(keep_idxs)
                conv.weight.data = conv.weight.data[keep_idxs]

                # 同步 BN
                module.bn.weight.data = module.bn.weight.data[keep_idxs]
                module.bn.bias.data = module.bn.bias.data[keep_idxs]
                module.bn.running_var.data = module.bn.running_var.data[keep_idxs]
                module.bn.running_mean.data = module.bn.running_mean.data[keep_idxs]
                module.bn.num_features = len(keep_idxs)
            else:
                conv.in_channels = len(keep_idxs)
                conv.weight.data = conv.weight.data[:, keep_idxs]
            return

        # 2. Detect 分支常见 Sequential
        if isinstance(module, Sequential):
            # 情况 A：Sequential(Conv, Conv)
            if len(module) >= 2 and self._is_conv_wrapper(module[0]) and self._is_conv_wrapper(module[1]):
                first = module[0]
                second = module[1]
                conv0 = first.conv
                conv1 = second.conv

                # 如果第一层是 depthwise conv
                if conv0.groups == conv0.in_channels == conv0.out_channels:
                    conv0.in_channels = len(keep_idxs)
                    conv0.out_channels = len(keep_idxs)
                    conv0.groups = len(keep_idxs)
                    conv0.weight.data = conv0.weight.data[keep_idxs]

                    # 同步第一层 BN
                    first.bn.weight.data = first.bn.weight.data[keep_idxs]
                    first.bn.bias.data = first.bn.bias.data[keep_idxs]
                    first.bn.running_var.data = first.bn.running_var.data[keep_idxs]
                    first.bn.running_mean.data = first.bn.running_mean.data[keep_idxs]
                    first.bn.num_features = len(keep_idxs)

                    # 第二层 pointwise conv 需要裁剪输入维
                    conv1.in_channels = len(keep_idxs)
                    conv1.weight.data = conv1.weight.data[:, keep_idxs]
                else:
                    # 普通 Conv -> Conv
                    conv0.in_channels = len(keep_idxs)
                    conv0.weight.data = conv0.weight.data[:, keep_idxs]
                return

            # 情况 B：旧版 depthwise + pointwise 兼容
            try:
                dw = module[0]
                pw = module[1].conv

                if self._is_conv_wrapper(dw):
                    conv_dw = dw.conv
                    if conv_dw.groups == conv_dw.in_channels == conv_dw.out_channels:
                        conv_dw.in_channels = len(keep_idxs)
                        conv_dw.out_channels = len(keep_idxs)
                        conv_dw.groups = len(keep_idxs)
                        conv_dw.weight.data = conv_dw.weight.data[keep_idxs]

                        dw.bn.weight.data = dw.bn.weight.data[keep_idxs]
                        dw.bn.bias.data = dw.bn.bias.data[keep_idxs]
                        dw.bn.running_var.data = dw.bn.running_var.data[keep_idxs]
                        dw.bn.running_mean.data = dw.bn.running_mean.data[keep_idxs]
                        dw.bn.num_features = len(keep_idxs)

                        pw.in_channels = len(keep_idxs)
                        pw.weight.data = pw.weight.data[:, keep_idxs]
                        return
            except Exception:
                pass

        # 3. 裸 conv
        if hasattr(module, "in_channels") and hasattr(module, "weight"):
            # depthwise/group conv
            if hasattr(module, "groups") and module.groups == module.in_channels == module.out_channels:
                module.in_channels = len(keep_idxs)
                module.out_channels = len(keep_idxs)
                module.groups = len(keep_idxs)
                module.weight.data = module.weight.data[keep_idxs]
            else:
                module.in_channels = len(keep_idxs)
                module.weight.data = module.weight.data[:, keep_idxs]

    def prune_conv(self, conv1: Conv, conv2):
        gamma = conv1.bn.weight.data.detach()
        keep_idxs = self._get_keep_idxs(gamma, min_keep=8)

        self._prune_conv_out(conv1, keep_idxs)

        if not isinstance(conv2, list):
            conv2 = [conv2]

        for item in conv2:
            self._prune_conv_in(item, keep_idxs)

    def prune_adown(self, adown, next_module):
        if not isinstance(adown, ADown):
            raise TypeError(f"prune_adown expects ADown, but got {type(adown).__name__}")

        gamma1 = adown.cv1.bn.weight.data.detach()
        gamma2 = adown.cv2.bn.weight.data.detach()

        keep1 = self._get_keep_idxs(gamma1, min_keep=4)
        keep2 = self._get_keep_idxs(gamma2, min_keep=4)

        self._prune_conv_out(adown.cv1, keep1)
        self._prune_conv_out(adown.cv2, keep2)

        old_c = gamma1.numel()
        keep_idxs = torch.cat([keep1, keep2 + old_c], dim=0)

        if isinstance(next_module, (C3k2_WTConv, C3k2, SPPF, C2PSA)):
            self._prune_conv_in(next_module.cv1, keep_idxs)
        else:
            self._prune_conv_in(next_module, keep_idxs)

    def prune(self, m1, m2):
        # 上游输出模块映射到真正输出通道所在的 Conv
        if isinstance(m1, (C3k2, C3k2_WTConv)):
            m1 = m1.cv2
        elif isinstance(m1, SPPF):
            m1 = m1.cv2
        elif isinstance(m1, C2PSA):
            m1 = m1.cv2
        elif isinstance(m1, Sequential):
            m1 = m1[1]

        if not isinstance(m2, list):
            m2 = [m2]

        # 下游输入模块映射到真正接收输入的 Conv
        for i, item in enumerate(m2):
            if isinstance(item, (C3k2, C3k2_WTConv, SPPF, C2PSA)):
                m2[i] = item.cv1

        self.prune_conv(m1, m2)
    def get_out_channels(self, module):
        """获取模块当前输出通道数"""
        if self._is_conv_wrapper(module):
            return module.conv.out_channels
        elif isinstance(module, (C3k2, C3k2_WTConv, SPPF, C2PSA)):
            return module.cv2.conv.out_channels
        elif isinstance(module, ADown):
            return module.cv1.conv.out_channels + module.cv2.conv.out_channels
        elif isinstance(module, Sequential):
            for m in reversed(module):
                if isinstance(m, Conv):
                    return m.conv.out_channels
        return None

    def update_fusion_input(self, fusion_block, source_modules):
        """
        对 Concat 后接的融合块，按当前实际分支输出通道数更新其输入通道数
        fusion_block: 例如 seq[15]/seq[20]/seq[23]/seq[26]
        source_modules: 参与 concat 的来源模块列表
        """
        if not hasattr(fusion_block, "cv1") or not hasattr(fusion_block.cv1, "conv"):
            return

        new_in = 0
        for m in source_modules:
            c = self.get_out_channels(m)
            if c is None:
                raise ValueError(f"Cannot infer output channels from module: {type(m).__name__}")
            new_in += c

        conv = fusion_block.cv1.conv
        old_in = conv.in_channels

        if new_in == old_in:
            return

        # 只保留前 new_in 个输入通道，保证结构可运行
        conv.weight.data = conv.weight.data[:, :new_in, :, :]
        conv.in_channels = new_in
    def _is_conv_wrapper(self, module):
        return hasattr(module, "conv") and isinstance(module.conv, nn.Conv2d) and hasattr(module, "bn")


def do_pruning(modelpath, savepath, pruning_rate, data_yaml="CityPersons.yaml"):
    pruning = PRUNE()
    yolo = YOLO(modelpath)
    pruning.get_threshold(yolo.model, pruning_rate)

    # 1. 先剪 block 内部可安全裁剪的普通 Conv-BN
    # WTConv 本体内部不直接做结构剪枝
    for name, m in yolo.model.named_modules():
        if isinstance(m, Bottleneck_WTConv):
            if isinstance(m.cv2, Conv):
                pruning.prune_conv(m.cv1, m.cv2)

    seq = yolo.model.model

    print("\n===== Model Structure =====")
    for i, m in enumerate(seq):
        print(i, type(m).__name__)
    print("===========================\n")

    # 2. backbone 主干结构剪枝（按你当前真实模型结构）
    pruning.prune(seq[3], seq[4])         # Conv -> C3k2_WTConv
    pruning.prune_adown(seq[5], seq[6])   # ADown -> C3k2_WTConv
    pruning.prune_adown(seq[7], seq[8])   # ADown -> C3k2_WTConv
    pruning.prune(seq[8], seq[9])         # C3k2_WTConv -> SPPF
    pruning.prune(seq[9], seq[10])        # SPPF -> C2PSA

    # 3. neck / detect head 剪枝（按当前真实结构）
    detect: Detect = seq[-1]

    last_inputs = [seq[20], seq[23], seq[26]]
    colasts = [seq[21], seq[24], None]

    for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
        pruning.prune(last_input, [colast, cv2[0], cv3[0]])
        pruning.prune(cv2[0], cv2[1])
        pruning.prune(cv2[1], cv2[2])
        pruning.prune(cv3[0], cv3[1])
        pruning.prune(cv3[1], cv3[2])

    for _, p in yolo.model.named_parameters():
        p.requires_grad = True
    # 4. 修正 Concat 后融合块的输入通道数
    # layer 15 <- concat(layer 12/11 branch, layer 13)
    pruning.update_fusion_input(seq[15], [seq[11], seq[13]])

    # layer 20 <- concat(layer 17/16 branch, layer 18)
    pruning.update_fusion_input(seq[20], [seq[16], seq[18]])

    # layer 23 <- concat(layer 21, layer 16)
    pruning.update_fusion_input(seq[23], [seq[21], seq[16]])

    # layer 26 <- concat(layer 24, layer 11)
    pruning.update_fusion_input(seq[26], [seq[24], seq[11]])

    yolo.val(data=data_yaml, batch=4, workers=4)
    torch.save(yolo.ckpt, savepath)




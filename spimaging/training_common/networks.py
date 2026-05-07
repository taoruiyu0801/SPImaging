"""3D neural networks for SPAD histogram reconstruction."""

from torch import nn
import torch
import torch.nn.functional as F


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SPAD3DHistogramNet(nn.Module):
    """A compact 3D CNN.

    Input shape:
        (B, 1, T, H, W), where the channel is normalized photon counts.

    Output shape:
        (B, 1, T, H, W), unnormalized logits over the time-bin dimension.
    """

    def __init__(self, in_channels=1, base_channels=8):
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            Conv3DBlock(in_channels, c),
            Conv3DBlock(c, c),
            nn.Conv3d(c, c, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.GroupNorm(_group_count(c), c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class DenseDilatedFeatureStack3D(nn.Module):
    """Dense dilated feature stack used by PRS-Net and PENonLocal."""

    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(out2)
        out4 = self.conv4(out1)
        return torch.cat((out1, out2, out3, out4), dim=1)


class DownsampleTime3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleTimeHead3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(channels[0], channels[1], kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(channels[1], channels[2], kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(channels[2], channels[3], kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(channels[3], channels[4], kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
        )

    def forward(self, x):
        return self.net(x)


class PixelWiseResidualShrinkageBlock3D(nn.Module):
    """Original PRS-Net-style pixel-wise residual shrinkage block."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.scales = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, t, h, w = x.shape
        residual = self.conv2(self.conv1(x))
        raw = residual.reshape(b * c, t, h, w)
        raw_abs = raw.abs()
        average = raw_abs.mean(dim=1)
        raw_max = raw_abs.max(dim=1, keepdim=True).values
        normalized = raw_abs / (raw_max + 1e-7)
        scale_input = normalized
        if t != 64:
            scale_input = F.interpolate(
                normalized.unsqueeze(1),
                size=(64, h, w),
                mode="trilinear",
                align_corners=False,
            ).squeeze(1)
        threshold = (average * self.scales(scale_input).squeeze(1)).unsqueeze(1)
        residual = torch.sign(raw) * F.relu(raw_abs - threshold)
        residual = residual.reshape(b, c, t, h, w)
        return self.out(x + residual)


class PRSNet3D(nn.Module):
    """3D PRS-Net architecture adapted from the original PyTorch repository."""

    def __init__(self, in_channels=1, base_channels=8, num_blocks=10):
        super().__init__()
        self.window = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.ReLU(inplace=True),
        )
        nn.init.constant_(self.window[0].weight, 1.0)
        self.window[0].weight.requires_grad_(False)

        self.features = DenseDilatedFeatureStack3D(in_channels, out_channels=2)
        self.compress = nn.Sequential(nn.Conv3d(8, 2, kernel_size=1), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(
            DownsampleTime3D(2, 4),
            DownsampleTime3D(4, 8),
            DownsampleTime3D(8, 16),
            DownsampleTime3D(16, 32),
        )
        self.blocks = nn.Sequential(*[PixelWiseResidualShrinkageBlock3D(32) for _ in range(int(num_blocks))])
        self.decoder = UpsampleTimeHead3D((32, 32, 16, 16, 4))
        self.head = nn.Conv3d(4, 1, kernel_size=1)

    def forward(self, x):
        x = self.window(x)
        x = self.compress(self.features(x))
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return self.head(x)


class OriginalNonLocal3D(nn.Module):
    """Compact Gaussian non-local block matching the PENonLocal repository."""

    def __init__(self, channels, use_scale=False, groups=1):
        super().__init__()
        self.use_scale = use_scale
        self.groups = groups
        self.theta = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.g = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.project = nn.Conv3d(channels, channels, kernel_size=1, groups=groups, bias=False)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        nn.init.constant_(self.norm.weight, 0.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def _kernel(self, theta, phi, value):
        b, c, d, h, w = theta.shape
        theta = theta.reshape(b, 1, c * d * h * w)
        phi = phi.reshape(b, 1, c * d * h * w)
        value = value.reshape(b, c * d * h * w, 1)
        attention = torch.bmm(phi, value)
        if self.use_scale:
            attention = attention / ((c * d * h * w) ** 0.5)
        out = torch.bmm(attention, theta)
        return out.reshape(b, c, d, h, w)

    def forward(self, x):
        residual = x
        theta = self.theta(x)
        phi = self.phi(x)
        value = self.g(x)

        if self.groups and self.groups > 1:
            chunks = []
            for theta_i, phi_i, value_i in zip(
                torch.chunk(theta, self.groups, dim=1),
                torch.chunk(phi, self.groups, dim=1),
                torch.chunk(value, self.groups, dim=1),
            ):
                chunks.append(self._kernel(theta_i, phi_i, value_i))
            x = torch.cat(chunks, dim=1)
        else:
            x = self._kernel(theta, phi, value)

        return self.norm(self.project(x)) + residual


class DenseFusionBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 16, 1), nn.ReLU(inplace=True))
        self.feat1 = nn.Sequential(nn.Conv3d(16, 8, 3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.feat2 = nn.Sequential(nn.Conv3d(16, 8, 3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv3d(24, 8, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        feat1 = self.feat1(x1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(x1)
        feat25 = self.feat25(feat2)
        fused = self.fuse(torch.cat((feat1, feat15, feat2, feat25), dim=1))
        return torch.cat((x, fused), dim=1)


class PENonLocal3D(nn.Module):
    """PENonLocal DeepBoosting architecture adapted from the original repo."""

    def __init__(self, in_channels=1, base_channels=8, num_blocks=10):
        super().__init__()
        self.features = DenseDilatedFeatureStack3D(in_channels, out_channels=2)
        self.compress = nn.Sequential(nn.Conv3d(8, 2, kernel_size=1), nn.ReLU(inplace=True))
        self.nonlocal_block = OriginalNonLocal3D(2, use_scale=False, groups=1)
        self.encoder = nn.Sequential(
            DownsampleTime3D(2, 4),
            DownsampleTime3D(4, 8),
            DownsampleTime3D(8, 16),
            DownsampleTime3D(16, 32),
        )

        channels = 32
        blocks = []
        for _ in range(int(num_blocks)):
            blocks.append(DenseFusionBlock3D(channels))
            channels += 8
        self.blocks = nn.Sequential(*blocks)
        self.decoder = UpsampleTimeHead3D((channels, channels // 2, channels // 4, max(channels // 8, 1), 7))
        self.head = nn.Conv3d(7, 1, kernel_size=1)

    def forward(self, x):
        x = self.compress(self.features(x))
        x = self.nonlocal_block(x)
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return self.head(x)


def _stin_group_count(channels, groups):
    if channels < groups:
        return 1
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class STINSingleConv3D(nn.Sequential):
    """Conv3d + GroupNorm + ReLU block used by DA-STIN."""

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=(1, 1, 1), groups=1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_stin_group_count(out_channels, groups), out_channels),
            nn.ReLU(inplace=True),
        )


class STInceptionBlock3D(nn.Module):
    """Spatio-temporal inception block from the STIN feature extractor."""

    def __init__(self, in_channels, out_channels, kernel_s=3, kernel_t=9, groups=1):
        super().__init__()
        branch_channels = out_channels // 4
        middle_channels = out_channels // 2
        spad_s = (kernel_s - 1) // 2
        pad_t = (kernel_t - 1) // 2

        self.branch1 = nn.Sequential(
            STINSingleConv3D(in_channels, middle_channels, (1, 1, 1), (0, 0, 0), groups=groups),
            STINSingleConv3D(middle_channels, middle_channels, (1, kernel_s, kernel_s), (0, spad_s, spad_s), groups=groups),
            STINSingleConv3D(middle_channels, branch_channels, (kernel_t, 1, 1), (pad_t, 0, 0), groups=groups),
        )
        self.branch2 = nn.Sequential(
            STINSingleConv3D(in_channels, middle_channels, (1, 1, 1), (0, 0, 0), groups=groups),
            STINSingleConv3D(middle_channels, middle_channels, (kernel_t, 1, 1), (pad_t, 0, 0), groups=groups),
            STINSingleConv3D(middle_channels, branch_channels, (1, kernel_s, kernel_s), (0, spad_s, spad_s), groups=groups),
        )
        self.branch3 = nn.Sequential(
            STINSingleConv3D(in_channels, middle_channels, (1, 1, 1), (0, 0, 0), groups=groups),
            STINSingleConv3D(middle_channels, middle_channels, (1, kernel_s, kernel_s), (0, spad_s, spad_s), groups=groups),
            STINSingleConv3D(middle_channels, middle_channels, (kernel_t, 1, 1), (pad_t, 0, 0), groups=groups),
            STINSingleConv3D(middle_channels, branch_channels, (1, kernel_s, kernel_s), (0, spad_s, spad_s), groups=groups),
        )
        self.branch4 = STINSingleConv3D(in_channels, branch_channels, (1, 1, 1), (0, 0, 0), groups=groups)

    def forward(self, x):
        return torch.cat(
            (
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ),
            dim=1,
        )


class STINFeatureExtractor3D(nn.Module):
    """Feature_extractor from DA-STIN: 8 ST-inception stages, 7 temporal pools."""

    def __init__(self, kernel_s=3, kernel_t=9):
        super().__init__()
        self.stage1 = STInceptionBlock3D(1, 4, kernel_s=kernel_s, kernel_t=kernel_t, groups=1)
        self.stage2 = STInceptionBlock3D(4, 8, kernel_s=kernel_s, kernel_t=kernel_t, groups=1)
        self.stage3 = STInceptionBlock3D(8, 12, kernel_s=kernel_s, kernel_t=kernel_t, groups=1)
        self.stage4 = STInceptionBlock3D(12, 16, kernel_s=kernel_s, kernel_t=kernel_t, groups=2)
        self.stage5 = STInceptionBlock3D(16, 24, kernel_s=kernel_s, kernel_t=kernel_t, groups=2)
        self.stage6 = STInceptionBlock3D(24, 32, kernel_s=kernel_s, kernel_t=kernel_t, groups=2)
        self.stage7 = STInceptionBlock3D(32, 40, kernel_s=kernel_s, kernel_t=kernel_t, groups=2)
        self.stage8 = STInceptionBlock3D(40, 48, kernel_s=kernel_s, kernel_t=kernel_t, groups=2)
        self.pool = nn.MaxPool3d(kernel_size=(2, 1, 1))

    def forward(self, x):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.pool(self.stage3(x))
        x = self.pool(self.stage4(x))
        x = self.pool(self.stage5(x))
        x = self.pool(self.stage6(x))
        x = self.pool(self.stage7(x))
        return self.stage8(x)


class STINTransposeConv3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(6, 3, 3), groups=2):
        super().__init__(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=(2, 1, 1),
                padding=(2, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(_stin_group_count(out_channels, groups), out_channels),
            nn.ReLU(inplace=True),
        )


class STINReconstructor3D(nn.Module):
    """Reconstructor from DA-STIN: 7 temporal transpose-conv stages."""

    def __init__(self, kernel_size=(6, 3, 3)):
        super().__init__()
        self.net = nn.Sequential(
            STINTransposeConv3D(48, 40, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(40, 32, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(32, 24, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(24, 16, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(16, 12, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(12, 8, kernel_size=kernel_size, groups=2),
            STINTransposeConv3D(8, 4, kernel_size=kernel_size, groups=2),
            nn.Conv3d(4, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class STIN3D(nn.Module):
    """DA-STIN model assembled from Feature_extractor and Reconstructor."""

    def __init__(self, in_channels=1, base_channels=8, num_blocks=10, min_time_bins=128):
        super().__init__()
        if in_channels != 1:
            raise ValueError("STIN3D expects one input channel containing SPAD counts.")
        self.min_time_bins = int(min_time_bins)
        self.feature_extractor = STINFeatureExtractor3D(kernel_s=3, kernel_t=9)
        self.reconstructor = STINReconstructor3D(kernel_size=(6, 3, 3))

    def forward(self, x):
        target_shape = x.shape[2:]
        if x.shape[2] < self.min_time_bins:
            x = F.interpolate(x, size=(self.min_time_bins, x.shape[3], x.shape[4]), mode="trilinear", align_corners=False)
        x = self.feature_extractor(x)
        x = self.reconstructor(x)
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        return x


class SPISRBackProjectionBlock3D(nn.Module):
    """A compact 3D back-projection block for SPISR."""

    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.error = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, hr_features, lr_reference, lr_size):
        lr_projection = F.interpolate(hr_features, size=lr_size, mode="trilinear", align_corners=False)
        lr_error = self.error(lr_reference - lr_projection)
        hr_error = F.interpolate(lr_error, size=hr_features.shape[2:], mode="trilinear", align_corners=False)
        return self.refine(torch.cat((self.up(hr_features), hr_error), dim=1))


class SPISRBackProjectionNet3D(nn.Module):
    """Self-supervised SPISR network.

    It maps a low-resolution photon counting cube to a higher-resolution cube.
    The training loss is supplied by the self-supervised PUKL/equivariance
    objective rather than by paired HR targets.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        num_blocks=4,
        time_scale=2,
        spatial_scale=2,
    ):
        super().__init__()
        self.time_scale = int(time_scale)
        self.spatial_scale = int(spatial_scale)
        c = int(base_channels)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([SPISRBackProjectionBlock3D(c) for _ in range(int(num_blocks))])
        self.tail = nn.Conv3d(c, 1, kernel_size=3, padding=1)

    def output_size(self, x):
        _, _, t, h, w = x.shape
        return (
            t * self.time_scale,
            h * self.spatial_scale,
            w * self.spatial_scale,
        )

    def forward(self, x):
        lr_size = x.shape[2:]
        hr_size = self.output_size(x)
        lr_features = self.head(x)
        hr_features = F.interpolate(lr_features, size=hr_size, mode="trilinear", align_corners=False)
        for block in self.blocks:
            hr_features = block(hr_features, lr_features, lr_size)
        return F.softplus(self.tail(hr_features)) + 1e-8


MODEL_REGISTRY = {
    "simple3d": SPAD3DHistogramNet,
    "prsnet": PRSNet3D,
    "penonlocal": PENonLocal3D,
    "stin": STIN3D,
}

SELF_SUPERVISED_MODEL_REGISTRY = {
    "spisr": SPISRBackProjectionNet3D,
}


def available_models():
    return tuple(MODEL_REGISTRY.keys())


def available_self_supervised_models():
    return tuple(SELF_SUPERVISED_MODEL_REGISTRY.keys())


def canonical_model_name(name):
    aliases = {
        "SPAD3DHistogramNet": "simple3d",
        "PRSNet3D": "prsnet",
        "PENonLocal3D": "penonlocal",
    }
    return aliases.get(name, name)


def build_model(name="simple3d", in_channels=1, base_channels=8, num_blocks=10, **kwargs):
    name = canonical_model_name(name)
    if name not in MODEL_REGISTRY:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{name}'. Available models: {choices}")

    model_cls = MODEL_REGISTRY[name]
    if name == "simple3d":
        return model_cls(in_channels=in_channels, base_channels=base_channels)
    return model_cls(
        in_channels=in_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        **kwargs,
    )


def build_self_supervised_model(
    name="spisr",
    in_channels=1,
    base_channels=16,
    num_blocks=4,
    time_scale=2,
    spatial_scale=2,
):
    if name not in SELF_SUPERVISED_MODEL_REGISTRY:
        choices = ", ".join(available_self_supervised_models())
        raise ValueError(f"Unknown self-supervised model '{name}'. Available models: {choices}")
    return SELF_SUPERVISED_MODEL_REGISTRY[name](
        in_channels=in_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        time_scale=time_scale,
        spatial_scale=spatial_scale,
    )

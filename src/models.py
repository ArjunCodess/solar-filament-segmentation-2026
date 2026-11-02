"""SegFormer/ConvNeXt ensemble members with a genuine full-resolution branch."""
import torch
from torch import nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())


class TinyUNet(nn.Module):
    """A runnable conventional U-Net baseline, also used for CPU smoke tests."""
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.mid = block(2, 16), block(16, 32), block(32, 64)
        self.dec2, self.dec1, self.head = block(96, 32), block(48, 16), nn.Conv2d(16, 3, 1)

    def forward(self, x, context=None):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        mid = self.mid(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(mid, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], 1))
        return self.head(self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], 1)))


class SpineGuidedModel(nn.Module):
    def __init__(self, architecture='segformer_b3', pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        kwargs = dict(in_channels=2, classes=32, encoder_weights='imagenet' if pretrained else None)
        if architecture == 'segformer_b3':
            self.core = smp.Segformer(encoder_name='mit_b3', **kwargs)
        elif architecture == 'convnext_unetpp':
            self.core = smp.UnetPlusPlus(encoder_name='tu-convnext_small', **kwargs)
        else:
            raise ValueError(architecture)
        self.native = block(2, 16)
        self.context = nn.Sequential(block(1, 16), nn.AdaptiveAvgPool2d(1))
        self.fuse = block(64, 32)
        self.head = nn.Conv2d(32, 3, 1)

    def forward(self, x, context):
        decoded = F.interpolate(self.core(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        full = self.native(x)
        global_context = self.context(context).expand(-1, -1, *x.shape[-2:])
        return self.head(self.fuse(torch.cat([decoded, full, global_context], 1)))


def build_model(architecture, pretrained=True):
    return TinyUNet() if architecture == 'unet' else SpineGuidedModel(architecture, pretrained)

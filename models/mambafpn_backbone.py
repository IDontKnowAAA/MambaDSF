"""
MambaFPN Backbone

MambaVision backbone with Enhanced FPN and PANet for multi-scale feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List
from collections import OrderedDict

from src.core import register


def detect_mambavision_available():
    """Check if MambaVision is available."""
    try:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        project_root = os.path.abspath(project_root)
        mamba_path = os.path.join(project_root, 'mambavision')
        if mamba_path not in sys.path:
            sys.path.insert(0, project_root)
        
        from mambavision.models.mamba_vision import MambaVision
        return True, MambaVision
    except ImportError as e:
        print(f"Failed to import MambaVision: {e}")
        return False, None


# ============================================================================
# Hybrid Block
# ============================================================================

class LocalBranch(nn.Module):
    """Local branch with multi-scale convolutions."""
    def __init__(self, dim):
        super().__init__()
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
    
    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        out = self.fuse(torch.cat([f3, f5], dim=1))
        return out


class ShortSequenceSSM(nn.Module):
    """Short sequence SSM with chunked processing."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv1d = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        x_flat = x.reshape(B, C, L).permute(0, 2, 1)
        
        num_chunks = (L + self.chunk_size - 1) // self.chunk_size
        pad_len = num_chunks * self.chunk_size - L
        
        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, 0, 0, pad_len))
        
        x_chunks = x_flat.reshape(B, num_chunks, self.chunk_size, C)
        x_chunks = x_chunks.reshape(B * num_chunks, self.chunk_size, C)
        
        xz = self.in_proj(x_chunks)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = x_branch.permute(0, 2, 1)
        x_branch = F.silu(self.conv1d(x_branch))
        x_branch = x_branch.permute(0, 2, 1)
        z = F.silu(z)
        out = self.out_proj(x_branch * z)
        out = self.norm(out)
        
        out = out.reshape(B, num_chunks, self.chunk_size, C)
        out = out.reshape(B, num_chunks * self.chunk_size, C)
        
        if pad_len > 0:
            out = out[:, :L, :]
        
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out


class HybridMambaBlock(nn.Module):
    """Hybrid block combining local convolutions with MLP."""
    def __init__(self, dim, chunk_size=64, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.local_branch = LocalBranch(dim)
        
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, dim, 1),
        )
        
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        x_norm = self.norm1(x)
        local_out = self.local_branch(x_norm)
        
        x = x + self.gamma1 * local_out
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        
        return x


# ============================================================================
# Shallow Feature Shortcut
# ============================================================================

class ShallowFeatureShortcut(nn.Module):
    """Shallow feature shortcut for preserving high-resolution details."""
    def __init__(self, shallow_dims, target_dim):
        super().__init__()
        
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, target_dim, 1, bias=False),
                nn.BatchNorm2d(target_dim),
            )
            for d in shallow_dims
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(target_dim * len(shallow_dims), target_dim, 1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.GELU(),
        )
        
        self.weight = nn.Parameter(torch.ones(1) * 0.3)
    
    def forward(self, shallow_feats, target_size):
        projected = []
        for feat, proj in zip(shallow_feats, self.projs):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        
        fused = self.fuse(torch.cat(projected, dim=1))
        return fused * self.weight


# ============================================================================
# Enhanced FPN Blocks
# ============================================================================

class ContrastEnhanceBlock(nn.Module):
    """Channel attention block for contrast enhancement."""
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        ca = self.channel_attn(x)
        x_ca = x * ca
        x_enhanced = self.spatial_enhance(x_ca)
        return x + self.gamma * x_enhanced


class EdgeAttentionBlock(nn.Module):
    """Edge attention block for boundary enhancement."""
    def __init__(self, channels):
        super().__init__()
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        edge = self.edge_conv(x)
        attn = self.attn(edge.abs())
        transformed = self.transform(x)
        out = x + attn * transformed
        return out


class SmallTargetEnhancer(nn.Module):
    """Multi-scale feature enhancer with ASPP-style dilated convolutions."""
    def __init__(self, channels):
        super().__init__()
        
        dilations = [1, 3, 5]
        branch_ch = channels // 4
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        aspp_outs = [aspp(x) for aspp in self.aspp]
        concat = torch.cat(aspp_outs, dim=1)
        fused = self.fuse(concat)
        ca = self.channel_attn(fused)
        return x + ca * fused


class EnhancedFPNBlock(nn.Module):
    """Enhanced FPN block with attention modules."""
    def __init__(self, channels, use_all_modules=True):
        super().__init__()
        
        modules = [
            ContrastEnhanceBlock(channels),
            EdgeAttentionBlock(channels),
        ]
        
        if use_all_modules:
            modules.append(SmallTargetEnhancer(channels))
        
        self.modules_list = nn.ModuleList(modules)
        self.final_norm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return self.final_norm(x)


# ============================================================================
# MambaFPN Backbone
# ============================================================================

@register()
class MambaFPNBackbone(nn.Module):
    """MambaFPN backbone with Enhanced FPN and PANet for multi-scale feature extraction."""
    
    __share__ = ['pretrained']
    
    def __init__(
        self,
        variant: str = 'mamba_vision_T',
        out_channels: List[int] = [256, 256, 256],
        return_idx: List[int] = [1, 2, 3],
        fpn_dim: int = 256,
        pretrained: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        custom_depths: List[int] = None,
        chunk_size: int = 64,
        use_hybrid_block: bool = True,
        use_shallow_shortcut: bool = True,
        use_enhanced_fpn: bool = True,
    ):
        super().__init__()
        
        self.variant = variant
        self.out_channels = out_channels
        self.return_idx = return_idx
        self.fpn_dim = fpn_dim
        self.pretrained = pretrained
        self.freeze_at = freeze_at
        self.freeze_norm = freeze_norm
        self.custom_depths = custom_depths
        self.chunk_size = chunk_size
        self.use_hybrid_block = use_hybrid_block
        self.use_shallow_shortcut = use_shallow_shortcut
        self.use_enhanced_fpn = use_enhanced_fpn
        
        if variant in ['mamba_vision_T', 'mamba_vision_T2']:
            self.level_channels = [80, 160, 320, 640]
        elif variant == 'mamba_vision_S':
            self.level_channels = [96, 192, 384, 768]
        elif variant == 'mamba_vision_B':
            self.level_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        self._init_mambavision(variant, pretrained, custom_depths)
        
        if use_hybrid_block:
            self.hybrid_blocks = nn.ModuleDict({
                'stage2': HybridMambaBlock(self.level_channels[3], chunk_size),
                'stage3': HybridMambaBlock(self.level_channels[3], chunk_size),
            })
        
        if use_shallow_shortcut:
            self.shallow_shortcut = ShallowFeatureShortcut(
                shallow_dims=[self.level_channels[0], self.level_channels[1]],
                target_dim=fpn_dim
            )
        
        self.lateral_convs = nn.ModuleList()
        for i in range(4):
            src_ch = self.level_channels[i]
            lateral = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(src_ch, fpn_dim, 1, bias=False)),
                ('norm', nn.BatchNorm2d(fpn_dim))
            ]))
            self.lateral_convs.append(lateral)
        
        if use_enhanced_fpn:
            self.fpn_blocks = nn.ModuleList([
                EnhancedFPNBlock(fpn_dim, use_all_modules=True)
                for _ in range(3)
            ])
            
            self.pan_downsample = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(2)
            ])
            
            self.pan_blocks = nn.ModuleList([
                EnhancedFPNBlock(fpn_dim, use_all_modules=True)
                for _ in range(2)
            ])
        else:
            self.fpn_convs = nn.ModuleList()
            for i in range(3):
                fpn_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(fpn_dim)),
                    ('act', nn.ReLU(inplace=True))
                ]))
                self.fpn_convs.append(fpn_conv)
            
            self.pan_downsample_convs = nn.ModuleList()
            self.pan_convs = nn.ModuleList()
            for i in range(3):
                down_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(fpn_dim, fpn_dim, 3, stride=2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(fpn_dim)),
                    ('act', nn.ReLU(inplace=True))
                ]))
                self.pan_downsample_convs.append(down_conv)
                
                pan_conv = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(fpn_dim)),
                    ('act', nn.ReLU(inplace=True))
                ]))
                self.pan_convs.append(pan_conv)
        
    def _init_mambavision(self, variant: str, pretrained: bool, custom_depths: List[int] = None):
        """Initialize MambaVision backbone."""
        available, MambaVision = detect_mambavision_available()
        
        if not available:
            raise ImportError("Failed to import MambaVision")
        
        configs = {
            'mamba_vision_T': {
                'dim': 80, 'in_dim': 32,
                'depths': [1, 3, 8, 4],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
            },
            'mamba_vision_T2': {
                'dim': 80, 'in_dim': 32,
                'depths': [1, 3, 11, 4],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
            },
            'mamba_vision_S': {
                'dim': 96, 'in_dim': 64,
                'depths': [3, 3, 7, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
            },
            'mamba_vision_B': {
                'dim': 128, 'in_dim': 64,
                'depths': [3, 3, 10, 5],
                'num_heads': [2, 4, 8, 16],
                'window_size': [8, 8, 14, 7],
            },
        }
        
        config = configs[variant]
        
        depths = custom_depths if custom_depths is not None else config['depths']
        
        self.mamba_model = MambaVision(
            dim=config['dim'],
            in_dim=config['in_dim'],
            depths=depths,
            window_size=config['window_size'],
            mlp_ratio=4.0,
            num_heads=config['num_heads'],
            drop_path_rate=0.2,
            resolution=224,
            in_chans=3,
            num_classes=1000
        )
        
        if pretrained:
            self._load_pretrained_weights(variant)
    
    def _load_pretrained_weights(self, variant: str):
        """Load MambaVision pretrained weights."""
        variant_suffix = {
            'mamba_vision_T': 'tiny',
            'mamba_vision_T2': 'tiny2',
            'mamba_vision_S': 'small',
            'mamba_vision_B': 'base',
        }.get(variant, 'tiny')
        
        possible_paths = [
            f"/home/featurize/work/pretrained_weights/mambavision_{variant_suffix}_1k.pth.tar",
            f"weights/mambavision_{variant_suffix}_1k.pth.tar",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.mamba_model.load_state_dict(state_dict, strict=False)
                return
    
    def _extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from MambaVision backbone."""
        stem = self.mamba_model.patch_embed(x)
        
        mamba_levels = []
        x_current = stem
        for i, stage in enumerate(self.mamba_model.levels):
            x_current = stage(x_current)
            
            if self.use_hybrid_block and i >= 2:
                stage_name = f'stage{i}'
                if stage_name in self.hybrid_blocks:
                    x_current = self.hybrid_blocks[stage_name](x_current)
            
            mamba_levels.append(x_current)
        
        return {
            'level0': stem,
            'level1': mamba_levels[0],
            'level2': mamba_levels[1],
            'level3': mamba_levels[3],
            'shallow_feats': [stem, mamba_levels[0]],
        }
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        backbone_features = self._extract_features(x)
        
        lateral_features = {}
        for i in range(4):
            lateral_features[f"L{i+2}"] = self.lateral_convs[i](backbone_features[f"level{i}"])
        
        if self.use_shallow_shortcut:
            target_size = lateral_features['L4'].shape[2:]
            shallow_contrib = self.shallow_shortcut(
                backbone_features['shallow_feats'], 
                target_size
            )
        
        if self.use_enhanced_fpn:
            P5 = lateral_features['L5']
            
            P4_up = F.interpolate(P5, size=lateral_features['L4'].shape[2:], mode='nearest')
            P4 = self.fpn_blocks[0](P4_up + lateral_features['L4'])
            
            P3_up = F.interpolate(P4, size=lateral_features['L3'].shape[2:], mode='nearest')
            P3 = self.fpn_blocks[1](P3_up + lateral_features['L3'])
            
            P2_up = F.interpolate(P3, size=lateral_features['L2'].shape[2:], mode='nearest')
            P2 = self.fpn_blocks[2](P2_up + lateral_features['L2'])
            
            N2 = P2
            
            N3_down = self.pan_downsample[0](N2)
            N3 = self.pan_blocks[0](N3_down + P3)
            
            if self.use_shallow_shortcut:
                N3 = N3 + F.interpolate(shallow_contrib, size=N3.shape[2:], mode='bilinear', align_corners=False)
            
            N4_down = self.pan_downsample[1](N3)
            N4 = self.pan_blocks[1](N4_down + P4)
            
            N5 = P5
            
            outputs = [N3, N4, N5]
        else:
            fpn_features = {'P5': lateral_features['L5']}
            for idx in range(4, 1, -1):
                upper_p = fpn_features[f"P{idx+1}"]
                lateral = lateral_features[f"L{idx}"]
                upsampled = F.interpolate(upper_p, size=lateral.shape[2:], mode='nearest')
                fused = upsampled + lateral
                conv_idx = 5 - idx - 1
                fpn_features[f"P{idx}"] = self.fpn_convs[conv_idx](fused)
            
            pan_features = {'N2': fpn_features['P2']}
            for i in range(2, 5):
                lower_n = pan_features[f"N{i}"]
                fpn_feat = fpn_features[f"P{i+1}"]
                downsampled = self.pan_downsample_convs[i - 2](lower_n)
                fused = downsampled + fpn_feat
                pan_features[f"N{i+1}"] = self.pan_convs[i - 2](fused)
            
            outputs = [pan_features['N3'], pan_features['N4'], pan_features['N5']]
        
        return outputs

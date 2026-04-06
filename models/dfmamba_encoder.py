"""
DFMamba Encoder (Dilated Fusion Mamba)

Multi-scale dilated attention with SSM-based cross-scale fusion.
"""
import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

from ...core import register

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, DFMambaEncoder will not work")


__all__ = ['DFMambaEncoder']


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention (ICASSP 2023)
    
    Models attention along both height and width directions independently,
    which is particularly effective for detecting objects with clear geometric edges (e.g., cube).
    
    Reference: https://arxiv.org/abs/2305.13563
    """
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0, f"channels({channels}) must be divisible by factor({factor})"
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool along width -> (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pool along height -> (B, C, 1, W)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # Split into groups
        group_x = x.reshape(b * self.groups, -1, h, w)  # (b*g, c//g, h, w)
        
        # Axial pooling: capture H and W directional information
        x_h = self.pool_h(group_x)  # (b*g, c//g, h, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # (b*g, c//g, w, 1) -> (b*g, c//g, 1, w) transposed
        
        # Fuse H and W information
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # (b*g, c//g, h+w, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        # Apply attention weights
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        
        # Cross-attention between two branches
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # (b*g, c//g, h*w)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # (b*g, c//g, h*w)
        
        # Compute attention weights and apply
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class DilateAttention(nn.Module):
    """Single dilation rate sliding window dilated attention"""
    def __init__(self, head_dim, qk_scale=None, attn_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        padding = dilation * (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride=1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        """
        Args:
            q, k, v: [B, C, H, W]
        Returns:
            x: [B, H, W, C]
        """
        B, d, H, W = q.shape
        
        q = q.reshape(B, d // self.head_dim, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(
            B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 2, 3)
        
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        v = self.unfold(v).reshape(
            B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 3, 2)
        
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MSDALayer(nn.Module):
    """
    MSDA Layer with FFN - outputs 4 dilation branches (without merging)
    
    Structure: LN -> MSDA -> Residual -> LN -> FFN -> Residual
    Now matches YOLO version's MSDABlock design.
    """
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None,
        attn_drop=0., 
        proj_drop=0., 
        kernel_size=3, 
        dilation=[1, 2, 3, 4],
        mlp_ratio=4.0,
        drop=0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_dilation = len(dilation)
        
        assert num_heads % self.num_dilation == 0, \
            f"num_heads({num_heads}) must be divisible by num_dilation({self.num_dilation})"
        
        self.branch_dim = dim // self.num_dilation
        
        # MSDA attention components
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])
        
        self.branch_projs = nn.ModuleList([
            nn.Linear(self.branch_dim, self.branch_dim)
            for _ in range(self.num_dilation)
        ])
        
        self.proj_drop = nn.Dropout(proj_drop)
        
        # FFN components (matching YOLO version)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # EMA: Efficient Multi-scale Attention for edge enhancement
        # Particularly effective for geometric objects like cube
        self.ema = EMA(channels=dim, factor=8)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            branches: list of [B, C//4, H, W], 4 dilation branches
        """
        B, C, H, W = x.shape
        
        # Step 1: LayerNorm before MSDA
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm1(x_norm)
        x_norm = rearrange(x_norm, 'b h w c -> b c h w')
        
        # Step 2: MSDA attention
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, 3, self.num_dilation, self.branch_dim, H, W)
        qkv = qkv.permute(2, 1, 0, 3, 4, 5)
        
        branches = []
        for i in range(self.num_dilation):
            q_i, k_i, v_i = qkv[i][0], qkv[i][1], qkv[i][2]
            
            out_i = self.dilate_attention[i](q_i, k_i, v_i)
            out_i = self.branch_projs[i](out_i)
            out_i = self.proj_drop(out_i)
            
            # Residual connection for MSDA
            x_branch = x[:, i*self.branch_dim:(i+1)*self.branch_dim, :, :]
            out_i = rearrange(out_i, 'b h w c -> b c h w') + x_branch
            
            branches.append(out_i)
        
        # Step 3: FFN on merged branches
        # Merge branches temporarily for FFN
        merged = torch.cat(branches, dim=1)  # [B, C, H, W]
        merged_ln = rearrange(merged, 'b c h w -> b h w c')
        merged_ln = self.norm2(merged_ln)
        merged_ffn = self.mlp(merged_ln)  # FFN
        merged_out = merged_ln + merged_ffn  # Residual
        merged_out = rearrange(merged_out, 'b h w c -> b c h w')
        
        # Step 4: Apply EMA for edge enhancement
        merged_out = self.ema(merged_out)
        
        # Step 5: Split back to branches
        branches_out = []
        for i in range(self.num_dilation):
            branch_i = merged_out[:, i*self.branch_dim:(i+1)*self.branch_dim, :, :]
            branches_out.append(branch_i)
        
        return branches_out


class BranchFusSSM(nn.Module):
    """
    Single branch FusSSM - handles cross-scale fusion for one dilation branch
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=1.0,
        dropout=0.,
        bias=False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        
        self.in_proj_x = nn.Linear(d_model, self.d_inner, bias=bias)
        self.in_proj_y = nn.Linear(d_model, self.d_inner, bias=bias)
        
        self.x_proj_weight = nn.Parameter(
            torch.randn(4, self.dt_rank + self.d_state * 2, self.d_inner) * 0.02
        )
        
        self.dt_projs_weight = nn.Parameter(torch.randn(4, self.d_inner, self.dt_rank) * 0.02)
        self.dt_projs_bias = nn.Parameter(torch.zeros(4, self.d_inner))
        self._init_dt_bias()
        
        self.A_logs = self._init_A_log(self.d_state, self.d_inner, copies=4)
        self.Ds = self._init_D(self.d_inner, copies=4)
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.selective_scan = selective_scan_fn if MAMBA_AVAILABLE else None

    def _init_dt_bias(self, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt.unsqueeze(0).expand(4, -1))

    def _init_A_log(self, d_state, d_inner, copies=4):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=d_inner
        ).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> r d n", r=copies).flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def _init_D(self, d_inner, copies=4):
        D = torch.ones(d_inner)
        D = repeat(D, "n -> r n", r=copies).flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x, y):
        """
        Args:
            x: [B, C, H, W] - modulation source
            y: [B, C, H, W] - fusion target
        Returns:
            out: [B, C, H, W] - fused features
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        y_flat = rearrange(y, 'b c h w -> b (h w) c')
        
        x_inner = self.in_proj_x(x_flat)
        y_inner = self.in_proj_y(y_flat)
        
        x_inner_2d = rearrange(x_inner, 'b (h w) c -> b c h w', h=H, w=W)
        y_inner_2d = rearrange(y_inner, 'b (h w) c -> b c h w', h=H, w=W)
        
        x_hwwh = torch.stack([
            x_inner_2d.view(B, -1, L),
            x_inner_2d.transpose(2, 3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
        y_hwwh = torch.stack([
            y_inner_2d.view(B, -1, L),
            y_inner_2d.transpose(2, 3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        ys_flat = ys.float().view(B, -1, L)
        dts_flat = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            ys_flat, dts_flat,
            As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        invwh_y = inv_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        
        y_merged = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y_merged = rearrange(y_merged, 'b c (h w) -> b h w c', h=H, w=W)
        
        y_merged = self.out_norm(y_merged)
        out = self.out_proj(y_merged)
        out = self.dropout(out)
        out = rearrange(out, 'b h w c -> b c h w')
        
        return out


class CrossScaleBranchFusion(nn.Module):
    """Cross-scale fusion module for a single dilation branch."""
    def __init__(self, branch_dim, d_state=16, expand=1.0, dropout=0.):
        super().__init__()
        self.branch_dim = branch_dim
        
        self.fus_ssm_up = BranchFusSSM(branch_dim, d_state, expand, dropout)
        self.fus_ssm_down = BranchFusSSM(branch_dim, d_state, expand, dropout)
        
        # Learnable scale for residual connection
        self.fusion_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, x_up=None, x_down=None):
        """
        Args:
            x: [B, branch_dim, H, W] - current scale branch features
            x_up: [B, branch_dim, H, W] - features from higher resolution (aligned)
            x_down: [B, branch_dim, H, W] - features from lower resolution (aligned)
        Returns:
            out: [B, branch_dim, H, W] - fused features
        """
        # Initialize with ones for multiplicative fusion
        # This way: if only one source, result = that source
        # if two sources, result = source1 * source2 (element-wise)
        fused = torch.ones_like(x)
        num_fusions = 0
        
        if x_up is not None:
            # Apply sigmoid to bound the FusSSM output to (0, 1) for stable multiplication
            up_gate = torch.sigmoid(self.fus_ssm_up(x_up, x))
            fused = fused * up_gate  # Multiplicative fusion
            num_fusions += 1
        
        if x_down is not None:
            # Apply sigmoid to bound the FusSSM output to (0, 1) for stable multiplication
            down_gate = torch.sigmoid(self.fus_ssm_down(x_down, x))
            fused = fused * down_gate  # Multiplicative fusion
            num_fusions += 1
        
        if num_fusions > 0:
            # Scale the gated features back to original magnitude
            # The multiplication with x applies the gate to the original features
            fused = x * fused
        else:
            fused = x
        
        # Residual connection: blend original with fused
        out = (1 - self.fusion_scale) * x + self.fusion_scale * fused
        
        return out


@register()
class DFMambaEncoder(nn.Module):
    """DFMamba Encoder with multi-scale dilated attention and SSM-based cross-scale fusion."""
    __share__ = ['eval_spatial_size', ]
    
    def __init__(
        self,
        in_channels=[256, 256, 256],  # Optimized: direct 256 from backbone
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_heads=8,
        dilation=[1, 2, 3, 4],
        d_state=16,
        expand=1.5,
        dropout=0.,
        kernel_size=3,
        mlp_ratio=4.0,  # FFN expansion ratio
        eval_spatial_size=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels)
        self.num_dilation = len(dilation)
        self.branch_dim = hidden_dim // self.num_dilation
        self.eval_spatial_size = eval_spatial_size
        
        # Output interface (same as HybridEncoder)
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # Input projection: now simplified since backbone outputs 256 directly
        # Still keep projection for flexibility (e.g., if using different backbone)
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel == hidden_dim:
                # Identity projection when channels match (optimized path)
                proj = nn.Identity()
            else:
                # Full projection when channels differ
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            self.input_proj.append(proj)
        
        # MSDA layers with FFN (one per scale)
        self.msda_layers = nn.ModuleList([
            MSDALayer(
                dim=hidden_dim,
                num_heads=num_heads,
                dilation=dilation,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,  # FFN enabled
            )
            for _ in range(self.num_scales)
        ])
        
        # Cross-scale fusion (one group per dilation branch)
        self.branch_fusions = nn.ModuleList([
            nn.ModuleList([
                CrossScaleBranchFusion(self.branch_dim, d_state, expand, dropout)
                for _ in range(self.num_scales)
            ])
            for _ in range(self.num_dilation)
        ])
        
        # Spatial alignment modules
        self.align_up = nn.ModuleDict({
            '1to0': self._make_upsample(self.branch_dim, 2),
            '2to1': self._make_upsample(self.branch_dim, 2),
        })
        
        self.align_down = nn.ModuleDict({
            '0to1': self._make_downsample(self.branch_dim, 2),
            '1to2': self._make_downsample(self.branch_dim, 2),
        })
        
        # Final LayerNorm for residual
        self.final_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.num_scales)
        ])
        
        # Learnable branch weights (smaller dilation = higher weight for localization)
        self.branch_scales = nn.Parameter(torch.tensor([1.0, 0.8, 0.6, 0.4]))
        
        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            if hasattr(proj, 'conv'):
                nn.init.xavier_uniform_(proj.conv.weight)

    def _make_upsample(self, channels, scale):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def _make_downsample(self, channels, scale):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, feats):
        """
        Args:
            feats: list of [B, C, H, W], backbone outputs [C3, C4, C5]
                   C3: [B, 512, H/8, W/8]
                   C4: [B, 1024, H/16, W/16]
                   C5: [B, 2048, H/32, W/32]
        Returns:
            outputs: list of [B, 256, H, W], [P3, P4, P5]
        """
        assert len(feats) == self.num_scales
        
        # Step 0: Input projection to unified channels
        proj_feats = [self.input_proj[i](feats[i]) for i in range(self.num_scales)]
        
        # Step 1: MSDA dilated attention -> 4 branches per scale
        all_branches = []
        for i, feat in enumerate(proj_feats):
            branches = self.msda_layers[i](feat)
            all_branches.append(branches)
        
        # Step 2: FusSSM cross-scale fusion (4x, one per dilation branch)
        fused_branches = [[] for _ in range(self.num_scales)]
        
        for d_idx in range(self.num_dilation):
            # Get current dilation branch from all scales
            p3_branch = all_branches[0][d_idx]  # [B, 64, H/8, W/8]
            p4_branch = all_branches[1][d_idx]  # [B, 64, H/16, W/16]
            p5_branch = all_branches[2][d_idx]  # [B, 64, H/32, W/32]
            
            # P3 fusion: receives P4 info
            p3_from_p4 = self.align_up['1to0'](p4_branch)
            e3_branch = self.branch_fusions[d_idx][0](
                p3_branch, x_up=None, x_down=p3_from_p4
            )
            fused_branches[0].append(e3_branch)
            
            # P4 fusion: receives P3 and P5 info
            p4_from_p3 = self.align_down['0to1'](p3_branch)
            p4_from_p5 = self.align_up['2to1'](p5_branch)
            e4_branch = self.branch_fusions[d_idx][1](
                p4_branch, x_up=p4_from_p3, x_down=p4_from_p5
            )
            fused_branches[1].append(e4_branch)
            
            # P5 fusion: receives P4 info
            p5_from_p4 = self.align_down['1to2'](p4_branch)
            e5_branch = self.branch_fusions[d_idx][2](
                p5_branch, x_up=p5_from_p4, x_down=None
            )
            fused_branches[2].append(e5_branch)
        
        # Step 3: Merge branches + residual
        outputs = []
        for i in range(self.num_scales):
            # Apply learnable branch weights
            scaled_branches = [
                self.branch_scales[d_idx] * fused_branches[i][d_idx]
                for d_idx in range(self.num_dilation)
            ]
            # Merge 4 weighted branches
            merged = torch.cat(scaled_branches, dim=1)
            
            # Residual connection with projected features
            merged = merged + proj_feats[i]
            
            # LayerNorm
            merged = rearrange(merged, 'b c h w -> b h w c')
            merged = self.final_norms[i](merged)
            merged = rearrange(merged, 'b h w c -> b c h w')
            
            outputs.append(merged)
        
        return outputs

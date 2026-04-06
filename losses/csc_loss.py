"""
CSC Loss: Cross-Scale Semantic Consistency Loss

Enforces semantic consistency across encoder outputs at different scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSCLoss(nn.Module):
    """Cross-Scale Semantic Consistency Loss."""
    
    def __init__(self, feat_strides=(8, 16, 32)):
        """
        Args:
            feat_strides: Feature stride for each scale level.
        """
        super().__init__()
        self.feat_strides = feat_strides
        
        self.pairs = []
        for i in range(len(feat_strides)):
            for j in range(i + 1, len(feat_strides)):
                self.pairs.append((i, j))
    
    def _sample_feature(self, feat_map, cx_norm, cy_norm):
        """Sample feature at normalized coordinates."""
        B, C, H, W = feat_map.shape
        
        nx = 2.0 * cx_norm - 1.0
        ny = 2.0 * cy_norm - 1.0
        
        grid = torch.stack([
            torch.full((B,), nx, device=feat_map.device, dtype=feat_map.dtype),
            torch.full((B,), ny, device=feat_map.device, dtype=feat_map.dtype)
        ], dim=-1).view(B, 1, 1, 2)
        
        sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return sampled.view(B, C)
    
    def forward(self, encoder_feats, targets):
        """Compute CSC loss."""
        device = encoder_feats[0].device
        B = encoder_feats[0].shape[0]
        
        total_sim = 0.0
        count = 0
        
        for b in range(B):
            boxes = targets[b].get('boxes', None)
            if boxes is None or len(boxes) == 0:
                continue
            
            for box_idx in range(len(boxes)):
                cx = boxes[box_idx, 0].item()
                cy = boxes[box_idx, 1].item()
                
                features = []
                for scale_idx, feat in enumerate(encoder_feats):
                    f = self._sample_feature(
                        feat[b:b+1], cx, cy
                    ).squeeze(0)
                    f = F.normalize(f, dim=0)
                    features.append(f)
                
                for (i, j) in self.pairs:
                    sim = (features[i].float() * features[j].float()).sum()
                    total_sim += sim
                    count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        avg_sim = total_sim / count
        loss = 1.0 - avg_sim
        
        # Handle NaN for numerical stability
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        loss = loss.clamp(min=0.0, max=2.0)
        
        return loss

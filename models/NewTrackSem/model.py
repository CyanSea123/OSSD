import torch

from .backbone import DGCNN
from .transformer import Transformer
from models.base_model import BaseModel
from .rpn import RPN
from transformers import RobertaModel, ViTFeatureExtractor, ViTModel
import torch.nn as nn
import torch.nn.functional as F

"""
Text-Guided Multi-Granularity Clustering Module
Compatible with your NewTrackSem pipeline.

Provides a single class `TextGuidedCluster` that supports two modes:
 - mode='soft' : Text-guided SoftCluster (lightweight)
 - mode='slot' : Text-guided Slot Attention (iterative, stronger semantic coupling)

Also provides multi-granularity fusion: point-level + cluster-level cross-attention with gating.

API:
    module = TextGuidedCluster(feat_dim=C, num_clusters=K, mode='slot', num_iters=3, use_multigranularity=True)
    fused_points, clusters, assign = module(point_feats, text_feats, tau)

Inputs:
    point_feats: (B, T, N, C)
    text_feats:  (B, T, L, C)
    tau: temperature for gumbel-softmax when using 'soft' mode (unused for slot)

Outputs:
    fused_points: (B, T, N, C)  # fused point-level features after multi-granularity fusion
    clusters:     (B, T, K, C)  # cluster / slot features
    assign:       (B, T, N, K)  # soft assignment from points to clusters (if available; for slot it's attention-derived)

This file aims to be drop-in and easy to adapt. It has a small self-test at the bottom.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math




# --------------- 辅助小模块 ---------------
class _MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x): return self.net(x)


# ------------------------------
# 1. Slot Attention with Orthogonal Loss
# ------------------------------
class GranularityOrthogonalLoss(torch.nn.Module):
    """
    Residual-based orthogonal disentanglement loss for multi-granularity representation.
    Includes:
        1. Cluster orthogonality (avoid mode collapse)
        2. Residual orthogonality (ensure hierarchical disentanglement)
    """

    def __init__(self, lambda_cluster=1, lambda_residual=1, detach_assign=False):
        super().__init__()
        self.lambda_cluster = lambda_cluster
        self.lambda_residual = lambda_residual
        self.detach_assign = detach_assign
        self.margin = 0.0

    def forward(self, clusters, points, assign):
        """
        clusters: (B, T, K, C) - coarse-grained slot features
        points: (B, T, N, C) - fine-grained point features
        assign: (B, T, N, K) - soft assignment
        """
        B, T, K, C = clusters.shape
        _, _, N, _ = points.shape
        # 自适应阈值
        effective_indices=K


        # 找出有效slot
        slot_norms = torch.norm(clusters, dim=-1).mean(dim=(0, 1))  # (K,)
        slot_usage = assign.mean(dim=(0, 1, 2))  # 每个slot的平均分配概率 (K,)

        # 同时考虑特征范数和分配概率
        combined_threshold = 1e-2  # 更宽松的范数阈值
        usage_threshold = 0.05  # 分配概率阈值

        # 无效slot：范数小 OR 分配概率低
        mask = (slot_norms > combined_threshold) & (slot_usage > usage_threshold)
        effective_count = mask.sum().item()

        # 安全保护：如果有效slot太少，使用所有slot
        if effective_count < 2:

            effective_indices = torch.arange(K, device=clusters.device)
            effective_clusters = clusters  # 使用所有slot
            effective_K = K
            ll=0
        else:
            effective_indices = torch.where(mask)[0]
            effective_clusters = clusters[:, :, mask, :]  # 只使用有效slot
            effective_K = effective_clusters.shape[2]
            ll=1


        # === 1. cluster 内部正交性 ===
        clusters_norm = F.normalize(effective_clusters, dim=-1)
        sim = torch.einsum('btkc,btlc->btkl', clusters_norm, clusters_norm)
        identity = torch.eye(effective_K, device=clusters.device).unsqueeze(0).unsqueeze(0)
        #cluster_orth_loss = ((sim * (1 - identity)) ** 2).mean()
        off_diag = sim * (1 - identity)
        cluster_orth_loss = F.relu(off_diag.abs() - self.margin).mean()  # 只惩罚超过margin的相似度

        # === 2. 残差正交性 ===

        assign_detached = assign
        clusters_detached = clusters




        assigned_clusters = torch.einsum('btnk,btkc->btnc', assign_detached, clusters_detached)
        residuals = points - assigned_clusters

        residuals_norm = F.normalize(residuals, dim=-1)
        clusters_norm = F.normalize(clusters_detached, dim=-1)
        sim_res = torch.einsum('btnc,btkc->btnk', residuals_norm, clusters_norm)
        residual_orth_loss = (sim_res ** 2).mean()

        # === 总损失 ===
        total_loss = 0*self.lambda_cluster * cluster_orth_loss + self.lambda_residual * residual_orth_loss



        return total_loss, cluster_orth_loss, residual_orth_loss,effective_count


class SlotAttentionBidirectional(nn.Module):
    """
    Slot Attention with optional orthogonal loss.
    Returns:
      slots: (B, T, K, C) - coarse-grained features
      assign: (B, T, N, K) - soft assignment
      orth_loss (optional): scalar
    """

    def __init__(self, dim, K, iters=3, mlp_hid=128, n_heads=4, use_orth_loss=False):
        super().__init__()



        self.dim = dim
        self.K = K
        self.iters = iters
        self.use_orth_loss = use_orth_loss

        self.norm_slots = nn.LayerNorm(dim)
        self.norm_points = nn.LayerNorm(dim)
        self.norm_temporal = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hid), nn.ReLU(), nn.Linear(mlp_hid, dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # Motion prediction for temporal coherence

        #Spatiotemporal attention
        self.temporal_attns = nn.ModuleList([
           nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=0.1)
           for _ in range(2)
        ])


        self.orth_loss_module = GranularityOrthogonalLoss(lambda_cluster=1, lambda_residual=2)




    def temporal_attention(self, slots, temporal_window=3):
       """
       纯几何的时序注意力，关注运动模式
      slots: (B*T, K, C)
       """
       B_T, K, C = slots.shape
       T = temporal_window
       B = B_T // temporal_window



       # Reshape to include temporal dimension
       slots_temp = slots.view(B, T, K, C)  # (B, T, K, C)

       for attn_layer in self.temporal_attns:
           # 时序自注意力

           slots_norm = self.norm_temporal(slots_temp)
           slots_reshaped = slots_norm.permute(0, 2, 1, 3).contiguous()  # (B, K, T, C)
           slots_flat = slots_reshaped.view(-1, T, C).permute(1, 0, 2)  # (T, B*K, C)

           attended_slots, _ = attn_layer(slots_flat, slots_flat, slots_flat)
           attended_slots = attended_slots.permute(1, 0, 2).view(B, K, T, C)
           attended_slots = attended_slots.permute(0, 2, 1, 3).contiguous()  # (B, T, K, C)

           # 残差连接
           slots_temp = slots_temp + self.alpha * attended_slots

       return slots_temp.view(B_T, K, C)



    def forward(self, points, slots_init, text,points_3d,tau):
        """
        points: (B, T, N, C)
        slots_init: (B, T, K, C)
        text: (B, T, L, C) or None
        """

        B, T, N, C = points.shape
        K = self.K
        pts = points.view(B * T, N, C)
        slots = slots_init.view(B * T, K, C)

        for iter_idx in range(self.iters):
            # 空间注意力
            pts_n = self.norm_points(pts)
            sl_n = self.norm_slots(slots)

            q = self.to_q(sl_n)
            k = self.to_k(pts_n)
            v = self.to_v(pts_n)

            scores = torch.einsum('bkc, bnc -> bkn', q, k)

            attn = F.softmax(scores / math.sqrt(C), dim=1)
            attn_sum = attn.sum(dim=2, keepdim=True) + 1e-8
            attn_normalized = attn / attn_sum
            updates = torch.einsum('bkn, bnc -> bkc', attn_normalized, v)

            # GRU更新
            slots_flat = slots.reshape(-1, C)
            updates_flat = updates.reshape(-1, C)
            slots_updated = self.gru(updates_flat, slots_flat)
            slots = slots_updated.reshape(B * T, K, C)

            # MLP精化
            mlp_out = self.mlp(self.norm_slots(slots))
            slots = slots + mlp_out

            # 在后半段迭代应用时序注意力
            if iter_idx >= self.iters // 2:
              slots = self.temporal_attention(slots, temporal_window=T)

            # Orthogonal loss between slots (coarse) and points (fine)

        # 计算最终分配
        pts_n = self.norm_points(pts)
        sl_n = self.norm_slots(slots)
        q = self.to_q(sl_n)
        k = self.to_k(pts_n)
        scores = torch.einsum('bkc, bnc -> bkn', q, k)
        # 每个输入特征在 slot 维度上的概率分布
        assign = F.softmax(scores / math.sqrt(C), dim=1)  # dim=1 对应 K
        assign = assign.permute(0, 2, 1)  # (B*T, N, K)
        assign = assign.view(B, T, N, K)  # (B, T, N, K)
        slots = slots.view(B, T, K, C)



        # 计算残差
        assign_detached = assign
        clusters_detached = slots
        assigned_clusters = torch.einsum('btnk,btkc->btnc', assign_detached, clusters_detached)
        residuals = points - assigned_clusters





        if self.use_orth_loss:
            orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices = self.orth_loss_module(slots, points, assign)


            return residuals, slots, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices
        else:
            return residuals, slots, assign


# ------------------------------
# 2. Cross-Granularity Interaction (Shared Orthogonal Bases)
# ------------------------------
# class ResidualGuidedInteraction(nn.Module):
#     """
#     Residual-Guided Interaction block.
#
#     Purpose:
#       - Let residuals (point - assigned_cluster) attend to cluster (coarse) features,
#         so residuals are re-interpreted in coarse's semantic space.
#       - Optionally let clusters attend residuals (bi-directional).
#       - Optional gating / residual scaling and optional orthogonality monitoring loss.
#
#     Inputs:
#       clusters: (B, T, K, C)
#       points:   (B, T, N, C)
#       assign:   (B, T, N, K)  # soft assignments from points->clusters (floating)
#       mask:     optional mask for attention (not used by default)
#
#     Args:
#       dim: channel dim C
#       num_heads: multihead attention heads
#       residual_scale: float initial multiplier for adding refined residual back to points
#       gate: bool, whether to use learned gate to fuse original residual and attended residual
#       bidirectional: bool, clusters can optionally attend residuals (updates clusters)
#       use_orth_loss: bool, return an orthogonality loss between residuals and clusters (after refinement)
#       dropout: attention dropout
#     Returns:
#       refined_points: (B, T, N, C) -- points updated by residual-guided info (points + refined_residual)
#       updated_clusters: (B, T, K, C) or None (if bidirectional False)
#       stats: dict (possible keys: 'orth_loss' scalar)
#     """
#     def __init__(self, dim, num_heads=4, residual_scale=0.1, gate=True,
#                  bidirectional=False, use_orth_loss=False, dropout=0.1):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))  # learnable scale
#         self.gate = gate
#         self.bidirectional = bidirectional
#         self.use_orth_loss = use_orth_loss
#
#         # projections for residual->cluster attention (residuals are queries)
#         self.q_proj_r = nn.Linear(dim, dim)
#         self.k_proj_c = nn.Linear(dim, dim)
#         self.v_proj_c = nn.Linear(dim, dim)
#         self.attn_r2c = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
#
#         # optional cluster -> residual attention for updating clusters
#         if bidirectional:
#             self.q_proj_c = nn.Linear(dim, dim)
#             self.k_proj_r = nn.Linear(dim, dim)
#             self.v_proj_r = nn.Linear(dim, dim)
#             self.attn_c2r = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
#
#         self.norm_r=nn.LayerNorm(dim)
#         self.norm_c = nn.LayerNorm(dim)
#
#         # small MLP for post-attn refinement
#         self.refine_mlp = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim, dim)
#         )
#
#         # layernorms
#         self.ln_res = nn.LayerNorm(dim)
#         self.ln_clu = nn.LayerNorm(dim)
#
#     def _to_attn_format(self, x):
#         # x: [B*T, L, C] -> [L, B*T, C] for legacy MultiheadAttention
#         return x.permute(1, 0, 2)
#
#     def _from_attn_format(self, x, B, T):
#         # x: [L, B*T, C] -> [B, T, L, C] then reshape depending caller
#         out = x.permute(1, 0, 2)  # [B*T, L, C]
#         return out.view(B, T, out.shape[1], out.shape[2])
#
#     def forward(self, clusters, points, assign=None, return_stats=False):
#         """
#         clusters: (B, T, K, C)
#         points:   (B, T, N, C)
#         assign:   (B, T, N, K)  -- if provided, used to compute assigned_clusters & residuals
#         """
#         B, T, K, C = clusters.shape
#         _, _, N, _ = points.shape
#         stats = {}
#         #assign_d=assign.detach()
#         # 1) compute assigned_clusters and residuals
#         if assign is None:
#             # fallback: nearest cluster by cosine (not ideal, but safe)
#             # compute soft assignment by similarity (B,T,N,K)
#             pts_n = F.normalize(points, dim=-1)
#             clu_n = F.normalize(clusters, dim=-1)
#             sim = torch.einsum('btnc,btkc->bt nk', pts_n, clu_n) if False else None
#             # we won't implement the fallback heavy path here; require assign in practice
#             assigned = torch.einsum('btnk,btkc->btnc', assign, clusters)
#         else:
#             # assign: (B,T,N,K), clusters: (B,T,K,C)
#             assigned = torch.einsum('btnk,btkc->btnc', assign, clusters)
#
#         residuals = points - assigned  # (B,T,N,C)
#
#         # normalize inputs for attention projections
#         # prepare shapes for attention: flatten B*T as batch dimension
#         residuals_flat = self.norm_r(residuals.reshape(B * T, N, C))
#         clusters_flat = self.norm_c(clusters.reshape(B * T, K, C))
#
#
#
#         # --- residual -> cluster attention (residuales query clusters) ---
#         q_r = self.q_proj_r(self.ln_res(residuals_flat))  # [B*T, N, C]
#         k_c = self.k_proj_c(self.ln_clu(clusters_flat))   # [B*T, K, C]
#         v_c = self.v_proj_c(self.ln_clu(clusters_flat))   # [B*T, K, C]
#
#         q_r_a = self._to_attn_format(q_r)  # [N, B*T, C]
#         k_c_a = self._to_attn_format(k_c)  # [K, B*T, C]
#         v_c_a = self._to_attn_format(v_c)  # [K, B*T, C]
#
#         attn_out_r, _ = self.attn_r2c(q_r_a, k_c_a, v_c_a)  # [N, B*T, C]
#         attn_out_r = attn_out_r.permute(1, 0, 2).reshape(B, T, N, C)  # [B,T,N,C]
#
#         # post-process attended residuals
#         attn_out_r = self.refine_mlp(attn_out_r)
#         # gating between original residual and attended residual
#
#         # scale and add back to points
#         #refined_points = points + self.residual_scale * refined_residual
#         refined_points = points + attn_out_r
#
#         # optional: clusters attend residuals to get cluster updates (bidirectional)
#         updated_clusters = clusters
#         if self.bidirectional:
#             # compute q from clusters, k/v from residuals
#             q_c = self.q_proj_c(self.ln_clu(clusters_flat))  # [B*T, K, C]
#             k_r = self.k_proj_r(self.ln_res(residuals_flat))  # [B*T, N, C]
#             v_r = self.v_proj_r(self.ln_res(residuals_flat))  # [B*T, N, C]
#
#             q_c_a = self._to_attn_format(q_c)  # [K, B*T, C]
#             k_r_a = self._to_attn_format(k_r)  # [N, B*T, C]
#             v_r_a = self._to_attn_format(v_r)  # [N, B*T, C]
#
#             attn_out_c, _ = self.attn_c2r(q_c_a, k_r_a, v_r_a)  # [K, B*T, C]
#             attn_out_c = attn_out_c.permute(1, 0, 2).reshape(B, T, K, C)
#             # small residual update to clusters (keep gentle)
#             updated_clusters = clusters + 1 * self.refine_mlp(attn_out_c)
#
#         # optional orthogonality stat / loss
#         if self.use_orth_loss:
#             # measure cos-sim between (normalized) refined residuals and clusters
#             r_n = F.normalize(refined_points, dim=-1)  # (B,T,N,C)
#             c_n = F.normalize(updated_clusters, dim=-1)          # (B,T,K,C)
#             sim = torch.einsum('btnc,btkc->btnk', r_n, c_n)  # (B,T,N,K)
#             # mean squared off-diagonal (we want sims ~ 0)
#             orth_loss = sim.pow(2).mean()
#
#
#         if return_stats:
#             return refined_points, updated_clusters, orth_loss
#         else:
#             return refined_points, updated_clusters


########################################################
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))  # [max_len, 1, d_model]

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


##################Multimodal-Fusion#########################
class MultiModalAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        # Point 分支
        self.q_proj_point = nn.Linear(dim, dim)
        self.k_proj_point = nn.Linear(dim * 2, dim)
        self.v_proj_point = nn.Linear(dim * 2, dim)
        self.attn_point = nn.MultiheadAttention(dim, num_heads, dropout=dropout)  # 移除batch_first
        self.out_proj_point = nn.Linear(dim, dim)
        self.norm_point = nn.LayerNorm(dim)  # 新增

        # Cluster 分支
        self.q_proj_cluster = nn.Linear(dim, dim)
        self.k_proj_cluster = nn.Linear(dim * 2, dim)
        self.v_proj_cluster = nn.Linear(dim * 2, dim)
        self.attn_cluster = nn.MultiheadAttention(dim, num_heads, dropout=dropout)  # 移除batch_first
        self.out_proj_cluster = nn.Linear(dim, dim)
        self.norm_cluster = nn.LayerNorm(dim)

        self.out_proj_final = nn.Linear(dim, dim)

        self.point_pos_encoding_q = SinusoidalPositionalEncoding(dim)
        self.cluster_pos_encoding_q = SinusoidalPositionalEncoding(dim)

        self.point_pos_encoding_k = SinusoidalPositionalEncoding(dim)
        self.cluster_pos_encoding_k = SinusoidalPositionalEncoding(dim)

        self.point_pos_encoding_v = SinusoidalPositionalEncoding(dim)
        self.cluster_pos_encoding_v = SinusoidalPositionalEncoding(dim)

    def forward(self, point_feats, cluster_feats):
        """
        point_feats: [B, T, N, C]
        cluster_feats: [B, T, N, C]
        text_feats_p: [B, T, N, C]
        text_feats_c: [B, T, N, C]
        return: [B, T, N, C]
        """
        B, T, N, C = point_feats.shape

        # 重塑为 [B*T, N, C] 来处理时间维度
        point_feats_flat = point_feats.reshape(B * T, N, C)
        cluster_feats_flat = cluster_feats.reshape(B * T, N, C)

        # 拼接四个模态的key/value源 [B*T, N, 4C]
        fused_input = torch.cat([point_feats_flat, cluster_feats_flat], dim=-1)

        # 转换为 [N, B*T, C] 格式供注意力使用
        def to_attn_format(x):
            return x.permute(1, 0, 2)  # [N, B*T, C]

        def from_attn_format(x):
            return x.permute(1, 0, 2)  # [B*T, N, C]

        # Point 注意力分支
        q_point = to_attn_format(self.q_proj_point(point_feats_flat))
        k_point = to_attn_format(self.k_proj_point(fused_input))
        v_point = to_attn_format(self.v_proj_point(fused_input))

        q_point = self.point_pos_encoding_q(q_point)
        k_point = self.point_pos_encoding_k(k_point)
        v_point = self.point_pos_encoding_v(v_point)

        attn_out_point, _ = self.attn_point(q_point, k_point, v_point)
        attn_out_point = from_attn_format(self.out_proj_point(attn_out_point))
        attn_out_point = self.norm_point(point_feats_flat + attn_out_point)

        # Cluster 注意力分支
        q_cluster = to_attn_format(self.q_proj_cluster(cluster_feats_flat))
        k_cluster = to_attn_format(self.k_proj_cluster(fused_input))
        v_cluster = to_attn_format(self.v_proj_cluster(fused_input))

        q_cluster = self.cluster_pos_encoding_q(q_cluster)
        k_cluster = self.cluster_pos_encoding_k(k_cluster)
        v_cluster = self.cluster_pos_encoding_v(v_cluster)

        attn_out_cluster, _ = self.attn_cluster(q_cluster, k_cluster, v_cluster)
        attn_out_cluster = from_attn_format(self.out_proj_cluster(attn_out_cluster))
        attn_out_cluster = self.norm_cluster(cluster_feats_flat + attn_out_cluster)

        # 不同的注意力输出融合策略

        attn_out = attn_out_point + attn_out_cluster

        # 转换回原始格式 [B, N, C]
        fused = self.out_proj_final(attn_out.reshape(B, T, N, C))  # [B, N, C]

        return fused


# ---------------- 主类：TextGuidedCluster（增强版） ----------------
class TextGuidedCluster(nn.Module):
    def __init__(self, feat_dim, num_clusters=8, mode='slot', num_iters=3,
                 use_multigranularity=True, dropout=0.0, use_contrastive=True,
                 top_m=16, neg_samples=32, intra_cluster=False, dynamic_top_m_flag=False, min_m=16,
                 max_m=64, return_losses=False):
        super().__init__()
        assert mode in ('soft', 'soft_ori', 'slot')
        self.feat_dim = feat_dim
        self.num_clusters = num_clusters
        self.mode = mode
        self.num_iters = num_iters
        self.use_multigranularity = use_multigranularity

        self.top_m = top_m
        self.neg_samples = neg_samples
        self.use_contrastive = use_contrastive
        self.intra_cluster = intra_cluster
        self.dynamic_top_m_flag = dynamic_top_m_flag
        self.min_m = min_m
        self.max_m = max_m
        self.return_losses = return_losses
        # SoftCluster path
        if mode == 'soft':
            self.proj_logits = nn.Linear(feat_dim, num_clusters, bias=False)
            self.text_to_logits = nn.Linear(feat_dim, num_clusters)

        # Slot path: use Bidirectional SlotAttention by default
        if mode == 'slot':
            self.slots_mu = self._initialize_slots_orthogonal(feat_dim, num_clusters)
            #self.slots_mu = nn.Parameter(torch.randn(1, 1, num_clusters, feat_dim))
            self.slots_sigma = nn.Parameter(torch.randn(1, 1, num_clusters, feat_dim))
            self.slot_from_text = nn.Linear(feat_dim, num_clusters * feat_dim)
            self.slot_attention = SlotAttentionBidirectional(feat_dim, num_clusters, iters=num_iters,
                                                             use_orth_loss=True)




        # MHA blocks
        self.point_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4,dropout=0.1)
        self.cluster_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4,dropout=0.1)

        # gating + dynamic residual gate
        #self.gate_proj = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())
        hidden = max(1, feat_dim // 4)
        self.residual_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )



        # projectors
        self.point_proj = _MLP(feat_dim, feat_dim, feat_dim)
        self.cluster_proj = _MLP(feat_dim, feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

        # for _soft_cluster_ori
        self.proj = nn.Linear(feat_dim, num_clusters)
        self.norm = nn.Softmax(dim=-1)

        # -------- Contrastive Queue --------
        # if use_contrastive:
        #     self.queue = ContrastiveQueue(feat_dim)

        # # ----------分层稀疏-------------
        # self.cluster_refiner = ClusterRefineLayer(C=feat_dim, top_m_local=64, residual_scale_global=0.1,
        #                                           residual_scale_local=0.05)
        ############################
        #self.interaction = ResidualGuidedInteraction(dim=feat_dim, num_heads=4, residual_scale=0.1, gate=True, bidirectional=True, use_orth_loss=True)

        #########----------Multimodal-Fusion
        self.fusion_module = MultiModalAttentionFusion(dim=feat_dim, num_heads=4, dropout=0.1)
        self.norm_f = nn.LayerNorm(feat_dim)

        self.norm_t_p = nn.LayerNorm(feat_dim)
        self.norm_t_c = nn.LayerNorm(feat_dim)
        self.norm_p_t = nn.LayerNorm(feat_dim)
        self.norm_c_t = nn.LayerNorm(feat_dim)

        self.norm_c_t_f= nn.LayerNorm(feat_dim)
        self.norm_p_t_f = nn.LayerNorm(feat_dim)

        self.out_proj_t_c = nn.Linear(feat_dim, feat_dim)
        self.out_proj_t_p = nn.Linear(feat_dim, feat_dim)
        self.out_proj_c_t = nn.Linear(feat_dim, feat_dim)
        self.out_proj_p_t = nn.Linear(feat_dim, feat_dim)

        self.out_proj_c_i = nn.Linear(feat_dim, feat_dim)
        self.out_proj_p_i = nn.Linear(feat_dim, feat_dim)

        self.norm_p_i = nn.LayerNorm(feat_dim)
        self.norm_c_i = nn.LayerNorm(feat_dim)



        self.point_pos_encoding_q = SinusoidalPositionalEncoding(feat_dim)
        self.cluster_pos_encoding_q = SinusoidalPositionalEncoding(feat_dim)
        self.text_pos_encoding_c_q = SinusoidalPositionalEncoding(feat_dim)
        self.text_pos_encoding_p_q = SinusoidalPositionalEncoding(feat_dim)

        self.point_pos_encoding_kv = SinusoidalPositionalEncoding(feat_dim)
        self.cluster_pos_encoding_kv = SinusoidalPositionalEncoding(feat_dim)
        self.text_pos_encoding_c_kv = SinusoidalPositionalEncoding(feat_dim)
        self.text_pos_encoding_p_kv = SinusoidalPositionalEncoding(feat_dim)

        self.fusion_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.norm_f = nn.LayerNorm(feat_dim)
        self.gate_proj = nn.Linear(feat_dim * 2, feat_dim)
        with torch.no_grad():
            self.gate_proj.bias.data.fill_(-2)
            # 同时初始化权重为非常小的值，防止开始时波动太大
            nn.init.uniform_(self.gate_proj.weight, -0.01, 0.01)

    def _initialize_slots_orthogonal(self, feat_dim, num_clusters):
        """纯初始化，不影响训练计算图"""
        slots_mu = torch.randn(1, 1, num_clusters, feat_dim)
        nn.init.orthogonal_(slots_mu)  # 在普通Tensor上操作
        slots_mu = slots_mu * 0.1  # 缩放也在普通Tensor上
        return nn.Parameter(slots_mu)  # 最后包装成Parameter
    # ---------- Soft cluster (unchanged) ----------
    def _soft_cluster(self, x, text_global=None, tau=1.0):
        logits = self.proj_logits(x)
        if text_global is not None:
            t2 = self.text_to_logits(text_global).unsqueeze(2).unsqueeze(2)
            logits = logits + t2
        probs = F.gumbel_softmax(logits, tau=tau, dim=-1)
        weighted = probs.unsqueeze(-1) * x.unsqueeze(-2)
        num = weighted.sum(dim=2)
        denom = probs.sum(dim=2).unsqueeze(-1).clamp(min=1e-6)
        clusters = num / denom
        return None,clusters, probs, None, None, None, None

    def _soft_cluster_ori(self, x, text_global=None):
        logits = self.proj(x)
        if text_global is not None:
            text_bias = self.text_to_logits(text_global).unsqueeze(2)
            logits = logits + text_bias
        probs = self.norm(logits)
        weighted = probs.unsqueeze(-1) * x.unsqueeze(-2)
        num = weighted.sum(dim=2)
        den = probs.sum(dim=2).unsqueeze(-1).clamp(min=1e-6)
        clusters = num / den
        return None,clusters, probs, None, None, None, None

    # ---------- Slot cluster using Bidirectional SlotAttention ----------
    def _slot_cluster(self, x, text_feats,points_3d,tau):
        B, T, N, C = x.shape
        mu = self.slots_mu.expand(B, T, -1, -1)
        sigma = F.softplus(self.slots_sigma).expand(B, T, -1, -1)
        slots_init = mu + sigma * torch.randn_like(mu)
        # if text_feats is not None:
        #     text_global = text_feats.mean(dim=2)
        #     proj = self.slot_from_text(text_global).view(B, T, self.num_clusters, C)
        #     slots_init = slots_init + 0.1*proj
        residuals,slots, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices = self.slot_attention(x, slots_init, text_feats,points_3d,tau)
        return residuals, slots, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices

    # ---------- cross-attention helpers (old MultiheadAttention shape) ----------
    def _cross_attend_point(self, points, text):
        B, T, N, C = points.shape
        _, _, L, _ = text.shape
        points_norm = self.norm_p_t(points)
        # 转换为 [N, B*T, C] 格式
        q = points_norm.reshape(B * T, N, C).permute(1, 0, 2)  # [N, B*T, C]
        kv = text.reshape(B * T, L, C).permute(1, 0, 2)  # [L, B*T, C]

        q = self.point_pos_encoding_q(q)
        kv = self.point_pos_encoding_kv(kv)

        out, _ = self.point_attn(q, kv, kv)  # [N, B*T, C]

        # 转换回原始形状 [B, T, N, C]
        out = out.permute(1, 0, 2).reshape(B, T, N, C)  # [B*T, N, C] -> [B, T, N, C]
        out = self.out_proj_p_t(out)

        # 残差连接：确保形状匹配
        out = self.norm_p_t_f(out + points)  # [B, T, N, C] + [B, T, N, C]

        return out

    def _cross_attend_cluster(self, clusters, text):
        B, T, K, C = clusters.shape
        _, _, L, _ = text.shape
        clusters_norm = self.norm_c_t(clusters)
        # 转换为 [K, B*T, C] 格式
        q = clusters_norm.reshape(B * T, K, C).permute(1, 0, 2)  # [K, B*T, C]
        kv = text.reshape(B * T, L, C).permute(1, 0, 2)  # [L, B*T, C]

        q = self.cluster_pos_encoding_q(q)
        kv = self.cluster_pos_encoding_kv(kv)

        out, _ = self.cluster_attn(q, kv, kv)  # [K, B*T, C]

        # 转换回原始形状 [B, T, K, C]
        out = out.permute(1, 0, 2).reshape(B, T, K, C)  # [B*T, K, C] -> [B, T, K, C]
        out = self.out_proj_c_t(out)

        # 残差连接：确保形状匹配
        out = self.norm_c_t_f(out + clusters)  # [B, T, K, C] + [B, T, K, C]

        return out

    # ---------- forward ----------
    def forward(self, point_feats, text_feats,points_3d, tau=1.0):
        """
        point_feats: (B,T,N,C)
        text_feats: (B,T,L,C)
        returns:
          fused, clusters, assign, gate, (optional losses)
        """
        B, T, N, C = point_feats.shape
        text_global = text_feats.mean(dim=2)

        point_feats_D=point_feats.detach()
        # clustering
        if self.mode == 'soft':
            residuals,clusters, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices = self._soft_cluster(point_feats,
                                                                                                    text_global, tau)
        elif self.mode == 'soft_ori':
            residuals,clusters, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices = self._soft_cluster_ori(point_feats,
                                                                                                        text_global)
        else:
            residuals,clusters, assign, orth_loss, cluster_orth_loss, residual_orth_loss,effective_indices = self._slot_cluster(point_feats_D,
                                                                                                    text_feats,points_3d,tau)

        slots_n = torch.nn.functional.normalize(clusters, dim=-1)
        points_n = torch.nn.functional.normalize(residuals, dim=-1)

        # 计算 cos 相似度
        cos_sim = torch.einsum('btkc, btnc -> btkn', slots_n, points_n)  # (B,T,K,N)
        orth_score = cos_sim.abs().mean()  # 越小越正交

        # cluster-level attention


        # sparse intra-cluster refine
        # if self.intra_cluster:
        #     clusters = self.cluster_refiner(clusters, point_feats_D, assign, mode='hierarchical')

        if self.use_multigranularity:
            point_attn_out = self._cross_attend_point(residuals, text_feats)
            point_attn_out = self.point_proj(point_attn_out)

            cluster_attn_out = self._cross_attend_cluster(clusters, text_feats)
            cluster_attn_out = self.cluster_proj(cluster_attn_out)

            # ensure assign_pt shape (B,T,N,K)
            if assign.shape[2] == clusters.shape[2]:
                assign_pt = assign.permute(0, 1, 3, 2).contiguous()
            else:
                assign_pt = assign

            point_text = point_attn_out
            cluster_text = cluster_attn_out

            #refined_points, refined_cluster , oth_loss_r= self.interaction(cluster_text, point_text, assign=assign_pt, return_stats=True)

            # refined_points=self.norm_p_i(refined_points)
            # refined_cluster = self.norm_c_i(refined_cluster)
            oth_loss_r=0
            refined_cluster_to_point = torch.einsum('btkc,btnk->btnc', cluster_text , assign_pt)
            # 5. 基于净变化的多粒度融合
            # 使用变化量而不是绝对特征进行融合
            fused = self.fusion_module(point_text, refined_cluster_to_point)
            #fused = self.fusion_module(point_text, refined_cluster_to_point)

            original_point = point_feats

            # 特征拼接
            concatenated = torch.cat([fused, point_feats], dim=-1)  # (B,T,N,2C)

            # 融合特征
            #fused = self.fusion_proj(concatenated)  # (B,T,N,C)

            # 生成门控值
            gate = torch.sigmoid(self.gate_proj(concatenated))  # (B,T,N,C) ∈ [0,1]

            # 应用门控混合：在融合特征和原始点特征之间平滑过渡
            fused = fused * gate + (1 - gate) * original_point  # (B,T,N,C)

            # 归一化
            #fused = self.norm_f(fused)
            #fused = self.dropout(fused)

            weight = self.fusion_proj.weight  # (C, 2C)
            fused_weights = weight[:, :C].detach().abs().mean()  # fused部分权重
            point_weights = weight[:, C:].detach().abs().mean()  # point部分权重





            if self.return_losses:
                try:
                    if self.use_contrastive:
                        losses = 0
                    else:
                        losses = None
                except Exception as e:
                    print(f'对比损失计算失败: {e}')
                    import traceback
                    traceback.print_exc()
                    losses = None
                return fused, clusters, assign, gate, losses, 1*orth_loss, cluster_orth_loss, residual_orth_loss,oth_loss_r,fused_weights,point_weights,effective_indices

            return fused, clusters, assign, gate, None, 1*orth_loss, cluster_orth_loss, residual_orth_loss,oth_loss_r,fused_weights,point_weights,effective_indices
        else:
            if assign.shape[2] == clusters.shape[2]:
                assign_pt = assign.permute(0, 1, 3, 2).contiguous()
            else:
                assign_pt = assign
            fused = torch.einsum('btkc,btnk->btnc', cluster_attn_out, assign_pt)
            res_gate = self.residual_gate(point_feats)
            fused = fused + res_gate * point_feats
            fused = self.dropout(fused)
            if return_losses:
                try:
                    if self.use_contrastive:
                        losses = 0
                    else:
                        losses = None
                except Exception as e:
                    print(f'对比损失计算失败: {e}')
                    import traceback
                    traceback.print_exc()
                    losses = None
                return fused, clusters, assign, res_gate, losses, orth_loss, cluster_orth_loss, residual_orth_loss,None,None,None,None
            return fused, clusters, assign, res_gate, None, orth_loss, cluster_orth_loss, residual_orth_loss,None,None,None,None


# ---------- quick self-test ----------


class NewTrackSem(BaseModel):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = Transformer(cfg.transformer_cfg)
        self.loc_net = RPN(cfg.rpn_cfg)
        # self.vit_extractor = ViTFeatureExtractor.from_pretrained('/data/code/iccv2025/vit-base-patch16-224')

        # self.bert = BertModel.from_pretrained('/data/code/iccv2025/bert-base-uncased')
        self.bert = RobertaModel.from_pretrained('/data/jyf/MBPTrack3D_ICCV2025-old/checkpoints/roberta-base')
        # frozen the bert model
        for param in self.bert.parameters():
            param.requires_grad = False

        self.cfg = cfg
        self.n_p = cfg.frame_npts // cfg.backbone_cfg.downsample_ratios[-1]  # backbone最后一层的点数
        self.c = cfg.backbone_cfg.out_channels

        # mlp，降维用
        self.text_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.c)
        )

        # mlp，降维用
        # self.img_mlp = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.c)
        # )

        # text and pc fusion weights
        self.text_linear = nn.Linear(self.c, self.c)
        self.lidar_text_linear = nn.Linear(self.c, self.c)

        self.text_fusion_linear = nn.Linear(self.c, self.c)
        self.text_fusion_layer_norm1 = nn.LayerNorm(self.c)
        self.text_fusion_layer_norm2 = nn.LayerNorm(self.c)
        self.text_layer_norm = nn.LayerNorm(self.c)
        self.lidar_text_layer_norm = nn.LayerNorm(self.c)



        # image and pc fusion weights
        # self.image_linear = nn.Linear(self.c, self.c)
        # self.lidar_image_linear = nn.Linear(self.c, self.c)
        #
        # self.image_fusion_linear = nn.Linear(self.c, self.c)
        # self.image_fusion_layer_norm1 = nn.LayerNorm(self.c)
        # self.image_fusion_layer_norm2 = nn.LayerNorm(self.c)
        # self.image_layer_norm = nn.LayerNorm(self.c)
        # self.lidar_image_layer_norm = nn.LayerNorm(self.c)
        #############软聚类#############
        self.text_guided_cluster = TextGuidedCluster(
            feat_dim=self.c,  # 点特征维度
            num_clusters=12,  # K，可调
            mode='slot',  # 'slot' 或 'soft' 或 "slot_ori"
            num_iters=5,  # slot 模式迭代次数
            use_multigranularity=True,  # 是否使用多粒度融合
            dropout=0.1,
            top_m=16,
            neg_samples=32,
            intra_cluster=False,
            dynamic_top_m_flag=True,
            min_m=16,
            max_m=64,
            return_losses=False
        )

    def forward_lidar_embed(self, input, tau):
        pcds = input['pcds']
        batch_size, duration, npts, _ = pcds.shape

        pcds = pcds.view(batch_size * duration, npts, -1)
        #print("backbone",pcds.shape)
        b_output = self.backbone_net(pcds)
        xyz = b_output['xyz']
        feat = b_output['feat']
        idx = b_output['idx']
        assert len(idx.shape) == 2
        return dict(
            xyzs=xyz.view(batch_size, duration, xyz.shape[1], xyz.shape[2]),
            feats=feat.view(batch_size, duration, feat.shape[1], feat.shape[2]),
            idxs=idx.view(batch_size, duration, idx.shape[1])
        )



    # def forward_text_embed(self, input, tau):
    #     text = input['text']
    #     input_ids = text['input_ids']  # b, t, max_len
    #     attention_mask = text['attention_mask']  # b, t, max_len
    #     print(input_ids.shape)
    #     batch_size, duration, max_len = input_ids.shape
    #
    #     # text features
    #     # x = self.bert(
    #     #     input_ids=input_ids.view(-1, max_len),
    #     #     attention_mask=attention_mask.view(-1, max_len)
    #     # ).last_hidden_state  # [b*t, max_len, 768]
    #     x = self.bert(
    #         input_ids=input_ids.view(-1, max_len),
    #         attention_mask=attention_mask.view(-1, max_len)
    #     ).pooler_output.unsqueeze(1)  # [b*t, 1, 768]
    #
    #     # print(f"After get_text_features: {x.shape}")
    #
    #     # 降维
    #     x = self.text_mlp(x).view(batch_size, duration, -1, self.c)
    #     print(f"After mlp: {x.shape}")
    #
    #
    #     return x  # b,t,n_t,c

    def forward_text_embed(self, input, tau):
        text = input['text']
        input_ids = text['input_ids']  # (b, t, max_len)
        attention_mask = text['attention_mask']  # (b, t, max_len)

        batch_size, duration, max_len = input_ids.shape

        # 使用last_hidden_state获取所有token的特征
        bert_output = self.bert(
            input_ids=input_ids.view(-1, max_len),
            attention_mask=attention_mask.view(-1, max_len)
        )
        #print("bert",input_ids.view(-1, max_len).shape,attention_mask.view(-1, max_len).shape)
        x = bert_output.last_hidden_state  # [b*t, max_len, 768] 所有token的特征

        # 打印调试信息
        #print(f"BERT输出形状: {x.shape}")  # 应该是 (b*t, max_len, 768)

        # 降维 - 对每个token独立处理
        x = self.text_mlp(x)  # [b*t, max_len, c] 每个token降维到特征维度c

        # 恢复原始形状
        x = x.view(batch_size, duration, max_len, self.c)  # [b, t, max_len, c]

        #print(f"最终文本特征形状: {x.shape}")  # 应该是 (b, t, L, c) 其中L是实际token数

        return x  # (b, t, L, c) - L是文本序列长度

    # def forward_lidar_fusion(self, input, tau):
    #     """
    #
    #     Args:
    #         input: [lidar_embed, another_embed]
    #
    #     Returns:
    #         fusion_feature
    #     """
    #
    #     batch_size, duration, c, n_p = input['lidar_embed'].shape
    #     lidar_embed = input['lidar_embed'].transpose(2, 3)  # b,t,n_p,c
    #     lidar_embed = lidar_embed.view(batch_size * duration, n_p, c)  # b*t,n_p,c
    #
    #     _, _, n_t, _ = input['another_embed'].shape
    #     another_embed = input['another_embed']  # b,t,n_t,c
    #     another_embed = another_embed.view(batch_size * duration, n_t, c)  # b*t,n_t,c
    #
    #     # LN
    #     another_embed = self.text_layer_norm(another_embed)  # b*t,n_t,c
    #     lidar_embed = self.lidar_text_layer_norm(lidar_embed)  # b*t,n_p,c
    #
    #     text_proj = self.text_linear(another_embed)  # b*t,n_t,c
    #     lidar_proj = self.lidar_text_linear(lidar_embed)  # b*t,n_p,c
    #
    #     # cal attn
    #     attn_score = torch.matmul(lidar_proj, text_proj.transpose(-1, -2))  # b*t,n_p,n_t
    #     attn_score = attn_score / (self.c ** 0.5)  # b*t,n_p,n_t
    #
    #     attn_score = F.softmax(attn_score, dim=-1)  # b*t,n_p,n_t
    #
    #     fusion_embed = torch.matmul(attn_score, text_proj)  # b*t,n_p,c
    #
    #     fusion_embed = fusion_embed + lidar_embed  # b*t,n_p,c
    #     # layer norm
    #     fusion_embed_1 = self.text_fusion_layer_norm1(fusion_embed)  # b*t,n_p,c
    #
    #     fusion_embed_1 = self.text_fusion_linear(fusion_embed_1)  # b*t,n_p,c
    #     fusion_embed = fusion_embed + fusion_embed_1
    #     fusion_embed = self.text_fusion_layer_norm2(fusion_embed)  # b*t,n_p,c
    #
    #     fusion_embed = 0.5 * fusion_embed + lidar_embed
    #
    #     return fusion_embed.view(batch_size, duration, n_p, c).transpose(-1, -2)  # b,t,c,n_p

    def forward_lidar_text_cluster(self, input, tau):
        """
        新版 Text-Guided Multi-Granularity Clustering 替代原来的 soft_cluster + cross_attn_text。
        """
        if 'xyzs' in input:
            points_3d = input['xyzs']  # (B,T,N,3) - 3D坐标
        else:
            points_3d = None
        lidar = input['lidar_embed'].transpose(2, 3)  # (B,T,N,C)
        text = input['another_embed']  # (B,T,L,C)
        #print("text_guided_cluster",lidar.shape,text.shape)
        fused_points, clusters, assign, gate, losses, orth_loss, cluster_orth_loss, residual_orth_loss,oth_loss_r ,fused_weights,point_weights,effective_indices= self.text_guided_cluster(
            lidar, text, points_3d,tau=tau)



        return fused_points.transpose(-1, -2)  # (b,t,c,n)



    def forward_update(self, input, tau):
        memory = input.pop('memory', None)
        layer_feats = input['layer_feats']  # nl, b, c, n
        xyz = input['xyz']  # b, n, 3
        mask = input['mask']  # b, n
        new_memory = dict()
        new_memory['feat'] = torch.cat((memory['feat'], layer_feats.unsqueeze(3)),
                                       dim=3) if memory is not None else layer_feats.unsqueeze(3)
        # nl, b, c, t, n
        new_memory['xyz'] = torch.cat((memory['xyz'], xyz.unsqueeze(1)),
                                      dim=1) if memory is not None else xyz.unsqueeze(1)
        # b, t, n, 3
        new_memory['mask'] = torch.cat((memory['mask'], mask.unsqueeze(1)),
                                       dim=1) if memory is not None else mask.unsqueeze(1)
        # b, t, n
        if self.training:
            memory_size = self.cfg.train_memory_size
        else:
            memory_size = self.cfg.eval_memory_size

        if new_memory['feat'].shape[3] > memory_size:
            new_memory['feat'] = new_memory['feat'][:, :, :, 1:, :]
            new_memory['xyz'] = new_memory['xyz'][:, 1:, :, :]
            new_memory['mask'] = new_memory['mask'][:, 1:, :]

        return dict(
            memory=new_memory
        )

    def forward_localize(self, input, tau):

        return self.loc_net(input)

    def forward_propagate(self, input, tau):

        memory = input.pop('memory', None)
        feat = input.pop('feat')
        xyz = input.pop('xyz')
        if memory is None:
            assert 'first_mask_gt' in input
            first_mask_gt = input.pop('first_mask_gt')
            #print("Transformer_mask", first_mask_gt.shape)
            # b,n
            mem = dict(
                mask=first_mask_gt.unsqueeze(1),  # b,1,n
            )
        else:
            mem = memory
        trfm_input = dict(
            memory=mem,
            feat=feat,
            xyz=xyz
        )
        #print("Transformer", feat.shape,xyz.shape)
        trfm_output = self.transformer(trfm_input)
        return trfm_output

    def soft_cluster(self, input, tau):
        """
        SoftCluster 调用接口，兼容 forward dispatcher。
        """
        if 'xyzs' in input:
            points_3d = input['xyzs']  # (B,T,N,3) - 3D坐标
        else:
            points_3d = None
        lidar = input['lidar_embed'].transpose(2, 3)  # (B,T,N,C)
        text = input['another_embed']  # (B,T,L,C)

        _, clusters, assign, gate, losses, orth_loss, cluster_orth_loss, residual_orth_loss,oth_loss_r ,fused_weights,point_weights,effective_indices= self.text_guided_cluster(
            lidar, text,points_3d, tau=tau)

        return clusters, assign, gate, losses, orth_loss, cluster_orth_loss, residual_orth_loss,oth_loss_r,fused_weights,point_weights,effective_indices

    def forward(self, input, tau=None, mode=None):
        """
        Dispatcher for different forward modes.

        Args:
            input: dict containing necessary data
            tau: temperature for soft clustering
            mode: string, choose which forward function to run
        """
        forward_dict = {
            'lidar_embed': self.forward_lidar_embed,
            'text_embed': self.forward_text_embed,
            'propagate': self.forward_propagate,
            'localize': self.forward_localize,
            'update': self.forward_update,
            'lidar_fusion': self.forward_lidar_text_cluster,
            'soft_cluster': self.soft_cluster
        }

        assert mode in forward_dict, f'{mode} has not been supported'

        forward_func = forward_dict[mode]

        # 尝试传 tau，如果不需要则忽略
        try:
            return forward_func(input, tau)
        except TypeError:
            return forward_func(input, tau)

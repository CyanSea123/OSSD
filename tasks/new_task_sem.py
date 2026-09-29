import torch.nn.functional as F
import torch
import numpy as np
import time
import json
import os.path as osp
import torchvision.transforms as transforms
from numpy import dtype
import math
from datasets.utils.pcd_utils import *
from .base_task import BaseTask
import os
import pickle

##########################################################################################
class NewTrackSemTask(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

        self.transform = transforms.Compose([
            transforms.Resize((
                self.cfg.dataset_cfg.image_size,
                self.cfg.dataset_cfg.image_size
            )),  # TODO: resize可能比较粗暴
            transforms.ToTensor()
        ])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer_cfg.lr,
            betas=(0.9, 0.999),  # 改 betas
            weight_decay=self.cfg.optimizer_cfg.weight_decay
        )

        def lr_lambda(epoch):
            warmup_epochs = 5
            max_epochs = self.cfg.train_cfg.max_epochs
            min_lr = 1e-5
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress)) * (
                            1 - min_lr / self.cfg.optimizer_cfg.lr) + min_lr / self.cfg.optimizer_cfg.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def build_mask_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        return F.binary_cross_entropy_with_logits(
            pred,
            gt,
        )

    def build_objectness_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.binary_cross_entropy_with_logits(
            pred,
            gt,
            pos_weight=torch.tensor([2.0], device=self.device),
            reduction='none'
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_bbox_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.smooth_l1_loss(pred, gt, reduction='none')
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_center_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.mse_loss(pred, gt, reduction='none')
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)
        return loss


    #################################################################
    #################################################################
    def log_cluster_usage(self, P, step, threshold=0.01):

        B, T, K, N = P.shape
        usage = P.mean(dim=(0, 1, 3))  # -> (K,)
        usage = usage / usage.sum()
        for k, val in enumerate(usage):
            self.log(f"cluster_usage/slot_{k}", val, on_step=True, prog_bar=True)
        unused = (usage < threshold).sum().item()
        self.log("cluster_usage/unused", unused, on_step=True, prog_bar=True)
    ####################################################################
    #################################################################
    def get_temperature(self):
        initial_tau = 1.0
        final_tau = 0.1
        total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        current_step = self.global_step

        # 余弦退火 - 更平滑的衰减
        progress = current_step / total_steps
        tau = final_tau + 0.5 * (initial_tau - final_tau) * (1 + math.cos(math.pi * progress))

        return tau

    def verify_assign_properties(self,assign):
        B, T, N, K = assign.shape

        # 1. 检查概率和是否为1
        sum_probs = assign.sum(dim=-1)
        prob_sum_mean = sum_probs.mean().item()
        prob_sum_std = sum_probs.std().item()

        print(f"概率和验证: 均值={prob_sum_mean:.8f}, 标准差={prob_sum_std:.8f}")

        # 2. 检查概率范围
        prob_min = assign.min().item()
        prob_max = assign.max().item()
        print(f"概率范围: [{prob_min:.6f}, {prob_max:.6f}]")

        # 3. 检查几个示例点
        print("示例点概率分布:")
        for i in range(min(3, N)):
            probs = assign[0, 0, i].detach().cpu().numpy()
            print(f"  点{i}: {probs} (和={probs.sum():.6f})")

        return prob_sum_mean, prob_sum_std




    def training_step(self, batch, batch_idx):

        """
                Args:
                    batch:
                        dict_keys([
                            'wlh', 'lwh', 'pcds',
                            'mask_gts', 'bbox_gts',
                            'first_mask_gt', 'first_bbox_gt', 'is_dynamic_gts',
                            'lang_description': {'input_ids', 'attention_mask'},
                            'color_image'
                            ])
                    batch_idx:

                Returns:

                """
        #print(">>> training_step is running", batch_idx)
        pcds = batch['pcds']  # b,t,n,3
        mask_gts = batch['mask_gts']  # b,t,n
        bbox_gts = batch['bbox_gts']  # b,t,4
        first_mask_gt = batch['first_mask_gt']  # b,n
        first_bbox_gt = batch['first_bbox_gt']  # b,4
        is_dynamic_gts = batch['is_dynamic_gts']  # b,t
        lwh = batch['lwh']  # b,3
        # multi modalities

        lang_description = batch['lang_description']  # b,t,max_length

        color_image = batch['color_image']  # b,t,3,h,w
        #################################################
        ###############################################
        #############################################
        # initial_tau = 1.0
        # final_tau = 0.1
        # total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        # current_step = self.global_step
        # tau = max(final_tau, initial_tau * (1 - current_step / total_steps))

        tau = self.get_temperature()
        ################################################
        ##################################################
        ###################################################





        embed_output = self.model(dict(
            pcds=pcds
        ),tau=tau, mode='lidar_embed')
        xyzs, geo_feats, idxs = embed_output['xyzs'], embed_output['feats'], embed_output['idxs']
        # xyzs: b,t,n_p,3
        # geo_feats: b,t,c,n_p

        # semantic information
        # print(f'geo_feats shape: {geo_feats.shape}')
        # image embedding, use ViT

        # print(f'image_embed shape: {image_embed.shape}')

        # -------text embedding, use bert--------
        text_embed = self.model(dict(
            text=lang_description
        ),tau=tau, mode='text_embed')  # b,t,n_t,c
        # print(f'text_embed shape: {text_embed.shape}')

        # semantic fusion
        # sem_feats = self.model(dict(
        #     lidar_embed=image_embed,
        #     another_embed=text_embed,
        # ), mode='semantic_fusion')

        # lidar fusion
        ########################################################
        #####################################################



        clusters,assign,gate,loss_db,oth_loss, cluster_orth_loss, residual_orth_loss ,oth_loss_r,fused_weights,point_weights,effective_indices= self.model(dict(lidar_embed=geo_feats, another_embed=text_embed,),tau=tau, mode='soft_cluster')
        self.log("gate_mean", gate.mean(), on_step=True, prog_bar=True)
        #self.log("loss_db", loss_db, on_step=True, prog_bar=True)
        self.log("oth_loss", oth_loss, on_step=True, prog_bar=True)
        self.log("cluster_orth_loss", cluster_orth_loss, on_step=True, prog_bar=True)
        self.log("residual_orth_loss", residual_orth_loss, on_step=True, prog_bar=True)
        self.log("oth_loss_r", oth_loss_r, on_step=True, prog_bar=True)
        self.log("fused_weights", fused_weights, on_step=True, prog_bar=True)
        self.log("point_weights", point_weights, on_step=True, prog_bar=True)
        self.log("effective_indices", effective_indices, on_step=True, prog_bar=True)

        #prob_sum_mean, prob_sum_std = self.verify_assign_properties(assign)



        #################################################
        B, T, N, K = assign.shape

        # 1. 熵检查
        entropy = - (assign * torch.log(assign + 1e-8)).sum(dim=-1)
        avg_entropy = entropy.mean().item()
        max_entropy = -math.log(1.0 / K)

        # 2. 方差检查
        variance = assign.var(dim=-1).mean().item()
        uniform_var = (1 / K) * (1 - 1 / K)

        # 3. 最大概率检查
        max_probs, _ = assign.max(dim=-1)
        avg_max_prob = max_probs.mean().item()
        # 新增：Slot-specific监控
        # 1. Slot利用率（有多少slot被有效使用）
        slot_utilization = (assign > 0.1).any(dim=2).float().mean(dim=-1).mean().item()

        # 2. 竞争强度（有多少点有明确的主slot）
        competitive_ratio = (max_probs > 0.5).float().mean().item()

        # 3. Slot负载均衡（避免某些slot过载）
        slot_load = assign.mean(dim=2)  # (B,T,K) 每个slot的平均负载
        load_imbalance = slot_load.std(dim=-1).mean().item()  # 负载不均衡程度

        self.log("avg_entropy/max_entropy>0.9", avg_entropy/max_entropy, on_step=True, prog_bar=True)
        self.log("avg_max_prob<0.4", avg_max_prob, on_step=True, prog_bar=True)
        self.log("variance/variance_u", variance/uniform_var, on_step=True, prog_bar=True)


        self.log("slot_utilization", slot_utilization, on_step=True, prog_bar=True)
        self.log("competitive_ratio", competitive_ratio, on_step=True, prog_bar=True)
        self.log("load_imbalance", load_imbalance, on_step=True, prog_bar=True)











        ################################################

        #self.log("gate_std", gate.std(), on_step=True, prog_bar=True)
        #self.log("gate_min", gate.min(), on_step=True, prog_bar=True)
        #self.log("gate_max", gate.max(), on_step=True, prog_bar=True)
        # self.log("gate>=0.6_ratio", (gate >= 0.6).float().mean(), on_step=True, prog_bar=True)
        # self.log("gate<=0.4_ratio", (gate <= 0.4).float().mean(), on_step=True, prog_bar=True)
        # self.log("gate>=0.7_ratio", (gate >= 0.7).float().mean(), on_step=True, prog_bar=True)
        # self.log("gate<=0.3_ratio", (gate <= 0.3).float().mean(), on_step=True, prog_bar=True)
        #self.log_cluster_usage(assign.detach(), step=self.global_step, threshold=0.01)
        B, T, N, K = assign.shape





        #######################################################
        #######################################################
        geo_feats = self.model(dict(
            lidar_embed=geo_feats,
            another_embed=text_embed,
        ),tau=tau, mode='lidar_fusion')

        propagate_output = self.model(dict(
            feat=geo_feats[:, 0, :, :],
            xyz=xyzs[:, 0, :, :],
            first_mask_gt=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
        ),tau=tau, mode='propagate')

        layer_feats = propagate_output['layer_feats']

        update_output = self.model(dict(
            layer_feats=layer_feats,
            xyz=xyzs[:, 0, :, :],
            mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
        ),tau=tau, mode='update')
        memory = update_output['memory']

        n_smp_frame = self.cfg.dataset_cfg.num_smp_frames_per_tracklet

        mask_loss, crs_obj_loss, rfn_obj_loss, center_loss, bbox_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(1, n_smp_frame):
            propagate_output = self.model(dict(
                memory=memory,
                feat=geo_feats[:, i, :, :],
                xyz=xyzs[:, i, :, :]
            ),tau=tau, mode='propagate')
            geo_feat, mask_feat = propagate_output['geo_feat'], propagate_output['mask_feat']
            layer_feats = propagate_output['layer_feats']

            localize_output = self.model(dict(
                geo_feat=geo_feat,
                mask_feat=mask_feat,
                xyz=xyzs[:, i, :, :],
                lwh=lwh,
                center_gt=bbox_gts[:, i, :3]
            ),tau=tau, mode='localize')
            mask_pred = localize_output['mask_pred']
            # b,n
            mask_loss += self.build_mask_loss(dict(
                pred=mask_pred,
                gt=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])
            ))
            center_pred = localize_output['center_pred']
            center_loss += self.build_center_loss(dict(
                pred=center_pred,
                gt=bbox_gts[:, i, :3].unsqueeze(
                    1).expand_as(center_pred),
                mask=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])
            ))

            dist = torch.sum(
                (center_pred - bbox_gts[:, i, None, :3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask = torch.ones_like(
                objectness_label, dtype=torch.float)
            objectness_pred = localize_output['objectness_pred']
            crs_obj_loss += self.build_objectness_loss(dict(
                pred=objectness_pred,
                gt=objectness_label,
                mask=objectness_mask
            ))

            bboxes_pred = localize_output['bboxes_pred']
            proposal_xyz = localize_output['proposal_xyz']
            dist = torch.sum(
                (proposal_xyz - bbox_gts[:, i, None, :3]) ** 2, dim=-1)

            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_pred = bboxes_pred[:, :, 4]  # B, K
            objectness_mask = torch.ones_like(
                objectness_label, dtype=torch.float)
            rfn_obj_loss += self.build_objectness_loss(dict(
                pred=objectness_pred,
                gt=objectness_label,
                mask=objectness_mask
            ))
            bbox_loss += self.build_bbox_loss(dict(
                pred=bboxes_pred[:, :, :4],
                gt=bbox_gts[:, i, None, :4].expand_as(
                    bboxes_pred[:, :, :4]),
                mask=objectness_label
            ))

            if i < n_smp_frame - 1:
                # 直接写死参数
                use_teacher_forcing = True  # 是否用 teacher forcing
                teacher_force_epochs_ratio = 0.5  # 比例，表示多少训练阶段内逐步减少 teacher forcing
                total_epochs = self.trainer.max_epochs  # 用 lightning 里设置的最大 epoch
                use_memory_ema = False  # 是否使用 memory EMA
                memory_ema_alpha = 0.9  # EMA 衰减系数

                # 防止除零
                denom = max(1.0, teacher_force_epochs_ratio * max(1, total_epochs))
                current_epoch = int(getattr(self, 'current_epoch', 0))

                if use_teacher_forcing:
                    teacher_force_ratio = max(0.0, 1.0 - current_epoch / denom)
                    if torch.rand(1, device=mask_pred.device).item() < teacher_force_ratio:
                        update_mask = torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :]).float()
                    else:
                        update_mask = mask_pred.sigmoid()
                else:
                    update_mask = mask_pred.sigmoid()

                update_output = self.model(dict(
                    layer_feats=layer_feats,
                    xyz=xyzs[:, i, :, :],
                    mask=update_mask,
                    memory=memory
                ),tau=tau, mode='update')
                new_memory = update_output['memory']

                if use_memory_ema:
                    for k in memory:
                        memory[k] = memory_ema_alpha * memory[k] + (1.0 - memory_ema_alpha) * new_memory[k]
                else:
                    memory = new_memory

        loss = self.cfg.loss_cfg.mask_weight * mask_loss + \
            self.cfg.loss_cfg.crs_obj_weight * crs_obj_loss + \
            self.cfg.loss_cfg.rfn_obj_weight * rfn_obj_loss + \
            self.cfg.loss_cfg.bbox_weight * bbox_loss + \
            self.cfg.loss_cfg.center_weight * center_loss+oth_loss

        # loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)


        ########################################################

        ###################################################################

        self.log("loss", loss, on_step=True, prog_bar=True)

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_bbox': bbox_loss,
                'loss_center': center_loss,
                'loss_mask': mask_loss,
                'loss_rfn_objectness': rfn_obj_loss,
                'loss_crs_objectness': crs_obj_loss,

            },
            global_step=self.global_step
        )
        #print ("loss:", loss)
        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data
    def forward_on_tracklet(self, tracklet):

        pred_bboxes = []
        gt_bboxes = []
        annos=[]
        # ##########测试时间
        frame_times = []
        warmup_frames = 5
# ##########
#         save_debug = True
#         debug_dir = '/data/jyf/MBPTrack3D_ICCV2025_1219/debug_vis'
#         debug_vis = []
#         if save_debug:
#             os.makedirs(debug_dir, exist_ok=True)
# ############

        memory = None
        lwh = None
        tau=0.1
        last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])

        with torch.no_grad():
            for frame_id, frame in enumerate(tracklet):
                # ##########测试时间
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                starter.record()
                # ##########测试时间
                # print(frame.keys())     # dict_keys(['pcd', 'bbox', 'anno', 'nlp', 'image'])
                #print(frame['anno'])
                gt_bboxes.append(frame['bbox'])
                annos.append(frame['anno'])
                if frame_id == 0:
                    base_bbox = frame['bbox']
                    lwh = np.array(
                        [base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
                else:
                    base_bbox = pred_bboxes[-1]

                pcd = crop_and_center_pcd(
                    frame['pcd'], base_bbox, offset=self.cfg.dataset_cfg.frame_offset, offset2=self.cfg.dataset_cfg.frame_offset2, scale=self.cfg.dataset_cfg.frame_scale)
                if frame_id == 0:
                    # print(pcd.nbr_points())
                    if pcd.nbr_points() == 0:
                        pcd.points = np.array([[0.0],[0.0],[0.0]])
                    bbox = transform_box(frame['bbox'], base_bbox)
                    mask_gt = get_pcd_in_box_mask(
                        pcd, bbox, scale=1.25).astype(int)
                    # print(pcd.nbr_points(), mask_gt.shape)
                    bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                        bbox.orientation.degrees if self.cfg.dataset_cfg.degree else bbox.orientation.radians) * bbox.orientation.axis[-1]])

                    # #########################
                    # # 保存原始crop后的点云
                    # raw_points = pcd.points.T.copy()
                    # ####################

                    pcd, idx = resample_pcd(
                        pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
                    mask_gt = mask_gt[idx]
                    # print(mask_gt.shape, pcd.nbr_points())
                else:
                    if pcd.nbr_points() <= 1:
                        bbox = get_offset_box(
                            pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                        pred_bboxes.append(bbox)
                        continue

                    # #########################
                    # # 保存原始crop后的点云
                    # raw_points = pcd.points.T.copy()
                    # ####################


                    pcd, idx = resample_pcd(
                        pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)

                embed_output = self.model(dict(
                    pcds=torch.tensor(pcd.points.T, device=self.device,
                                      dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                ),tau=tau, mode='lidar_embed')




                text_embed = self.model(
                    dict(
                        text={
                            'input_ids':
                                frame['nlp']['input_ids'].to(self.device).unsqueeze(0).unsqueeze(0),
                            'attention_mask':
                                frame['nlp']['attention_mask'].to(self.device).unsqueeze(0).unsqueeze(0)
                        }
                    ),tau=tau,
                    mode='text_embed'
                )

                xyzs, geo_feats, idxs = embed_output['xyzs'], embed_output['feats'], embed_output['idxs']

                # # ==========================================================
                # # OSSD
                # # ==========================================================
                # clusters, assign, gate, *_ = self.model(
                #     dict(
                #         lidar_embed=geo_feats,
                #         another_embed=text_embed,
                #     ),
                #     tau=tau,
                #     mode='soft_cluster'
                # )
                #
                # # ==========================================================
                # # to numpy
                # # ==========================================================
                # sampled_xyz = xyzs[:, 0, :, :].squeeze(0).detach().cpu().numpy()
                #
                # assign_np = assign.squeeze(0).squeeze(0).detach().cpu().numpy()
                # cluster_np = clusters.detach().cpu().numpy()





                # semantic fusion
                # sem_feats = self.model(dict(
                #     lidar_embed=image_embed,
                #     another_embed=text_embed,
                # ), mode='semantic_fusion')

                # lidar fusion
                geo_feats = self.model(dict(
                    lidar_embed=geo_feats,
                    another_embed=text_embed,
                ),tau=tau, mode='lidar_fusion')

                if frame_id == 0:
                    first_mask_gt = torch.tensor(
                        mask_gt, device=self.device, dtype=torch.float32).unsqueeze(0)
                    first_bbox_gt = torch.tensor(
                        bbox_gt, device=self.device, dtype=torch.float32).unsqueeze(0)
                    propagate_output = self.model(dict(
                        feat=geo_feats[:, 0, :, :],
                        xyz=xyzs[:, 0, :, :],
                        first_mask_gt=torch.gather(
                            first_mask_gt, 1, idxs[:, 0, :])
                    ),tau=tau, mode='propagate')
                    layer_feats = propagate_output['layer_feats']
                    update_output = self.model(dict(
                        layer_feats=layer_feats,
                        xyz=xyzs[:, 0, :, :],
                        mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
                    ),tau=tau, mode='update')
                    memory = update_output['memory']

                    pred_bboxes.append(frame['bbox'])
                else:
                    propagate_output = self.model(dict(
                        memory=memory,
                        feat=geo_feats[:, 0, :, :],
                        xyz=xyzs[:, 0, :, :]
                    ),tau=tau, mode='propagate')
                    geo_feat, mask_feat = propagate_output['geo_feat'], propagate_output['mask_feat']
                    layer_feats = propagate_output['layer_feats']

                    localize_output = self.model(dict(
                        geo_feat=geo_feat,
                        mask_feat=mask_feat,
                        xyz=xyzs[:, 0, :, :],
                        lwh=torch.tensor(lwh, device=self.device,
                                         dtype=torch.float32).unsqueeze(0),
                    ),tau=tau, mode='localize')
                    mask_pred = localize_output['mask_pred']
                    bboxes_pred = localize_output['bboxes_pred']
                    bboxes_pred_cpu = bboxes_pred.squeeze(
                        0).detach().cpu().numpy()

                    bboxes_pred_cpu[np.isnan(bboxes_pred_cpu)] = -1e6
                    # remove bboxes whose objectness pred is nan
                    # it may happen at the early stage of training

                    best_box_idx = bboxes_pred_cpu[:, 4].argmax()
                    bbox_cpu = bboxes_pred_cpu[best_box_idx, 0:4]
                    if torch.max(mask_pred.sigmoid()) < self.cfg.missing_threshold:
                        bbox = get_offset_box(
                            pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    else:
                        bbox = get_offset_box(
                            pred_bboxes[-1], bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                        last_bbox_cpu = bbox_cpu

                    pred_bboxes.append(bbox)
                    if frame_id < len(tracklet)-1:
                        update_output = self.model(dict(
                            layer_feats=layer_feats,
                            xyz=xyzs[:, 0, :, :],
                            mask=mask_pred.sigmoid(),
                            memory=memory
                        ),tau=tau, mode='update')
                        memory = update_output['memory']
                    ##################测试时间
                    ender.record()
                    torch.cuda.synchronize()

                    frame_time = starter.elapsed_time(ender)

                    if frame_id >= warmup_frames:
                        frame_times.append(frame_time)

                    #print(f"[Frame {frame_id}] latency: {frame_time:.3f} ms")
                    ##################测试时间
                    # if save_debug:
                    #     debug_vis.append({
                    #         # meta
                    #         "frame_id": frame_id,
                    #         "scene": frame['anno']['scene'],
                    #
                    #         # geometry
                    #         "raw_points": raw_points,
                    #         "sampled_xyz": sampled_xyz,
                    #
                    #         # OSSD
                    #         "assign": assign_np,
                    #         "clusters": cluster_np,
                    #
                    #         # bbox
                    #         "gt_bbox_center": frame['bbox'].center.copy(),
                    #         "gt_bbox_wlh": frame['bbox'].wlh.copy(),
                    #         "gt_bbox_rot": frame['bbox'].rotation_matrix.copy(),
                    #     })

        if len(frame_times) > 0:
            avg_time = np.mean(frame_times)

            fps = 1000.0 / avg_time

            print("\n==============================")
            print(f"Average latency: {avg_time:.3f} ms")
            print(f"FPS: {fps:.2f}")
            print("==============================")
        return pred_bboxes, gt_bboxes,annos
    # def forward_on_tracklet(self, tracklet):
    #     use_ema_memory = True
    #     ema_alpha = 0.9
    #     bbox_jump_threshold = 2.0
    #     bbox_smooth_alpha = 0.7
    #     blend_alpha = 0.5
    #     pred_bboxes = []
    #     gt_bboxes = []
    #     annos = []
    #     memory = None
    #     lwh = None
    #
    #     last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])
    #     last_mask = None  # for mask fallback
    #
    #     with torch.no_grad():
    #         for frame_id, frame in enumerate(tracklet):
    #             gt_bboxes.append(frame['bbox'])
    #             annos.append(frame['anno'])
    #             if frame_id == 0:
    #                 base_bbox = frame['bbox']
    #                 lwh = np.array([base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
    #             else:
    #                 base_bbox = pred_bboxes[-1]
    #
    #             # ----------------- crop point cloud -----------------
    #             pcd = crop_and_center_pcd(
    #                 frame['pcd'], base_bbox,
    #                 offset=self.cfg.dataset_cfg.frame_offset,
    #                 offset2=self.cfg.dataset_cfg.frame_offset2,
    #                 scale=self.cfg.dataset_cfg.frame_scale
    #             )
    #             if frame_id == 0:
    #                 if pcd.nbr_points() == 0:
    #                     pcd.points = np.array([[0.0], [0.0], [0.0]])
    #                 bbox = transform_box(frame['bbox'], base_bbox)
    #                 mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(int)
    #                 bbox_gt = np.array([
    #                     bbox.center[0], bbox.center[1], bbox.center[2],
    #                     (bbox.orientation.degrees if self.cfg.dataset_cfg.degree else bbox.orientation.radians) *
    #                     bbox.orientation.axis[-1]
    #                 ])
    #                 pcd, idx = resample_pcd(pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
    #                 mask_gt = mask_gt[idx]
    #             else:
    #                 if pcd.nbr_points() <= 1:
    #                     # fallback: 如果点云为空, 直接用 offset_box
    #                     bbox = get_offset_box(pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
    #                                           is_training=False)
    #                     pred_bboxes.append(bbox)
    #                     continue
    #                 pcd, idx = resample_pcd(pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
    #
    #             # ----------------- embed -----------------
    #             embed_output = self.model(dict(
    #                 pcds=torch.tensor(pcd.points.T, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #             ), mode='lidar_embed')
    #
    #             image_embed = self.model(dict(
    #                 image=self.transform(frame['image']).to(self.device).unsqueeze(0).unsqueeze(0)
    #             ), mode='image_embed')
    #
    #             text_embed = self.model(dict(
    #                 text={
    #                     'input_ids': frame['nlp']['input_ids'].to(self.device).unsqueeze(0).unsqueeze(0),
    #                     'attention_mask': frame['nlp']['attention_mask'].to(self.device).unsqueeze(0).unsqueeze(0)
    #                 }
    #             ), mode='text_embed')
    #
    #             xyzs, geo_feats, idxs = embed_output['xyzs'], embed_output['feats'], embed_output['idxs']
    #
    #             # lidar fusion
    #             geo_feats = self.model(dict(
    #                 lidar_embed=geo_feats,
    #                 another_embed=text_embed,
    #             ), mode='lidar_fusion')
    #
    #             # ----------------- first frame -----------------
    #             if frame_id == 0:
    #                 first_mask_gt = torch.tensor(mask_gt, device=self.device, dtype=torch.float32).unsqueeze(0)
    #                 first_bbox_gt = torch.tensor(bbox_gt, device=self.device, dtype=torch.float32).unsqueeze(0)
    #                 propagate_output = self.model(dict(
    #                     feat=geo_feats[:, 0, :, :],
    #                     xyz=xyzs[:, 0, :, :],
    #                     first_mask_gt=torch.gather(first_mask_gt, 1, idxs[:, 0, :])
    #                 ), mode='propagate')
    #                 layer_feats = propagate_output['layer_feats']
    #                 update_output = self.model(dict(
    #                     layer_feats=layer_feats,
    #                     xyz=xyzs[:, 0, :, :],
    #                     mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
    #                 ), mode='update')
    #                 memory = update_output['memory']
    #                 pred_bboxes.append(frame['bbox'])
    #                 last_mask = first_mask_gt
    #             else:
    #                 # propagate
    #                 propagate_output = self.model(dict(
    #                     memory=memory,
    #                     feat=geo_feats[:, 0, :, :],
    #                     xyz=xyzs[:, 0, :, :]
    #                 ), mode='propagate')
    #                 geo_feat, mask_feat = propagate_output['geo_feat'], propagate_output['mask_feat']
    #                 layer_feats = propagate_output['layer_feats']
    #
    #                 # localize
    #                 localize_output = self.model(dict(
    #                     geo_feat=geo_feat,
    #                     mask_feat=mask_feat,
    #                     xyz=xyzs[:, 0, :, :],
    #                     lwh=torch.tensor(lwh, device=self.device, dtype=torch.float32).unsqueeze(0),
    #                 ), mode='localize')
    #
    #                 mask_pred = localize_output['mask_pred']
    #                 bboxes_pred = localize_output['bboxes_pred']
    #                 bboxes_pred_cpu = bboxes_pred.squeeze(0).detach().cpu().numpy()
    #                 bboxes_pred_cpu[np.isnan(bboxes_pred_cpu)] = -1e6
    #
    #                 best_box_idx = bboxes_pred_cpu[:, 4].argmax()
    #                 bbox_cpu = bboxes_pred_cpu[best_box_idx, 0:4]
    #
    #                 # ----------- mask fallback -----------
    #                 if torch.max(mask_pred.sigmoid()) < self.cfg.missing_threshold:
    #                     # if last_mask is not None:
    #                     #     mask_pred = last_mask  # fallback 到上一次 mask
    #                     bbox = get_offset_box(pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
    #                                           is_training=False)
    #                 else:
    #                     # ----------- bbox fallback + smoothing -----------
    #                     if np.linalg.norm(bbox_cpu - last_bbox_cpu) > bbox_jump_threshold:
    #                         # 平滑处理, 避免跳动
    #                         bbox_cpu = bbox_smooth_alpha * last_bbox_cpu + (1 - bbox_smooth_alpha) * bbox_cpu
    #                     bbox = get_offset_box(pred_bboxes[-1], bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
    #                                           is_training=False)
    #                     last_bbox_cpu = bbox_cpu
    #                     # last_mask = mask_pred.sigmoid()
    #
    #                 pred_bboxes.append(bbox)
    #
    #                 if frame_id < len(tracklet) - 1:
    #                     # ----------- EMA memory update -----------
    #                     update_output = self.model(dict(
    #                         layer_feats=layer_feats,
    #                         xyz=xyzs[:, 0, :, :],
    #                         mask=mask_pred.sigmoid(),
    #                         memory=memory
    #                     ), mode='update')
    #                     new_memory = update_output['memory']
    #                     if use_ema_memory:
    #                         # decay = getattr(self.cfg, "ema_decay", 0.9)
    #                         for k in memory:
    #                             if torch.max(mask_pred.sigmoid()) > self.cfg.missing_threshold:
    #                                 memory[k] = ema_alpha * memory[k] + (1 - ema_alpha) * new_memory[k]
    #                     else:
    #                         memory = new_memory
    #
    #     return pred_bboxes, gt_bboxes, annos


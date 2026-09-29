import os.path as osp
import pickle as pkl
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import bisect
import torch
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import nuscenes.utils.splits

from .utils import *
from .base_dataset import BaseDataset, EvalDatasetWrapper
from utils import pl_ddp_rank
from transformers import RobertaTokenizer

class NuscenesFULL(BaseDataset):

    cat2n_cat = {
        'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                          'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                          'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                          'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction', 'vehicle.bicycle', 'vehicle.motorcycle'],
        'Bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
        'Car': ['vehicle.car'],
        'Pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                       'human.pedestrian.police_officer'],
        'Trailer': ['vehicle.trailer'],
        'Truck': ['vehicle.truck'],
        'All': ['vehicle.bus.bendy', 'vehicle.bus.rigid','vehicle.car','human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                       'human.pedestrian.police_officer','vehicle.trailer','vehicle.truck']

    }

    n_cat2cat = {
        "animal": "void / ignore",
        "human.pedestrian.personal_mobility": "void / ignore",
        "human.pedestrian.stroller": "void / ignore",
        "human.pedestrian.wheelchair": "void / ignore",
        "movable_object.barrier": "void / ignore",
        "movable_object.debris": "void / ignore",
        "movable_object.pushable_pullable": "void / ignore",
        "movable_object.trafficcone": "void / ignore",
        "static_object.bicycle_rack": "void / ignore",
        "vehicle.emergency.ambulance": "void / ignore",
        "vehicle.emergency.police": "void / ignore",
        "vehicle.construction": "void / ignore",
        "vehicle.bicycle": "void / ignore",
        "vehicle.bus.bendy": "Bus",
        "vehicle.bus.rigid": "Bus",
        "vehicle.car": "Car",
        "vehicle.motorcycle": "void / ignore",
        "human.pedestrian.adult": "Pedestrian",
        "human.pedestrian.child": "Pedestrian",
        "human.pedestrian.construction_worker": "Pedestrian",
        "human.pedestrian.police_officer": "Pedestrian",
        "vehicle.trailer": "Trailer",
        "vehicle.truck": "Truck",
    }

    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        assert cfg.category_name in [
            'Bus', 'Car', 'Pedestrian', 'Truck', 'Trailer',"All"]

        self.data_root_dir = cfg.data_root_dir

        self.preload_offset = cfg.train_cfg.preload_offset if split_type == 'train' else cfg.eval_cfg.preload_offset
        self.cache = cfg.train_cfg.cache if split_type == 'train' else cfg.eval_cfg.cache
        self.key_frame_only = True
        self.min_points = 1 if split_type in ['val', 'test'] else -1
        #################
        #lang_description#
        #################
        bert_tokenizer = RobertaTokenizer.from_pretrained('/data/jyf/MBPTrack3D_ICCV2025-old/checkpoints/roberta-base')
        self.scene_lang_input_ids = {}  # scene_token -> encoded input_ids
        self.scene_lang_attn_mask = {}  # scene_token -> encoded attention_mask

        if self.cache:
            if not cfg.debug:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'NuXcenes_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            else:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir, f'NuXcenes_DEBUG_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            if osp.exists(cache_file_dir):
                self.log.info(f'Loading data from cache file {cache_file_dir}')
                with open(cache_file_dir, 'rb') as f:
                    tracklets = pkl.load(f)
                tracklets = self.filter_tracklets(tracklets, split_type)
                self.tracklet_num_frames = [len(tracklet['frames'])
                                            for tracklet in tracklets]
                self.tracklet_st_frame_id = []
                self.tracklet_ed_frame_id = []
                last_ed_frame_id = 0
                for num_frames in self.tracklet_num_frames:
                    assert num_frames > 0
                    self.tracklet_st_frame_id.append(last_ed_frame_id)
                    last_ed_frame_id += num_frames
                    self.tracklet_ed_frame_id.append(last_ed_frame_id)

            else:
                self.nusc = NuScenes(version='v1.0-trainval' if not cfg.debug else 'v1.0-mini',
                                     dataroot=cfg.data_root_dir, verbose=False)
                track_instances = self._build_track_instances(
                    split_type, cfg.category_name, self.min_points)
                ###################################
                ##################################
                all_scene_tokens = set()
                for inst in track_instances:
                    anno = self.nusc.get('sample_annotation', inst['first_annotation_token'])
                    sample = self.nusc.get('sample', anno['sample_token'])
                    scene_token = sample['scene_token']
                    all_scene_tokens.add(scene_token)

                # Encode every scene description
                for scene_token in all_scene_tokens:
                    scene = self.nusc.get('scene', scene_token)
                    desc = scene['description_new']

                    encoded = bert_tokenizer(
                        desc,
                        padding="max_length",
                        truncation=True,
                        max_length=cfg.max_length,
                        return_tensors="pt"
                    )
                    self.scene_lang_input_ids[scene_token] = encoded['input_ids'][0]
                    self.scene_lang_attn_mask[scene_token] = encoded['attention_mask'][0]

                ##################################
                #################################
                self.tracklet_annotations = self._build_tracklet_annotations(
                    track_instances)
                self.tracklet_annotations = self.filter_tracklet_annos(
                    self.tracklet_annotations, split_type)
                self.tracklet_num_frames = [len(tracklet_anno)
                                            for tracklet_anno in self.tracklet_annotations]
                self.tracklet_st_frame_id = []
                self.tracklet_ed_frame_id = []
                last_ed_frame_id = 0
                for num_frames in self.tracklet_num_frames:
                    assert num_frames > 0
                    self.tracklet_st_frame_id.append(last_ed_frame_id)
                    last_ed_frame_id += num_frames
                    self.tracklet_ed_frame_id.append(last_ed_frame_id)

                tracklets = []
                for tracklet_id in tqdm(range(len(self.tracklet_annotations)), desc='[%6s]Loading pcds ' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
                    frames = []
                    for frame_anno in self.tracklet_annotations[tracklet_id]:
                        frames.append(self._build_frame(frame_anno))
                    # continue
                    comp_template_pcd = merge_template_pcds(
                        [frame['pcd'] for frame in frames],
                        [frame['bbox'] for frame in frames],
                        offset=cfg.target_offset,
                        scale=cfg.target_scale
                    )
                    assert comp_template_pcd is not None
                    if self.preload_offset > 0:
                        for frame in frames:
                            frame['pcd'] = crop_pcd_axis_aligned(
                                frame['pcd'], frame['bbox'], offset=self.preload_offset)

                    tracklets.append({
                        'comp_template_pcd': comp_template_pcd,
                        'frames': frames
                    })
                # assert False
                with open(cache_file_dir, 'wb') as f:
                    self.log.info(
                        f'Saving data to cache file {cache_file_dir}')
                    pkl.dump(tracklets, f)
            self.tracklets = tracklets
        else:
            if split_type != 'train':
                version = 'v1.0-mini' if not cfg.test_version else cfg.test_version
            else:
                version = 'v1.0-trainval' if not cfg.debug else 'v1.0-mini'
            self.nusc = NuScenes(version=version,
                                 dataroot=cfg.data_root_dir, verbose=False)
            track_instances = self._build_track_instances(
                split_type, cfg.category_name, self.min_points)

            ###################################
            ##################################
            all_scene_tokens = set()
            for inst in track_instances:
                anno = self.nusc.get('sample_annotation', inst['first_annotation_token'])
                sample = self.nusc.get('sample', anno['sample_token'])
                scene_token = sample['scene_token']
                all_scene_tokens.add(scene_token)

            # Encode every scene description
            for scene_token in all_scene_tokens:
                scene = self.nusc.get('scene', scene_token)
                desc = scene['description_new']

                encoded = bert_tokenizer(
                    desc,
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.max_length,
                    return_tensors="pt"
                )
                self.scene_lang_input_ids[scene_token] = encoded['input_ids'][0]
                self.scene_lang_attn_mask[scene_token] = encoded['attention_mask'][0]

            ##################################
            #################################


            self.tracklet_annotations = self._build_tracklet_annotations(
                track_instances)
            self.tracklet_annotations = self.filter_tracklet_annos(
                self.tracklet_annotations, split_type)
            self.tracklet_num_frames = [len(tracklet_anno)
                                        for tracklet_anno in self.tracklet_annotations]
            self.tracklet_st_frame_id = []
            self.tracklet_ed_frame_id = []
            last_ed_frame_id = 0
            for num_frames in self.tracklet_num_frames:
                assert num_frames > 0
                self.tracklet_st_frame_id.append(last_ed_frame_id)
                last_ed_frame_id += num_frames
                self.tracklet_ed_frame_id.append(last_ed_frame_id)

            self.tracklets = None

    def filter_tracklets(self, tracklets, split_type):
        if split_type == 'train':
            return [tracklet for tracklet in tracklets if len(tracklet['frames']) >= self.cfg.tracklet_length_lb]
        else:
            return tracklets

    def filter_tracklet_annos(self, tracklet_annos, split_type):
        if split_type == 'train':
            return [tracklet_anno for tracklet_anno in tracklet_annos if len(tracklet_anno) >= self.cfg.tracklet_length_lb]
        else:
            return tracklet_annos

    def get_dataset(self):
        if self.split_type == 'train':
            return MultiInputTrainDatasetWrapper(self, self.cfg, self.log)
        else:
            return EvalDatasetWrapper(self, self.cfg, self.log)

    def num_tracklets(self):
        return len(self.tracklet_annotations)

    def num_frames(self):
        return self.tracklet_ed_frame_id[-1]

    def num_tracklet_frames(self, tracklet_id):
        return self.tracklet_num_frames[tracklet_id]

    def get_frame(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.tracklets[tracklet_id]['frames'][frame_id]
            return frame
        else:
            frame_anno = self.tracklet_annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno)
            if self.preload_offset > 0:
                frame['pcd'] = crop_pcd_axis_aligned(
                    frame['pcd'], frame['bbox'], offset=self.preload_offset)
            return frame

    def get_comp_pcd(self, tracklet_id):
        comp_template_pcd = self.tracklets[tracklet_id]['comp_template_pcd']
        return comp_template_pcd

    def get_tracklet_frame_id(self, idx):
        tracklet_id = bisect.bisect_right(
            self.tracklet_ed_frame_id, idx)
        assert self.tracklet_st_frame_id[
            tracklet_id] <= idx and idx < self.tracklet_ed_frame_id[tracklet_id]
        frame_id = idx - \
            self.tracklet_st_frame_id[tracklet_id]
        return tracklet_id, frame_id

    def _build_track_instances(self, split_type, category_name, min_points):
        general_classes = self.cat2n_cat[category_name]
        instances = []
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation',
                                 instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get(
                'category', instance['category_token'])['name']
            if scene['name'] in scene_splits['train_track' if split_type == 'train' else 'val'] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                   instances.append(instance)
        return instances

    def _build_tracklet_annotations(self, track_instances):
        tracklet_annotations = []

        for instance in track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':

                ann_record = self.nusc.get(
                    'sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                scene_token = sample['scene_token']
                sample_data_lidar = self.nusc.get(
                    'sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append(
                    {"sample_data_lidar": sample_data_lidar, "box_anno": ann_record,"scene_token": scene_token})

            tracklet_annotations.append(track_anno)
        return tracklet_annotations

    def _build_frame(self, frame_anno):
        scene_token = frame_anno['scene_token']
        lang_input_ids = self.scene_lang_input_ids[scene_token]
        lang_attn_mask = self.scene_lang_attn_mask[scene_token]
        sample_data_lidar = frame_anno['sample_data_lidar']
        box_anno = frame_anno['box_anno']
        bbox = BoundingBox(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                           name=box_anno['category_name'])
        pcd_path = osp.join(
            self.data_root_dir, sample_data_lidar['filename'])
        # if osp.exists(pcd_path):
        #     return {"pcd": None, "bbox": None, 'anno': None}
        # else:
        #     print(pcd_path)
        #     return {"pcd": None, "bbox": None, 'anno': None}
        pcd = LidarPointCloud.from_file(pcd_path)

        cs_record = self.nusc.get(
            'calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pcd.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pcd.translate(np.array(cs_record['translation']))

        poserecord = self.nusc.get(
            'ego_pose', sample_data_lidar['ego_pose_token'])
        pcd.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pcd.translate(np.array(poserecord['translation']))

        pcd = PointCloud(points=pcd.points)
        return {"pcd": pcd, "bbox": bbox, 'anno': frame_anno,'nlp': {
                'input_ids': lang_input_ids,
                'attention_mask': lang_attn_mask,
            }}


def print_np(**kwargs):
    for k, v in kwargs.items():
        print(k, np.concatenate((v[:5], v[-5:]), axis=0))

class MultiInputTrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        return self.dataset.num_frames() * self.cfg.num_candidates_per_frame

    def _generate_item(self, template_frames, search_frame, candidate_id):
        batch = {
            'l_template_pcd' : [],
            'l_template_bbox_gt' : [],
            'l_template_mask_gt' : [],
            'l_template_bc_gt' : [],
            'l_template_mask_ref' : [],
            'l_template_bc_ref' : [],
        }
        for template_frame in template_frames:
            template_frame_pcd, template_frame_bbox = template_frame['pcd'], template_frame['bbox']

            # print('1. template_frame_pcd.nbr_points()=>', template_frame_pcd.nbr_points())
            if self.cfg.train_cfg.use_augmentation:
                template_frame_pcd, template_frame_bbox = augment3d(template_frame_pcd, template_frame_bbox)
            if self.cfg.train_cfg.use_z:
                if candidate_id == 0:
                    bbox_offset = np.zeros(4)
                else:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
                    bbox_offset[3] = bbox_offset[3] * (5 if self.cfg.degree else np.deg2rad(5))
            else:
                if candidate_id == 0:
                    bbox_offset = np.zeros(3)
                else:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=3)
                    bbox_offset[2] = bbox_offset[2] * (5 if self.cfg.degree else np.deg2rad(5))
            base_bbox = get_offset_box(
                template_frame_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z, offset_max=self.cfg.offset_max, degree=self.cfg.degree,  is_training=True)

            template_bbox_gt = transform_box(template_frame_bbox, base_bbox)

            template_pcd, template_bbox_ref = crop_and_center_pcd(
                template_frame_pcd, base_bbox, offset=self.cfg.template_offset, offset2=self.cfg.template_offset2, scale=self.cfg.template_scale, return_box=True)
            # print('2. template_pcd.nbr_points()=>', template_pcd.nbr_points())
            assert template_pcd.nbr_points() > 10, 'not enough multi template points'
            template_mask_ref = get_pcd_in_box_mask(template_pcd, template_bbox_ref).astype(np.float32)

            if candidate_id != 0:
                template_mask_ref[template_mask_ref == 0] = 0.2
                template_mask_ref[template_mask_ref == 1] = 0.8

            template_mask_gt = get_pcd_in_box_mask(template_pcd, template_bbox_gt).astype(np.float32)

            template_bc_ref = get_point_to_box_distance(template_pcd, template_bbox_ref)
            template_bc_gt = get_point_to_box_distance(template_pcd, template_bbox_gt)

            template_bbox_gt_label = np.array([template_bbox_gt.center[0], template_bbox_gt.center[1], template_bbox_gt.center[2], (template_bbox_gt.orientation.degrees
                                                if self.cfg.degree else template_bbox_gt.orientation.radians) * template_bbox_gt.orientation.axis[-1]])

            template_pcd, idx_t = resample_pcd(template_pcd, self.cfg.template_npts, return_idx=True, is_training=True)
            template_mask_gt = template_mask_gt[idx_t]
            template_mask_ref = template_mask_ref[idx_t]
            template_bc_gt = template_bc_gt[idx_t]
            template_bc_ref = template_bc_ref[idx_t]

            batch['l_template_pcd'].append(template_pcd.points.T)
            batch['l_template_bbox_gt'].append(template_bbox_gt_label)
            batch['l_template_mask_gt'].append(template_mask_gt)
            batch['l_template_bc_gt'].append(template_bc_gt)
            batch['l_template_mask_ref'].append(template_mask_ref)
            batch['l_template_bc_ref'].append(template_bc_ref)

        search_frame_pcd, search_frame_bbox = search_frame['pcd'], search_frame['bbox']
        lang_description = search_frame['nlp']
        # print('1. search_frame_pcd.nbr_points()=>', search_frame_pcd.nbr_points())
        if self.cfg.train_cfg.use_augmentation:
            search_frame_pcd, search_frame_bbox = augment3d(search_frame_pcd, search_frame_bbox)

        if self.cfg.train_cfg.use_z:
            if candidate_id == 0:
                bbox_offset = np.zeros(4)
            else:
                gaussian = KalmanFiltering(bnd=[1, 1, 1, 1])
                bbox_offset = gaussian.sample(1)[0]
                bbox_offset[1] /= 2.0
                bbox_offset[0] *= 2
        else:
            if candidate_id == 0:
                bbox_offset = np.zeros(3)
            else:
                gaussian = KalmanFiltering(bnd=[1, 1, 5])
                bbox_offset = gaussian.sample(1)[0]

        base_bbox = get_offset_box(search_frame_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z,
                                   offset_max=self.cfg.offset_max, degree=self.cfg.degree, is_training=True)

        search_bbox_gt = transform_box(search_frame_bbox, base_bbox)
        search_pcd = crop_and_center_pcd(search_frame_pcd, base_bbox, offset=self.cfg.search_offset,
                                         offset2=self.cfg.search_offset2, scale=self.cfg.search_scale)
        # print('2. search_pcd.nbr_points()=>', search_pcd.nbr_points())
        assert search_pcd.nbr_points() > 20, 'not enough search points'
        search_mask_gt = get_pcd_in_box_mask(search_pcd, search_bbox_gt).astype(np.float32)
        search_mask_ref = np.ones_like(search_mask_gt) * 0.5

        search_bc_gt = get_point_to_box_distance(search_pcd, search_bbox_gt)
        search_bc_ref = np.zeros((search_pcd.points.shape[1], 9), dtype=np.float32)

        search_bbox_gt_label = np.array(
            [search_bbox_gt.center[0], search_bbox_gt.center[1], search_bbox_gt.center[2],
             (search_bbox_gt.orientation.degrees if self.cfg.degree else search_bbox_gt.orientation.radians) *
             search_bbox_gt.orientation.axis[-1]])

        search_pcd, idx_s = resample_pcd(search_pcd, self.cfg.search_npts, return_idx=True, is_training=True)
        search_mask_gt = search_mask_gt[idx_s]
        search_mask_ref = search_mask_ref[idx_s]
        search_bc_gt = search_bc_gt[idx_s]
        search_bc_ref = search_bc_ref[idx_s]

        batch.update({
            'search_pcd': search_pcd.points.T,
            'search_bbox_gt': search_bbox_gt_label,
            'search_mask_gt': search_mask_gt,
            'search_bc_gt': search_bc_gt,
            'search_mask_ref': search_mask_ref,
            'search_bc_ref': search_bc_ref
        })
        return self._to_float_tensor(batch)

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            if k == 'lang_description':
                # v 是一个字典 {'input_ids': tensor, 'attention_mask': tensor}
                # 直接处理这个字典
                tensor_data[k] = {
                    'input_ids': torch.tensor(v['input_ids']),  # 添加batch维度
                    'attention_mask': torch.tensor(v['attention_mask']),
                }
            else:
                # 其他数据转换为FloatTensor
                tensor_data[k] = torch.FloatTensor(v)
        return tensor_data

    def __getitem__(self, idx):
        global_frame_id = idx // self.cfg.num_candidates_per_frame
        candidate_id = idx % self.cfg.num_candidates_per_frame
        tracklet_id, frame_id = self.dataset.get_tracklet_frame_id(global_frame_id)

        template_frames = []
        for front in range(self.cfg.template_set_size):
            pre_frame_id = max(frame_id - front - 1, 0)
            template_frames.append(self.dataset.get_frame(tracklet_id, pre_frame_id))
        search_frame = self.dataset.get_frame(tracklet_id, frame_id)

        try:
            return self._generate_item(template_frames, search_frame, candidate_id)
        except:
            # print('self[torch.randint(0, len(self), size=(1,)).item()]=>', self[torch.randint(0, len(self), size=(1,)).item()])
            return self[torch.randint(0, len(self), size=(1,)).item()]

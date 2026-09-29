import os.path as osp
import pandas as pd
import pickle as pkl
import random
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from tqdm import tqdm
import bisect
import torch

from .utils import *
from .base_dataset import BaseDataset, EvalDatasetWrapper
from utils import pl_ddp_rank, cal_3dbb_vertices, draw_bbox


def sample_data(data_list, percentage=0.1, seed=42):
    # randomly get 10% of the data
    random.seed(seed)
    sample_size = int(len(data_list) * percentage)
    return random.sample(data_list, sample_size)


class GSOT3D(BaseDataset):

    def get_comp_pcd(self, tracklet_id):
        raise ValueError('func "get_comp_pcd" in GSOT3D Not implemented')
        pass

    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        assert cfg.category_name in [
            'backpack', 'balloon', 'basketball', 'basketball_player', 'bench',
            'bicycle', 'book', 'bottle', 'bus', 'can',
            'car', 'cat', 'chair', 'charger', 'computer_case',
            'cup', 'cyclist', 'displayer', 'dog', 'doll',
            'earphone', 'electric_bike', 'electro-tricycle', 'fire_extinguisher', 'football',
            'football_player', 'frisbee', 'gamepad', 'handbag', 'handcart',
            'headwear', 'instant_noodles', 'keyboard', 'laptop', 'medicine_box',
            'melon', 'mineral_water', 'motorbike', 'pedestrian', 'pillow',
            'printer', 'shampoo', 'shoes', 'sign', 'sink',
            'stool', 'suitcase', 'table', 'tools', 'toy_transportation',
            'trash_bin', 'umbrella', 'van', 'warning_post', 'All'
        ]
        # read train/test split
        train_split = osp.join(cfg.data_root_dir, 'train_set.txt')
        test_split = osp.join(cfg.data_root_dir, 'test_set.txt')
        with open(train_split, 'r') as f:
            train_scene_ids = f.readlines()
            train_scene_ids = [train_scene.strip() for train_scene in train_scene_ids]
        with open(test_split, 'r') as f:
            test_scene_ids = f.readlines()
            test_scene_ids = [test_scene.strip() for test_scene in test_scene_ids]

        if cfg.scale == 'FULL':
            split_type_to_scene_ids = dict(
                train=train_scene_ids,
                val=test_scene_ids,
                test=test_scene_ids
            )
        elif cfg.scale == 'TINY':
            train_scene_ids_sampled = sample_data(train_scene_ids, percentage=0.2)
            test_scene_ids_sampled = sample_data(test_scene_ids, percentage=0.2)
            split_type_to_scene_ids = dict(
                train=train_scene_ids_sampled,
                val=test_scene_ids_sampled,
                test=test_scene_ids_sampled
            )
        else:
            print(f'scale must be FULL or TINY, got{cfg.scale} instead')
            raise NotImplementedError

        self.preload_offset = cfg.preload_offset if split_type == 'train' else -1
        self.cache = cfg.cache_train if split_type == 'train' else cfg.cache_eval
        self.calibration_info = {}

        scene_ids = split_type_to_scene_ids[split_type]
        self.tracklet_annotations = self._build_tracklet_annotations(scene_ids)

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

        if self.cache:
            if not cfg.debug:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir,
                    f'GSOT3D_{cfg.scale}_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            else:
                cache_file_dir = osp.join(
                    self.cfg.data_root_dir,
                    f'GSOT3D_{cfg.scale}_DEBUG_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            if osp.exists(cache_file_dir):
                self.log.info(f'Loading data from cache file {cache_file_dir}')
                with open(cache_file_dir, 'rb') as f:
                    tracklets = pkl.load(f)
            else:
                tracklets = []
                for tracklet_id in tqdm(
                        range(len(self.tracklet_annotations)),
                        desc='[%6s]Loading pcds ' % self.split_type.upper(),
                        disable=pl_ddp_rank() != 0):

                    frames = []
                    for frame_anno in self.tracklet_annotations[tracklet_id]:
                        frames.append(self._build_frame(frame_anno))

                    comp_template_pcd = merge_template_pcds(
                        [frame['pcd'] for frame in frames],
                        [frame['bbox'] for frame in frames],
                        offset=cfg.model_offset,
                        scale=cfg.model_scale
                    )
                    if self.preload_offset > 0:
                        for frame in frames:
                            frame['pcd'] = crop_pcd_axis_aligned(
                                frame['pcd'], frame['bbox'], offset=self.preload_offset)

                    tracklets.append({
                        'comp_template_pcd': comp_template_pcd,
                        'frames': frames
                    })

                with open(cache_file_dir, 'wb') as f:
                    self.log.info(
                        f'Saving data to cache file {cache_file_dir}')
                    pkl.dump(tracklets, f)
            self.tracklets = tracklets
        else:
            self.tracklets = None

    def get_dataset(self):
        if self.split_type == 'train':
            return TrainDatasetWrapper(self, self.cfg, self.log)
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

    def get_comp_template_pcd(self, tracklet_id):
        comp_template_pcd = self.tracklets[tracklet_id]['comp_template_pcd']
        return comp_template_pcd

    def get_tracklet_frame_id(self, idx):
        tracklet_id = bisect.bisect_right(
            self.tracklet_ed_frame_id, idx)
        assert self.tracklet_st_frame_id[tracklet_id] <= idx < self.tracklet_ed_frame_id[tracklet_id]
        frame_id = idx - \
            self.tracklet_st_frame_id[tracklet_id]
        return tracklet_id, frame_id

    def _build_tracklet_annotations(self, scene_ids):
        tracklet_annotations = []
        for scene_id in tqdm(
                scene_ids, desc='[%6s]Loading annos' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
            annotation_file_dir = osp.join(
                self.cfg.data_root_dir, f'data/{scene_id}/gt.txt')
            data = pd.read_csv(
                annotation_file_dir,
                sep=' ',
                names=[
                    "classname", "pointN", "out-of-view", "occlusion",
                    "length", "width", "height",
                    "center_x", "center_y", "center_z",
                    "roll", "pitch", "yaw"
                ]
            )
            data.insert(0, 'frame', range(1, len(data) + 1))   # add frame id
            if self.cfg.category_name == 'All':
                # we don't have 'DontCare'
                data = data
                # data = data[(data["type"] == 'Car') |
                #             (data["type"] == 'Van') |
                #             (data["type"] == 'Pedestrian') |
                #             (data["type"] == 'Cyclist')]
            else:
                data = data[data["classname"] == self.cfg.category_name]

            data.insert(loc=0, column='scene', value=scene_id)

            data_tracklet = data.copy()
            data_tracklet = data_tracklet.reset_index(drop=True)
            tracklet_anno = [anno for index, anno in data_tracklet.iterrows()]
            tracklet_annotations.append(tracklet_anno)
        return tracklet_annotations

    @staticmethod
    def _read_calibration_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    def _build_frame(self, frame_anno):
        scene_id = frame_anno['scene']
        frame_id = frame_anno['frame']
        if scene_id in self.calibration_info:
            calib = self.calibration_info[scene_id]     # TODO
        else:
            calib = self._read_calibration_file(
                osp.join(self.cfg.data_root_dir, f'data/{scene_id}/calib.json'))
            self.calibration_info[scene_id] = calib
        # velo_to_cam = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))   TODO

        assert self.cfg.coordinate_mode in ['camera', 'velodyne']

        if self.cfg.coordinate_mode == 'camera':    # TODO
            bbox_center = [frame_anno["x"], frame_anno["y"] -
                           frame_anno["height"] / 2, frame_anno["z"]]
            size = [frame_anno["width"],
                    frame_anno["length"], frame_anno["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=frame_anno["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            bbox = BoundingBox(bbox_center, size, orientation)
        else:
            # coordinate is in velodyne, which is different from kitti(camera coordinate)
            box_center_velo = np.array(
                [frame_anno["center_x"], frame_anno["center_y"], frame_anno["center_z"]])
            size = [frame_anno["width"], frame_anno["length"], frame_anno["height"]]

            orientation = (Quaternion(axis=[0, 0, -1], radians=frame_anno["yaw"]) *
                           Quaternion(axis=[0, -1, 0], radians=frame_anno["pitch"]) *
                           Quaternion(axis=[-1, 0, 0], radians=frame_anno["roll"]))
            bbox = BoundingBox(box_center_velo, size, orientation)
        try:
            pcd_file_dir = osp.join(
                self.cfg.data_root_dir, f'data/{scene_id}/point_cloud_bin/{str(frame_id).zfill(5)}.bin')
            pcd = PointCloud(np.fromfile(
                pcd_file_dir, dtype=np.float32).reshape(-1, 4).T)
            if self.cfg.coordinate_mode == 'camera':
                pass
                # pcd.transform(velo_to_cam)    TODO
        except Exception as e:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            pcd = PointCloud(np.array([[0, 0, 0]]).T)
            print(f'Error: {e}')
        # visualize the point cloud and bounding box
        # draw_bbox(bbox, pcd, name=f"bbox_{scene_id}_{frame_id}")

        return {'pcd': pcd, 'bbox': bbox, 'anno': frame_anno}


def print_np(**kwargs):
    for k, v in kwargs.items():
        print(k, np.concatenate((v[:5], v[-5:]), axis=0))


class TrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        return self.dataset.num_frames() * self.cfg.num_candidates_per_frame

    def _generate_item(self, st_frame, template_frame, search_frame, tracklet_id, frame_id, candidate_id):
        st_frame_pcd, st_frame_bbox = st_frame['pcd'], st_frame['bbox']
        template_frame_pcd, template_frame_bbox = template_frame['pcd'], template_frame['bbox']
        search_frame_pcd, search_frame_bbox = search_frame['pcd'], search_frame['bbox']

        if self.cfg.train_cfg.use_augmentation:
            template_frame_pcd, template_frame_bbox = augment3d(
                template_frame_pcd, template_frame_bbox)
            search_frame_pcd, search_frame_bbox = augment3d(
                search_frame_pcd, search_frame_bbox)

        if self.cfg.train_cfg.use_z:
            if candidate_id == 0:
                bbox_offset = np.zeros(4)
            else:
                bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
                bbox_offset[3] = bbox_offset[3] * \
                    (5 if self.cfg.degree else np.deg2rad(5))
        else:
            if candidate_id == 0:
                bbox_offset = np.zeros(3)
            else:
                bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=3)
                bbox_offset[2] = bbox_offset[2] * \
                    (5 if self.cfg.degree else np.deg2rad(5))
        base_bbox = get_offset_box(
            template_frame_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z, offset_max=self.cfg.offset_max, degree=self.cfg.degree,  is_training=True)
        # print(base_bbox)
        template_bbox_gt = transform_box(template_frame_bbox, base_bbox)

        template_pcd, template_bbox_ref = crop_and_center_pcd(
            template_frame_pcd, base_bbox, offset=self.cfg.template_offset, offset2=self.cfg.template_offset2, scale=self.cfg.template_scale, return_box=True)

        assert template_pcd.nbr_points() > 20, 'not enough template points'

        template_mask_ref = get_pcd_in_box_mask(
            template_pcd, template_bbox_ref).astype(np.float32)

        if candidate_id != 0:
            template_mask_ref[template_mask_ref == 0] = 0.2
            template_mask_ref[template_mask_ref == 1] = 0.8

        template_mask_gt = get_pcd_in_box_mask(
            template_pcd, template_bbox_gt).astype(np.float32)

        template_bc_ref = get_point_to_box_distance(
            template_pcd, template_bbox_ref)
        template_bc_gt = get_point_to_box_distance(
            template_pcd, template_bbox_gt)

        template_bbox_gt_orientation_angle = template_bbox_gt.orientation.degrees \
            if self.cfg.degree else template_bbox_gt.orientation.radians
        # 6-DoF bbox
        template_bbox_gt_label = np.array([
            template_bbox_gt.center[0], template_bbox_gt.center[1], template_bbox_gt.center[2],
            template_bbox_gt_orientation_angle * template_bbox_gt.orientation.axis[0],
            template_bbox_gt_orientation_angle * template_bbox_gt.orientation.axis[1],
            template_bbox_gt_orientation_angle * template_bbox_gt.orientation.axis[2]
        ])

        template_pcd, idx_t = resample_pcd(
            template_pcd, self.cfg.template_npts, return_idx=True, is_training=True)

        template_mask_gt = template_mask_gt[idx_t]
        template_mask_ref = template_mask_ref[idx_t]
        template_bc_gt = template_bc_gt[idx_t]
        template_bc_ref = template_bc_ref[idx_t]

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
                gaussian = KalmanFiltering(
                    bnd=[1, 1, 5])
                bbox_offset = gaussian.sample(1)[0]

        base_bbox = get_offset_box(
            search_frame_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z, offset_max=self.cfg.offset_max, degree=self.cfg.degree, is_training=True)

        search_bbox_gt = transform_box(search_frame_bbox, base_bbox)
        search_pcd = crop_and_center_pcd(
            search_frame_pcd, base_bbox, offset=self.cfg.search_offset, offset2=self.cfg.search_offset2, scale=self.cfg.search_scale)
        assert search_pcd.nbr_points() > 20, 'not enough search points'
        search_mask_gt = get_pcd_in_box_mask(
            search_pcd, search_bbox_gt).astype(np.float32)
        search_mask_ref = np.ones_like(search_mask_gt) * 0.5

        search_bc_gt = get_point_to_box_distance(
            search_pcd, search_bbox_gt)
        search_bc_ref = np.zeros(
            (search_pcd.points.shape[1], 9), dtype=np.float32)      # 9 = 8 + 1, 8 for corners and 1 for center

        search_bbox_gt_orientation_angle = search_bbox_gt.orientation.degrees \
            if self.cfg.degree else search_bbox_gt.orientation.radians
        # 6-DoF bbox
        search_bbox_gt_label = np.array([
            search_bbox_gt.center[0], search_bbox_gt.center[1], search_bbox_gt.center[2],
            search_bbox_gt_orientation_angle * search_bbox_gt.orientation.axis[0],
            search_bbox_gt_orientation_angle * search_bbox_gt.orientation.axis[1],
            search_bbox_gt_orientation_angle * search_bbox_gt.orientation.axis[2]
        ])

        search_pcd, idx_s = resample_pcd(
            search_pcd, self.cfg.search_npts, return_idx=True, is_training=True)
        search_mask_gt = search_mask_gt[idx_s]
        search_mask_ref = search_mask_ref[idx_s]
        search_bc_gt = search_bc_gt[idx_s]
        search_bc_ref = search_bc_ref[idx_s]

        motion_bbox_gt = transform_box(search_bbox_gt, template_bbox_gt)
        # 6-DoF bbox
        motion_bbox_gt_orientation_angle = motion_bbox_gt.orientation.degrees \
            if self.cfg.degree else motion_bbox_gt.orientation.radians
        motion_bbox_gt_label = np.array([
            motion_bbox_gt.center[0], motion_bbox_gt.center[1], motion_bbox_gt.center[2],
            motion_bbox_gt_orientation_angle * motion_bbox_gt.orientation.axis[0],
            motion_bbox_gt_orientation_angle * motion_bbox_gt.orientation.axis[1],
            motion_bbox_gt_orientation_angle * motion_bbox_gt.orientation.axis[2]
        ])

        data = {
            'template_pcd': template_pcd.points.T,          # N, 3
            'search_pcd': search_pcd.points.T,              # M, 3
            'search_bbox_gt': search_bbox_gt_label,         # 4,
            'template_bbox_gt': template_bbox_gt_label,     # 4,
            'motion_bbox_gt': motion_bbox_gt_label,         # 4,
            'search_mask_gt': search_mask_gt,               # M,
            'template_mask_gt': template_mask_gt,           # N,
            'search_mask_ref': search_mask_ref,             # M,
            'template_mask_ref': template_mask_ref,         # N,
            'search_bc_ref': search_bc_ref,     # boxCloud  # M, 9
            'template_bc_ref': template_bc_ref,             # N, 9
            'search_bc_gt': search_bc_gt,                   # M, 9
            'template_bc_gt': template_bc_gt,               # N, 9
            # 'is_dynamic_gt': is_dynamic_gt,
        }
        return self._to_float_tensor(data)

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.FloatTensor(v)
        return tensor_data

    def __getitem__(self, idx):
        global_frame_id = idx // self.cfg.num_candidates_per_frame
        candidate_id = idx % self.cfg.num_candidates_per_frame

        tracklet_id, frame_id = self.dataset.get_tracklet_frame_id(
            global_frame_id)
        pre_frame_id = max(frame_id-1, 0)
        st_frame_id = 0
        st_frame, template_frame, search_frame = self.dataset.get_frame(
            tracklet_id, st_frame_id), self.dataset.get_frame(tracklet_id, pre_frame_id), self.dataset.get_frame(tracklet_id, frame_id)

        # comp_template_pcd = self.dataset.get_comp_template_pcd(tracklet_id)

        try:
            return self._generate_item(st_frame, template_frame, search_frame, tracklet_id, frame_id, candidate_id)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]

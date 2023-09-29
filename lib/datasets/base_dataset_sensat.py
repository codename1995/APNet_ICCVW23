import copy
import pickle
import random
import os
import math
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from matplotlib import pyplot as plt

from lib.config import config
from lib.utils.data_process import DataProcessing as DP
from lib.datasets.helper_ply import read_ply
from lib.utils.utils import sensaturban_color_map

from sklearn.neighbors import KDTree
from scipy.spatial.ckdtree import cKDTree
import lib.utils.distributed as dist


def worker_init_fn(worker_id):
    # print("worker: {} === rank: {}".format(worker_id, dist.get_rank()))
    worker_info = get_worker_info()
    # worker_seed = (worker_id + torch.initial_seed() + dist.get_rank()) % 2**32
    worker_seed = (torch.initial_seed() + dist.get_rank()) % 2**32
    np.random.seed(worker_seed)


    dataset = worker_info.dataset
    dataset.init_possibility(worker_id)

def enhance_some_classes_in_img_label(img_label, classes=[4, 6]):
    # classes: 6-railway
    for i in classes:
        img_label[i+1] *= 20
    return img_label

def complete_img_label(img_label, iterations=3, enhanced_classes=[6]):
    enhance_some_classes_in_img_label(img_label, classes=enhanced_classes)
    C = img_label.shape[0]
    img_label = img_label.transpose((1, 2, 0))  # -> H*W*C
    kernel = np.ones((3, 3), np.uint8)
    for i in range(iterations):
        idx_holes = np.repeat(np.expand_dims(np.argmax(img_label, axis=2) == 0, 2), C, axis=2)
        new_img_label = cv2.dilate(img_label, kernel=kernel, iterations=iterations)
        # new_img_label = img_label * (~idx_holes) + new_img_label * idx_holes
        img_label = new_img_label
    return img_label.transpose((2, 0, 1)).copy()


class SensatBaseDataset(Dataset):
    def __init__(self,
                 mode,
                 cfg_dataset,
                 cfg_model,
                 cfg_trick,
                 data_list_path,
                 ignore_label=255,
                 threshold_possibility=0.1,
                 visualize=False,
                 debug=False,
                 ):
        assert mode in ["training", "tra_val", "tra_all", "eval_val", "testing"]
        # Difference of three modes:
        # 1. training: labels, data augmentation for both data and labels
        # 2. tra_val:  labels, no data augmentation. During training, use val set to evaluate the model.
        # 3. eval_val: labels, no data augmentation
        # 4. testing: no labels, no data augmentation
        # Dataset basic
        self.mode = mode
        self.name = 'SensatUrban'
        self.dataset_path = cfg_dataset.ROOT
        self.num_classes = cfg_dataset.NUM_CLASSES
        self.num_points = cfg_dataset.NUM_POINTS
        self.ignored_labels = np.sort([ignore_label])
        self.threshold_possibility = threshold_possibility
        self.num_samples = cfg_dataset.SAMPLES_TRA if mode == "training" else cfg_dataset.SAMPLES_VAL
        # Model basic
        self.src_pts_in_fusion_module = cfg_model.SRC_PTS
        # RandLA-Net
        self.num_layers = cfg_model.NUM_LAYERS
        self.k_n = cfg_model.K_N
        self.sub_sampling_ratio = cfg_model.SUB_SAMPLING_RATIO
        # HRNet
        pass
        # New data augmentation
        self.city_wise_sub = cfg_trick.CITY_WISE_SUB
        if mode == 'training':
            self.augment_rotation = cfg_trick.POINT_ROTATION
            self.augment_scale_anisotropic = cfg_trick.POINT_SCALE_ANISOTROPIC
            self.augment_scale = cfg_trick.POINT_SCALE
            self.augment_symmetreis = cfg_trick.POINT_SYMMETRIES
            self.augment_point_noise = cfg_trick.POINT_NOISE
            self.augment_point_noise_max = cfg_trick.POINT_NOISE_MAX
            self.color_dropping = cfg_trick.COLOR_DROPPING
            self.color_auto_contrast = cfg_trick.COLOR_AUTO_CONTRAST
        elif mode == 'tra_eval':
            pass
        elif mode == 'eval_val' or mode == 'testing':
            self.augment_rotation = cfg_trick.POINT_ROTATION
            self.augment_scale_anisotropic = cfg_trick.POINT_SCALE_ANISOTROPIC
            self.augment_scale = cfg_trick.POINT_SCALE
            self.augment_symmetreis = cfg_trick.POINT_SYMMETRIES
            self.augment_point_noise = cfg_trick.POINT_NOISE
            self.augment_point_noise_max = cfg_trick.POINT_NOISE_MAX
        # Visualize
        self.visualize = visualize
        # Debug
        self.debug = debug


        # Internal default value
        self.mean = [0.485, 0.456, 0.406, 36.40]  # R, G, B, Height
        self.std = [0.229, 0.224, 0.225, 13.88]  # R, G, B, Height
        self.pixel_size = 0.04  # unit: meter
        self.safe_margin = 6  # unit: meter
        self.image_size = [512, 512]
        self.random_points = True  # False is only valid for training set. It needs more implementation for val and tst set.
        self.image_label_mapping = {0: ignore_label,
                                    1: 0,
                                    2: 1,
                                    3: 2,
                                    4: 3,
                                    5: 4,
                                    6: 5,
                                    7: 6,
                                    8: 7,
                                    9: 8,
                                    10: 9,
                                    11: 10,
                                    12: 11,
                                    13: 12,
                                    }

        # A brief description for how this dataloader works
        # When initialize == __init__():
        #   read_files(): join root path with specific file names. return a dict, `files`
        #   (tra, tra_val)
        # When initialize workers externally == worker_init_fn():
        #   (each worker)
        #       load_trees(): load all kd-tree to memory
        #       init_possibility(): initialize point possibility
        #   (eval_val, tst)
        #       sample_points_cache(): load all sub_points files and sample points based on probability.
        #                              return `samples` with `cloud_idx, center_point_idx, queried_point_idx`
        # When generate one single sample == __getitem__():
        #   1. find_query_point_and_its_nns(): sample query point and its nearest neighbours
        #   2. get_data(): load point cloud and source point cloud
        #   3. project2BEV(): project source point cloud to BEV image
        #   9. process_data(): map labels, normalize xyz and rgb
        #   9. data augmentation
        # When form a batch == collate_fn():
        #   1. tf_map(): mainly used to generate neighbour_idx, sub_points etc. that are used to down-/up-sample
        #   2. transform numpy array to torch.Tensor
        if mode in ["training", "tra_all", "tra_val"]:
            self.files = self.read_files(self.dataset_path, data_list_path, mode)
            self.trees = self.load_trees()
            self.init_possibility(seed=1)

        self.bike_probability = 0.1
        if "110" in self.dataset_path and mode == "training" or mode == "tra_all":
            self.bikes = self.load_bike_files(self.dataset_path, mode)

        # |     |  training  |  tra_all  |  tra_val  | eval_val  |  testing  |
        # |     |            |           |           |           |           |
        # | wImgL|    1     |       1     |     1    |     0   |       0     |
        # | wL|     1      |       1     |     1    |      1   |       0     |
        self.wImgL = mode not in ["eval_val", "testing"]
        self.wL = mode not in ["testing"]
        self.wkNN = 'KPCONV' in cfg_model.FUSION
        # if mode != "training":
        #     self.samples = self.sample_points_cache()

    def __len__(self):
        return self.num_samples

    def get_class_weights(self, weights_type="ce"):
        if weights_type == "ce":
            num_per_class = np.array([439244099, 544284030, 861977674, 20606217,
                                      3044907, 49452238, 498977, 131470199,
                                      26669467, 37557130, 43733933, 174125,
                                      7088841])
            weights = num_per_class / float(sum(num_per_class))
            weights = 1 / np.sqrt(weights)  # sart
            weights = np.expand_dims(weights, axis=0)

        return weights

    def read_files(self, root, list_path, mode):
        files = []
        short_names = [line.strip() for line in open(list_path)]
        # sort short_names according to the number of points. Useful for equally distribute files to different gpus.
        cnt = [int(name.split("_")[-1]) for name in short_names]
        short_names = [name for _, name in sorted(zip(cnt, short_names), key=lambda pair: pair[0])]
        if self.debug:
            short_names = short_names[:20]

        for short_name in short_names:
            split = "train"
            files.append({
                "name": short_name,
                "points": join(root, split, "original_points", short_name + ".ply"),
                "subpoints": join(root, split, "subpoints_0.200", short_name + ".ply"),
                "kdtree": join(root, split, "subpoints_0.200", short_name + "_KDTree.pkl"),
                "proj": join(root, split, "subpoints_0.200", short_name + "_proj.pkl")
            })

        ws = dist.get_world_size()
        if ws >= 2:
            # distribute data to different GPU
            rank = dist.get_rank()
            files = files[rank::ws]
            names = str([f['name'] for f in files])
            print("rank: {} ===".format(rank) + names)

        return files

    def load_bike_files(self, root, mode):
        if mode == "training":
            list_path = "./data/list/sensaturban/tra_list_bike.txt"
        else:
            # mode == "tra_all"
            list_path = "./data/list/sensaturban/tra_val_list_bike.txt"
        short_names = [line.strip() for line in open(list_path)]
        src_bike = []
        for short_name in short_names:
            bike_file = os.path.join(root, 'bike_instance', short_name + ".ply")
            points = read_ply(bike_file)
            src_bike.append(self.get_pts(points))

        return src_bike

    def load_trees(self):
        trees = []
        for i, file in enumerate(self.files):
            with open(file['kdtree'], 'rb') as f:
                tree = pickle.load(f)
            trees.append(tree)
        return trees

    def init_possibility(self, seed=0):
        np.random.seed(seed)
        self.possibility = []
        self.min_possibility = np.zeros(len(self.files)).astype(float)
        for i, file in enumerate(self.files):
            tree = self.trees[i]
            self.possibility.append(np.random.rand(tree.data.shape[0]) * 1e-3)
            self.min_possibility[i] = float(np.min(self.possibility[-1]))

    def get_pts(self, points):
        pts = np.vstack((points['x'], points['y'], points['z'],
                         points['red'], points['green'], points['blue']))
        labels = points['class'] if self.mode != "testing" else np.ones((pts.shape[1]))

        pts = np.vstack((pts, labels)).T
        return pts

    def get_pts_separatly(self, points):
        xyz = np.vstack((points['x'], points['y'], points['z'])).T
        rgb = (np.vstack((points['red'], points['green'], points['blue'])).T).astype(float)
        labels = points['class'] if self.mode != "testing" else np.ones((len(xyz)))
        return xyz, rgb, labels

    def load_points(self, cloud_idx, src_points=False):
        path_key = "points" if src_points else "subpoints"
        if self.mode in ["training", "tra_val"]:
            point_path = self.files[cloud_idx][path_key]
            points = read_ply(point_path)
        else:
            points = self.points[cloud_idx][path_key]

        if src_points:
            return self.get_pts(points)
        else:
            return self.get_pts_separatly(points)

    def get_data(self, cloud_idx, selected_idx):
        file_path = self.files[cloud_idx]

        # load points
        xyz, rgb, labels = self.load_points(cloud_idx, src_points=False)
        xyz = xyz[selected_idx, :]
        rgb = rgb[selected_idx, :]
        labels = labels[selected_idx]

        # load source points
        pts = self.load_points(cloud_idx, src_points=True)

        # cut points from original points
        # V1: cut a square
        # x0, y0 = np.min(xyz[:, :2]) - self.safe_margin
        # x1, y1 = np.max(xyz[:, :2]) + self.safe_margin
        # valid_idx = (x0 <= pts[:, 0]) & (pts[:, 0] < x1) & \
        #             (y0 <= pts[:, 1]) & (pts[:, 1] < y1)
        # V2: cut a larger circle
        x0 = np.min(xyz[:, 0])
        y0 = np.min(xyz[:, 1])
        x1 = np.max(xyz[:, 0])
        y1 = np.max(xyz[:, 1])
        x0_5 = (x0+x1)/2 # x 0.5
        y0_5 = (y0+y1)/2 # y 0.5
        radius = (x1-x0 + y1-y0) / 4 + self.safe_margin
        valid_idx = ((pts[:, 0] - x0_5) ** 2 + (pts[:, 1] - y0_5) ** 2) <= radius**2
        pts = pts[valid_idx, :]

        return xyz.copy(), rgb.copy(), labels.copy(), pts.copy()

    @staticmethod
    def getIdx4PointInRegion(x_v, y_v, x0, y0, x1, y1):
        return np.logical_and(
            np.logical_and(x0 <= x_v, x_v < x1),
            np.logical_and(y0 <= y_v, y_v < y1)
        )

    def imshowImgLabel(self, img_label, with_invalid_label=False):
        n = self.num_classes
        if with_invalid_label:
            n += 1
        w, h = img_label.shape
        rgb = np.zeros((w, h, 3)).astype(np.uint8)
        for i in range(n):
            rgb[img_label == i] = sensaturban_color_map(with_invalid_label, [255,255,255])[i]
        plt.imshow(rgb)
        plt.show()

    def gen_img_label(self, w, h, pts=None, first_idx=None, r=None, c=None, hash_idx2=None, num_class=13, iterations=0):
        # in both algorithms, 0 - invalid pixels; 1 ~ num_class+1 - semantic labels
        # while complete == 0, two algorithm output same results
        if first_idx is not None:
            # random pick a label from the pillar
            img_label = np.zeros((w, h))
            img_label[r, c] = pts[first_idx, 6] + 1
        else:
            temp_img_label = np.zeros((num_class+1, w, h))
            v, cnt = np.unique(hash_idx2, return_counts=True)
            temp_img_label[v%num_class+1, v//num_class%w, v//num_class//w] = cnt
            temp_img_label = complete_img_label(temp_img_label, iterations=iterations)
            img_label = np.argmax(temp_img_label, axis=0)

        return img_label

    def insert_bike(self, xyz, rgb, labels):
        bike = None
        if np.random.random() <= self.bike_probability:
            bike_idx = np.random.randint(len(self.bikes))
            bike_xyz = self.bikes[bike_idx][:, :3]
            bike_rgb = self.bikes[bike_idx][:, 3:6].astype(np.float32)

            bike_xyz = self.rotate_pc([bike_xyz, ])[0]
            bike_xyz = self.scale_pc([bike_xyz, ])[0]

            ground_idx = np.where(labels == 0)[0]
            if len(ground_idx) > 0:
                ground_idx = np.random.choice(ground_idx, 1)
                dst_xyz = xyz[ground_idx][0]
                # move bike
                bike_xyz[:, :2] += (dst_xyz[:2] - np.mean(bike_xyz[:, :2], axis=0))
                bike_xyz[:, 2] += (dst_xyz[2] - np.min(bike_xyz[:, 2]))  # Height constraint

                sub_bike_xyz, sub_bike_rgb = DP.grid_sub_sampling(bike_xyz.astype(np.float32), bike_rgb, grid_size=0.2)
                n_bike_pts = len(sub_bike_xyz)

                # crop xyz
                n_pts = self.num_points
                if n_bike_pts <= 0.1 * n_pts:
                    random_pick = np.random.choice(n_pts, n_pts - n_bike_pts, replace=False)
                    shuffle_pick = np.random.choice(n_pts, n_pts, replace=False)
                    xyz = np.vstack((xyz[random_pick, :], sub_bike_xyz))[shuffle_pick, :]
                    rgb = np.vstack((rgb[random_pick, :], sub_bike_rgb))[shuffle_pick, :]
                    labels = np.hstack((labels[random_pick], np.ones(n_bike_pts) * 11))[shuffle_pick]
                else:
                    random_pick = np.random.choice(n_pts+n_bike_pts, n_pts, replace=False)  # Also shuffle the points
                    xyz = np.vstack((xyz, sub_bike_xyz))[random_pick, :]
                    rgb = np.vstack((rgb, sub_bike_rgb))[random_pick, :]
                    labels = np.hstack((labels, np.ones(n_bike_pts) * 11))[random_pick].astype(np.int8)

                bike = {}
                bike['xyz'] = bike_xyz
                bike['rgb'] = bike_rgb
                bike['labels'] = np.ones((len(bike_xyz), 1)) * 11

        return xyz, rgb, labels, bike

    def project2BEV(self, pts, xyz, rgb, labels, pixel_size, center_point=None):
        """
        1. Project the source points to an aerial view image
        2. At the same time, calculate the sub_pts location in image
        @param pts: source points
        @param sub_pts: sub-downsampled points
        @param pixel_size:
        @return:
        """
        # Calculate corners V1: based on src_pts
        # x0 = np.min(pts[:, 0])
        # y0 = np.min(pts[:, 1])
        # x1 = np.max(pts[:, 0])
        # y1 = np.max(pts[:, 1])
        # w = int( (x1-x0) / pixel_size) + 1
        # h = int( (y1-y0) / pixel_size) + 1
        # Calculate corners V2: based on center of sub_pts
        mean_pt = np.mean(xyz, axis=0)
        w = self.image_size[0]
        h = self.image_size[1]
        x0 = mean_pt[0] - w//2 * self.pixel_size
        y0 = mean_pt[1] - h//2 * self.pixel_size
        x1 = mean_pt[0] + w//2 * self.pixel_size
        y1 = mean_pt[1] + h//2 * self.pixel_size

        # init image, mask, etc
        img = np.zeros((w, h, 3))
        mask = np.zeros((w, h))
        img_cnt = np.zeros((w, h))

        # translate sub_pts and center_point to image coords
        if self.mode == 'training':
            xyz, rgb, labels, bike = self.insert_bike(xyz, rgb, labels)
        else:
            bike = None
        temp_xy = (xyz[:, :2] - np.array((x0, y0))) / self.pixel_size / np.array((w, h))
        pts_img_idx = temp_xy * 2 - 1 # scale to [-1, 1)
        pts_img_idx = pts_img_idx[:, ::-1]  # swap x and y
        pts_img_idx = np.array([pts_img_idx], dtype=np.float)
        cp_img_idx = np.array((0, 0))
        if center_point is not None:
            cp_img_idx = (center_point[0, :2] - np.array((x0, y0))) / self.pixel_size / np.array((w, h))

        # translate src_pts to image coords and generate the image
        if bike is not None:
            bike_pts = np.hstack((bike['xyz'], bike['rgb'], bike['labels']))
            pts = np.vstack((pts, bike_pts))
        idxInRegion = self.getIdx4PointInRegion(pts[:, 0], pts[:, 1], x0, y0, x1, y1)
        pts = pts[idxInRegion, ...]
        xs = ((pts[:, 0] - x0) / pixel_size).astype(int)  # nparray.astype is a floor int
        ys = ((pts[:, 1] - y0) / pixel_size).astype(int)
        idxInRegion = self.getIdx4PointInRegion(xs, ys, 0, 0, self.image_size[0], self.image_size[1])
        xs = xs[idxInRegion]
        ys = ys[idxInRegion]
        pts = pts[idxInRegion]
        # shuffle src_pts
        shffled_idx = np.random.choice(range(len(xs)), len(xs), replace=False)
        xs = xs[shffled_idx]
        ys = ys[shffled_idx]
        pts = pts[shffled_idx]
        hash_idx = xs + ys * w
        valid_idx, first_idx, cnt = np.unique(hash_idx, return_index=True, return_counts=True)
        r = (valid_idx % w).astype(int)
        c = (valid_idx // w).astype(int)
        if (r>=h).any() or (c>=h).any:
            pass
        img[r, c] = pts[first_idx, 5:2:-1]  # BGR
        if self.wImgL:
            hash_idx2 = (hash_idx * self.num_classes + pts[:, 6]).astype(int)
            img_label = self.gen_img_label(w, h, hash_idx2=hash_idx2, num_class=self.num_classes, iterations=2)
        else:
            img_label = None
        mask[r, c] = 1
        img_cnt[r, c] = cnt

        return xyz, rgb, labels, img, img_label, mask, img_cnt, pts_img_idx, cp_img_idx

    @staticmethod
    def crop_src_points(ds_xyz, src_xyz, src_labels, num_points, mode='training'):
        def pick_valid(xyz, a, idx, b=None, c=None, d=None):
            if b is None:
                return xyz[idx, :], a[idx]
            elif d is None:
                return xyz[idx, :], a[idx], b[idx], c[idx]
            else:
                return xyz[idx, :], a[idx], b[idx], c[idx], d[idx]

        # randomly delete half of source point to speed up
        if mode == "training" or mode == "tra_val":
            random_pick = np.random.choice(len(src_xyz), len(src_xyz)//2, replace=False)
            xyz, labels = pick_valid(src_xyz, src_labels, random_pick)
            src_idx = np.arange(3)
        else:
            xyz = copy.deepcopy(src_xyz)
            labels = copy.deepcopy(src_labels)
            src_idx = np.arange(len(src_xyz))

        # find the nearest neighbour of source points in down-sampled points
        # delete src points with dist > 0.2
        tree = KDTree(ds_xyz)
        dist, src_proj_idx = tree.query(xyz, return_distance=True)
        valid_idx = np.squeeze(dist <= 0.2)
        if mode == "training" or mode == "tra_val":
            xyz, labels, proj_idx, dist = pick_valid(xyz, labels, valid_idx, src_proj_idx, dist)
        else:
            xyz, labels, proj_idx, dist, src_idx = pick_valid(xyz, labels, valid_idx, src_proj_idx, dist, src_idx)
        cdist = 1 - (dist / 0.2)  # complementary distance. \in [0,1]. Same location = 1; dist=0.2 ==> 0

        if mode == "training" or mode == "tra_val":
            # if #src_points >= 5*num_points: remove extra points
            # else: duplicate src_points to 5*num_points
            num_in = sum(valid_idx)
            if num_in >= 5*num_points:
                pick_idx = np.random.choice(num_in, 5*num_points, replace=False)  # each element can be chosen only once
                xyz, labels, proj_idx, cdist = pick_valid(xyz, labels, pick_idx, proj_idx, cdist)
            else:
                idx_dup = np.random.choice(num_in, 5*num_points - num_in)
                idx_dup = list(range(num_in)) + list(idx_dup)
                xyz, labels, proj_idx, cdist = pick_valid(xyz, labels, idx_dup, proj_idx, cdist)
            return xyz, labels, proj_idx, cdist, src_idx
        else:
            return xyz, labels, proj_idx, cdist, src_idx

    def gen_point_idx_in_image(self, xyz):
        # 1
        # --2022/12/08: Finally, I understand why this is the correct one. --
        # --            A photo for understanding is put into Evernote under the name of 'Why swap x and y` --
        selected_pc_image_idx = xyz[:, :2] / 20 * 2 - 1  # scale x and y to [-1, 1]
        selected_pc_image_idx = selected_pc_image_idx[:, ::-1]  # swap x and y
        pts_img_idx = np.array([selected_pc_image_idx], dtype=np.float)
        return pts_img_idx

    def convert_label(self, label):
        temp = label.copy()
        for k, v in self.image_label_mapping.items():
            label[temp == k] = v
        return label

    def process_data(self, xyz, rgb, img, img_label, mask=None, src_xyz=None):
        xyz_mean = np.mean(xyz[:, :2], axis=0)
        if src_xyz is not None:
            src_xyz[:, :2] -= xyz_mean
            src_xyz[:, :3] /= 10
        xyz[:, :2] -= xyz_mean
        xyz[:, :3] /= 10

        rgb = rgb / 255.0
        rgb -= np.array(self.mean[:3])
        rgb /= np.array(self.std[:3])

        img = img.astype(np.float32)[:, :, ::-1]  # BGR to RGB
        img = img / 255.0
        img -= self.mean[:3]
        img /= self.std[:3]
        if mask is not None:
            # Set invalid pixel to zeros
            # mask = mask // 255  # binary mask
            mask = (np.repeat(np.expand_dims(mask, 2), repeats=3, axis=2)).astype(np.float32)  # 3-channel mask
            mask = mask.transpose((2, 0 ,1))  # H,W,CH to CH,H,W
            # img = img * mask_3channels
        img = img.transpose((2, 0, 1))  # H,W,CH to CH,H,W

        # In image label, 0 indicates ignored label
        # map 0 to -1, map x={1,2,..13} to {0,1,..12}
        if self.wImgL:
            img_label = self.convert_label(img_label)
            img_label = np.array(img_label).astype('int32')

        return xyz, rgb, img, img_label, mask, src_xyz

    def getitem(self, index):
        cloud_idx = self.samples[index]["file_idx"]
        _, xyz, labels, rgb, img, img_label, mask = self.get_data(cloud_idx)

        # crop a sub point set if necessary
        selected_idx = self.samples[index]["queried_point_idx"]  # selected_idx is shuffled in sample_points_cache()
        selected_labels = labels[selected_idx]
        xyz = xyz[selected_idx, :]
        rgb = rgb[selected_idx, :]
        tree = cKDTree(xyz)
        _, knns = tree.query(xyz, k=7)

        # crop a sub point set of source points
        # generate points' indices in image
        if self.src_pts_in_fusion_module:
            src_xyz, src_labels = self.load_points(cloud_idx, src_points=True)
            src_xyz, src_labels, proj_idx, cdist, src_idx\
                = self.crop_src_points(xyz, src_xyz, src_labels, self.num_points, self.mode)
            pts_img_idx = self.gen_point_idx_in_image(src_xyz)
        else:
            pts_img_idx = self.gen_point_idx_in_image(xyz)

        xyz, rgb, img, img_label, mask = self.process_data(xyz, rgb, img, img_label, mask)

        # ======================================================
        # Data augmentation
        xyz = self.subtract_city_wise_height(cloud_idx, xyz)
        rgb = self.point_color_dropping(rgb)
        rgb = self.point_color_auto_contrast(rgb)
        # ======================================================

        cloud_ind = self.samples[index]["file_idx"]

        if self.src_pts_in_fusion_module:
            return xyz, selected_labels, selected_idx, cloud_ind, rgb, img, img_label,\
                   pts_img_idx, knns, mask, src_labels, proj_idx, cdist, src_idx#, doubleh
        else:
            return xyz, selected_labels, selected_idx, cloud_ind, rgb, img, img_label, pts_img_idx, knns, mask# , doubleh

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx, batch_pc_rgb, pts_img_idx):#, doubleh):
        # features = batch_pc
        features = np.concatenate((batch_pc, batch_pc_rgb), axis=2)
        # features = np.concatenate((batch_pc, batch_pc_rgb, doubleh), axis=2)

        def foo(batch_pc, pts_img_idx, first_round=True):
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            input_pts_img_idx = []

            _batch_pc = batch_pc
            _pts_img_idx = pts_img_idx
            for i in range(self.num_layers):
                n_points = int(_batch_pc.shape[1] // self.sub_sampling_ratio[i])
                if first_round:
                    idx = range(n_points)
                else:
                    idx = list(range(_batch_pc.shape[1]))
                    np.random.shuffle(idx)
                    idx = idx[:n_points]

                neighbour_idx = DP.knn_search(_batch_pc, _batch_pc, self.k_n)
                sub_points = _batch_pc[:, idx, :]
                pool_i = neighbour_idx[:, idx, :]
                pts_img_i = _pts_img_idx[:, :, idx, :]
                up_i = DP.knn_search(sub_points, _batch_pc, 1)

                input_points.append(_batch_pc)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                input_pts_img_idx.append(pts_img_i)
                _batch_pc = sub_points
                _pts_img_idx = pts_img_i

            return input_points + input_neighbors + input_pools + input_up_samples + input_pts_img_idx

        input_list = foo(batch_pc, pts_img_idx, first_round=True)
        input_list += foo(batch_pc, pts_img_idx, first_round=False)
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):
        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        selected_fea = []
        pts_img_idx = []
        knns = []
        mask = []
        center_point = []
        if self.src_pts_in_fusion_module:
            src_xyz, src_labels = [], []
        imgs, img_labels = [], []
        for sample in batch:
            selected_pc.append(sample[0])
            selected_labels.append(sample[1])
            selected_idx.append(sample[2])
            cloud_ind.append(sample[3])
            selected_fea.append(sample[4])
            imgs.append(sample[5])
            img_labels.append(sample[6])
            pts_img_idx.append(sample[7])
            knns.append(sample[8])
            mask.append(sample[9])
            center_point.append(sample[10])
            if self.src_pts_in_fusion_module:
                src_xyz.append(sample[11])
                src_labels.append(sample[12])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        selected_fea = np.stack(selected_fea)
        pts_img_idx = np.stack(pts_img_idx)
        imgs = np.stack(imgs)
        img_labels = np.stack(img_labels)
        knns = np.stack(knns)
        mask = np.stack(mask)
        center_point = np.stack(center_point)
        if self.src_pts_in_fusion_module:
            src_xyz = np.stack(src_xyz)
            src_labels = np.stack(src_labels)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind,
                                  selected_fea, pts_img_idx)#,

        num_layers = self.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['xyz2'] = []
        for tmp in flat_inputs[5 * num_layers:6 * num_layers]:
            inputs['xyz2'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx2'] = []
        for tmp in flat_inputs[6 * num_layers: 7 * num_layers]:
            inputs['neigh_idx2'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx2'] = []
        for tmp in flat_inputs[7 * num_layers:8 * num_layers]:
            inputs['sub_idx2'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx2'] = []
        for tmp in flat_inputs[8 * num_layers:9 * num_layers]:
            inputs['interp_idx2'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[10 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[10 * num_layers + 1]).long()
        inputs["input_inds"] = flat_inputs[10 * num_layers + 2]
        inputs["cloud_inds"] = flat_inputs[10 * num_layers + 3]

        inputs['pts_img_idx0'] = torch.from_numpy(pts_img_idx).float()
        if self.wkNN:
            inputs['knns'] = torch.from_numpy(knns).long()
        if self.src_pts_in_fusion_module:
            inputs["src_xyz"] = torch.from_numpy(src_xyz).float()
            inputs["src_labels"] = torch.from_numpy(src_labels).long()
        inputs['imgs'] = torch.from_numpy(imgs).float()
        if self.wImgL:
            inputs['img_labels'] = torch.from_numpy(img_labels).long()
        inputs['mask'] = torch.from_numpy(mask).float()
        inputs['center_point'] = torch.from_numpy(center_point).float()

        return inputs

    # ======================================================
    # Data augmentation
    def augment_pc_basic(self, xyz):
        xyzT = xyz.T
        ##########
        # Rotation
        ##########
        R = self.create_rotation_matrix()
        xyzT = np.matmul(R, xyzT)

        ##########
        # Scale
        ##########
        s = self.create_scale_matrix()
        xyzT = s * xyzT

        ##########
        # Noise
        ##########
        noise = np.random.normal(0, self.augment_point_noise, xyzT.shape)
        noise = np.clip(noise, -self.augment_point_noise_max, self.augment_point_noise_max)
        # divide noise on x,y by 10 because they are normalized by dividing 10
        noise[:2] /= 10.
        noise[:2] = np.clip(noise[:2], -self.augment_point_noise_max/10., self.augment_point_noise_max/10.)
        #
        xyzT += noise

        return xyzT.T

    def create_rotation_matrix(self):
        if self.augment_rotation == "":
            R = np.eye(3)
        elif self.augment_rotation == "z":
            theta = np.random.random() * np.pi * 2.
            c, s = np.cos(theta), np.sin(theta)
            cs0 = 0.
            cs1 = 1.
            R = [[c, -s, cs0],
                 [s, c, cs0],
                 [cs0, cs0, cs1]]
            R = np.array(R)
        elif self.augment_rotation == "arbitrarily":
            # refer to https://github.com/yanx27/Urban3D-2021-2nd/blob/01778aee3d7e66b4975233bade39423069f0d4f5/datasets/custom_dataset.py#L71
            # for future implementation
            raise NotImplemented
        return R

    def create_scale_matrix(self):
        lo, hi = self.augment_scale
        if self.augment_scale_anisotropic:
            s = np.random.uniform(lo, hi, 3)
        else:
            s = np.random.uniform(lo, hi, 1)
        symmetries = []
        for i in range(3):
            if self.augment_symmetreis[i]:
                symmetries.append(np.round(np.random.random()) * 2 - 1)
            else:
                symmetries.append(1.0)
        s = np.expand_dims(s * np.array(symmetries), 1)
        return s

    def rotate_pc(self, xyz_list, R=None):
        ##########
        # Rotation
        ##########
        res = []
        for xyz in xyz_list:
            R = self.create_rotation_matrix() if R is None else R
            xyzT = np.matmul(R, xyz.T)
            res.append(xyzT.T)
        return res

    def scale_pc(self, xyz_list, s=None):
        ##########
        # Scale
        ##########
        res = []
        for xyz in xyz_list:
            s = self.create_scale_matrix() if s is None else s
            xyzT = s * xyz.T
            res.append(xyzT.T)
        return res


    def subtract_city_wise_height(self, cloud_idx, xyz):
        if self.city_wise_sub:
            if "birmingham" in self.files[cloud_idx]["name"]:
                xyz[:, 2] -= 9.25
            elif "cambridge" in self.files[cloud_idx]["name"]:
                xyz[:, 2] -= 35.51
            else:
                print("height error")

        return xyz

    def append_two_heights(self, cloud_idx, xyz):
        h1 = (xyz[:, 2] - min(xyz[:, 2]))  # XYZAlign from PointNext: xy -= mean(xy), h -= min(h)
        h1 = h1/100  # normalized height

        additional_heights = np.expand_dims(xyz[:, 2], 1)
        xyz[:, 2] = h1
        return xyz, additional_heights

    def point_color_dropping(self, rgb):
        # randomly select a portion of points and set (normalized) color of them as zeros
        if hasattr(self, 'color_dropping') and self.color_dropping > 0:
            idx = list(range(rgb.shape[0]))
            np.random.shuffle(idx)
            idx = idx[:int(np.float(rgb.shape[0]) * self.color_dropping)]

            rgb[idx, :] = 0

        return rgb

    def point_color_auto_contrast(self, rgb):
        # Default setting
        randomize_blend_factor = True
        blend_factor = 0.5

        if hasattr(self, 'color_auto_contrast') and self.color_auto_contrast > 0:
            idx_not_color_dropped_points = np.mean(rgb, axis=1) != 0
            idx_auto_contrast_points = np.random.random(rgb.shape[0]) < self.color_auto_contrast
            idx_target = idx_not_color_dropped_points & idx_auto_contrast_points

            lo = np.min(rgb[idx_target, :], axis=1, keepdims=True)
            hi = np.max(rgb[idx_target, :], axis=1, keepdims=True)
            scale = 1 / (hi - lo)
            contrast_rgb = (rgb[idx_target, :] - lo) * scale

            blend_factor = random.random() if randomize_blend_factor else blend_factor
            rgb[idx_target] = (1 - blend_factor) * rgb[idx_target] + blend_factor * contrast_rgb

        return rgb

    def random_brightness(self, img):
        if not config.TRAIN.RANDOM_BRIGHTNESS:
            return img
        if random.random() < 0.5:
            return img
        self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

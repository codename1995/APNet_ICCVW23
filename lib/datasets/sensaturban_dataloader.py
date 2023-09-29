import copy

import numpy as np
import threading

from scipy.spatial.ckdtree import cKDTree

from lib.datasets.base_dataset_sensat import SensatBaseDataset


class SensatTrainingSet(SensatBaseDataset):
    def __init__(self,
                 mode,
                 cfg_dataset,
                 cfg_model,
                 cfg_trick,
                 data_list_path,
                 ignore_label=255,
                 # Dataset-specific
                 visualize=False,
                 debug=False,
                 ):
        super(SensatTrainingSet, self).__init__(
            mode=mode,
            cfg_dataset=cfg_dataset,
            cfg_model=cfg_model,
            cfg_trick=cfg_trick,
            data_list_path=data_list_path,
            ignore_label=ignore_label,
            visualize = visualize,
            debug=debug,
        )

        # Obtain values by conducting function
        self.lock = threading.Lock()

    def crop_src_pts(self, xyz, src_pts):
        sub_mean = np.mean(xyz, axis=0)
        max_dist = np.max(np.sum(np.square(xyz - sub_mean), axis=1))

        # First-time roughly delete src_pts
        dist = np.sum(np.square(src_pts[:, :3] - sub_mean), axis=1)
        valid_idx = np.where(dist <= (max_dist + 0.2))[0]
        n_valid = len(valid_idx)
        if n_valid > 10 * self.num_points:
            valid_idx = np.random.choice(valid_idx, 10 * self.num_points, replace=False)
        src_pts = src_pts[valid_idx, :]

        # Second-time carefully cut by kd-tree
        tree = cKDTree(xyz)
        d, knns = tree.query(src_pts[:, :3], k=1)
        valid_idx = np.where(d <= 0.2)[0]
        n_valid = len(valid_idx)
        if n_valid > 3 * self.num_points:
            valid_idx = np.random.choice(valid_idx, 3 * self.num_points, replace=False)
        else:
            # print("src_pts <= 3 * grid pts")
            idx_dup = np.random.choice(valid_idx, 3 * self.num_points - n_valid)
            valid_idx = list(valid_idx) + list(idx_dup)

        return src_pts[valid_idx, :]

    def __getitem__(self, item):
        self.lock.acquire()
        center_point, cloud_idx, selected_idx = self.find_query_point_and_its_nns()
        cp = copy.deepcopy(center_point)
        self.lock.release()

        # obtain selected xyz, rgb, labels and the src_pts in patch
        xyz, rgb, labels, src_pts = self.get_data(cloud_idx, selected_idx)

        # point augmentation
        if self.mode == 'training':
            xyz, src_pts[:, :3], center_point = self.rotate_pc([xyz, src_pts[:, :3], center_point])
            xyz, src_pts[:, :3], center_point = self.scale_pc([xyz, src_pts[:, :3], center_point])


        xyz, rgb, labels, img, img_label, mask, img_cnt, pts_img_idx, cp_img_idx\
            = self.project2BEV(src_pts, xyz, rgb, labels, self.pixel_size, center_point)
        # crop a sub point set if necessary
        if self.wkNN and not self.src_pts_in_fusion_module:
            tree = cKDTree(xyz)
            _, knns = tree.query(xyz, k=7)
        elif self.wkNN and self.src_pts_in_fusion_module:
            src_pts = self.crop_src_pts(xyz, src_pts)
            tree = cKDTree(xyz)
            _, knns = tree.query(src_pts[:, :3], k=7)
        else:
            knns = None

        # crop a sub point set of source points
        # generate points' indices in image
        if self.src_pts_in_fusion_module:
            src_xyz = src_pts[:, :3]
            src_labels = src_pts[:, 6]
            xyz, rgb, img, img_label, mask, src_xyz = self.process_data(xyz, rgb, img, img_label, mask, src_xyz)
        else:
            xyz, rgb, img, img_label, mask, _ = self.process_data(xyz, rgb, img, img_label, mask)

        cloud_ind = np.array([cloud_idx], dtype=np.int32)
        if self.src_pts_in_fusion_module:
            return xyz, labels, selected_idx, cloud_ind, rgb, img, img_label, \
                   pts_img_idx, knns, mask, cp, src_xyz, src_labels   # , doubleh
        else:
            return xyz, labels, selected_idx, cloud_ind, rgb, img, img_label,\
                   pts_img_idx, knns, mask, cp# , doubleh

    def find_query_point_and_its_nns(self):
        # find query point and its nearest neighbours
        # query point is the point with minimum possibility
        cloud_idx = int(np.argmin(self.min_possibility))
        center_point_ind = np.argmin(self.possibility[cloud_idx])
        points = np.array(self.trees[cloud_idx].data, copy=False)
        center_point = points[center_point_ind, :].reshape(1, -1)

        # Search the k Nearest Neighbours
        queried_idx = self.trees[cloud_idx].query(center_point, k=self.num_points)[1][0]

        # Update possibility
        dists = np.sum(np.square((points[queried_idx] - center_point).astype(np.float32)), axis=1)
        self.possibility[cloud_idx][queried_idx] += np.square(1 - dists / np.max(dists))
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        return center_point, cloud_idx, queried_idx

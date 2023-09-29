import os
import pickle
import shutil

import numpy as np
import threading

import torch
from sklearn.metrics import confusion_matrix

import lib.utils.distributed as dist



def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


class IoUCalculator:
    def __init__(self, num_classes):
        self.gt_classes = [0 for _ in range(num_classes)]
        self.positive_classes = [0 for _ in range(num_classes)]
        self.true_positive_classes = [0 for _ in range(num_classes)]
        self.num_classes = num_classes
        self.lock = threading.Lock()

    def add_data(self, logits, labels):
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels = np.arange(0, self.num_classes, 1))

        # if dist.is_distributed():
        #     conf_matrix = torch.from_numpy(conf_matrix).cuda()
        #     reduced_conf_matrix = reduce_tensor(conf_matrix)
        #     conf_matrix = reduced_conf_matrix.cpu().numpy()

        self.lock.acquire()
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)
        self.lock.release()

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / \
                    float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        return mean_iou, iou_list


def merge_results_dist(iou_calc, output_dir, num_classes=13):
    # Reference: https://github.com/open-mmlab/OpenPCDet/blob/e948c537858d4e468a9ef50b023324f6bc96db2a/pcdet/utils/common_utils.py#L211
    #
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tmpdir = os.path.join(output_dir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    res = [iou_calc.gt_classes, iou_calc.positive_classes, iou_calc.true_positive_classes]
    pickle.dump(res, open((os.path.join(tmpdir, 'res_{}.pkl'.format(rank))), 'wb'))
    dist.barrier()

    if rank != 0:
        return 0, [], 0.0

    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'res_{}.pkl'.format(i))
        res = pickle.load(open(part_file, 'rb'))
        if i == 0:
            gt_classes = res[0]
            positive_classes = res[1]
            true_positive_classes = res[2]
        else:
            gt_classes += res[0]
            positive_classes += res[1]
            true_positive_classes += res[2]

    shutil.rmtree(tmpdir)
    iou_list = []
    for n in range(0, num_classes, 1):
        if float(gt_classes[n] + positive_classes[n] - true_positive_classes[n]) != 0:
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        else:
            iou_list.append(0.0)
    mean_iou = sum(iou_list) / float(num_classes)
    OA = float(sum(true_positive_classes)) / sum(gt_classes)
    return mean_iou, iou_list, OA





def compute_acc(logits, labels):
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    return acc

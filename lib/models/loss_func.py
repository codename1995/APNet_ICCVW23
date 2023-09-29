import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        assert len(ignore_label) == 1
        self.ignore_label = ignore_label[0]
        self.weight = weight
        self.epsilon = epsilon
        self.reduction = reduction


    def forward(self, preds, target):
        if self.weight is not None:
            self.weight.to(preds.device)

        n = preds.size()[-1]  #
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), 'none')
        nll = F.nll_loss(log_preds, target, weight=self.weight,
                         ignore_index=self.ignore_label, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def compute_kl_loss(p, q, class_weight=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if class_weight is not None:
        bs = p.shape[0]
        n_points = p.shape[2]
        class_weight = torch.from_numpy(class_weight).float().cuda()
        class_weight = class_weight.unsqueeze(2)
        class_weight = class_weight.repeat((bs, 1, n_points))

        p_loss *= class_weight
        q_loss *= class_weight

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def interpolate_img_pred(img_pred, img_label):
    _, _, ph, pw = img_pred.shape
    _, h, w = img_label.shape
    if ph != h or pw != w:
        img_pred = F.interpolate(input=img_pred, size=(h, w), mode='bilinear', align_corners=False)

    return img_pred


def compute_img_loss(img_pred, img_label, criterion, num_classes):
    img_pred = interpolate_img_pred(img_pred, img_label)
    img_pred = reshape_logits(img_pred, num_classes=num_classes)
    img_label = reshape_labels(img_label)
    loss = criterion(img_pred, img_label).mean()
    return loss, img_pred, img_label


def compute_lovasz(outputs, lovasz_criterion, src_pts=False):
    preds = outputs['logits']
    if src_pts:
        labels = outputs['src_labels']  # b_s x (8192*5)
    else:
        labels = outputs['labels']  # b_s x 8192
    preds = preds.unsqueeze(3)  #  b_s x 13 x 8192 x 1
    lovasz_labels = labels.unsqueeze(2)  # b_s x 8192 x 1
    lovasz_loss = lovasz_criterion(preds, lovasz_labels) if lovasz_criterion else 0
    return lovasz_loss

def reshape_logits(array, num_classes=13):
    # The shapes of logits are b_s x 13 x 8192 or b_s x 13 x W x H
    # reshape to (b_s x N) x 13, where N = 8192 or WxH
    n_dim = len(array.shape)
    if n_dim == 3:
        array = array.transpose(1, 2).reshape(-1, num_classes)
    else:
        array = array.transpose(1, 2).transpose(2, 3).reshape(-1, num_classes)
    return array


def reshape_labels(array):
    # Two- or three-dimension array to one-dimension array
    # b_s x 8192 ==> (b_s x 8192)
    # or
    # b_s x W x H ==> (b_s x W x H)
    return array.reshape(-1)


def compute_loss(outputs, criterion, multiloss=False, src_pts=False, num_classes=13):
    ####################################
    # V1: src_pts = False
    # pred:     a,      p,      fused
    # label:    labels, labels, labels
    ####################################
    # V2: src_pts = True
    # pred:     a,          p,      fused
    # label:    src_labels, labels, src_labels
    ####################################
    # V3: src_pts = False, remove 2D seg head
    # pred:     a,          p,      fused
    # label:    img_labels, labels, src_labels
    ####################################
    # init results
    loss, logits, labels = {}, {}, {}

    logits['fused'] = reshape_logits(outputs['logits'], num_classes)
    if src_pts:
        labels['fused'] = reshape_labels(outputs['src_labels'])
    else:
        labels['fused'] = reshape_labels(outputs['labels'])
    loss['fused'] = criterion(logits['fused'], labels['fused']).mean()

    if multiloss:
        logits['p'] = reshape_logits(outputs['p_out'], num_classes)
        labels['p'] = reshape_labels(outputs['labels'])
        loss['p'] = criterion(logits['p'], labels['p']).mean()

        loss['a'], logits['a'], labels['a'] = compute_img_loss(
            outputs['a_out'],
            outputs['img_labels'],
            criterion,
            num_classes
        )



    # score1 = outputs['score1']
    # score2 = outputs['score2']
    # loss3 = kl_weight*compute_kl_loss(score1, score2, class_weight=dataset.get_class_weights())

    return loss, logits, labels

class Unittest_dataset():
    def __init__(self):
        self.num_classes = 13
        self.ignored_labels = np.sort([-1])


    def get_class_weights(self, weights_type="ce"):
        if weights_type == "ce":
            num_per_class = np.array([439244099, 544284030, 861977674, 20606217,
                                      3044907, 49452238, 498977, 131470199,
                                      26669467, 37557130, 43733933, 174125,
                                      7088841])
            weights = num_per_class / float(sum(num_per_class))
            weights = 1 / (weights + 0.02)
            weights = np.expand_dims(weights, axis=0)
        return weights


if __name__ == '__main__':
    # Test 2 for reshape operation
    # test2()

    # Test 1
    dataset = Unittest_dataset()

    labels = np.random.randint(0, 13, (16, 8192), dtype=np.int8)
    logits = np.random.random((16, 13, 8192))
    outputs = {}
    outputs['labels'] = torch.Tensor(labels)
    outputs['logits'] = torch.Tensor(logits)

    from lib.models.lovasz_softmax import Lovasz_softmax
    lovasz_criterion = Lovasz_softmax(ignore=dataset.ignored_labels)
    import torch.nn as nn
    class_weights = torch.from_numpy(dataset.get_class_weights()).float().cuda()
    point_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='none')
    compute_loss(outputs, dataset, point_criterion, True, lovasz_criterion=lovasz_criterion)

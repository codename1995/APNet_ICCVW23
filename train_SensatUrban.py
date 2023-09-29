# Common
import os
import logging
import time
import warnings
import argparse
import pprint
import numpy as np
from tqdm import tqdm
import random
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from lib.datasets.sensaturban_dataloader import SensatTrainingSet
from lib.datasets.base_dataset_sensat import worker_init_fn
from lib.utils.metric import compute_acc, IoUCalculator, merge_results_dist
from lib.models.app_fusion_net import APNet
from lib.models.loss_func import compute_loss, compute_img_loss, compute_lovasz
# HRNet
from lib.config import config as cfg
from lib.config import update_config
from lib.models.loss_func import LabelSmoothingCrossEntropy
from lib.utils.utils import create_logger, FullModel, AverageMeter, log_IoU
import lib.utils.distributed as dist
# Loss
from lib.models.lovasz_softmax import Lovasz_softmax

os.environ["NCCL_DEBUG"] = "INFO"
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'


def parse_args():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        default="experiments/local_sanity_check.yaml",
                        help='experiment configure file name',
                        # required=True,
                        type=str)
    parser.add_argument('--debug', action='store_true')  # default=False
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(cfg, args)

    return args


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


def train_one_epoch(PRT_FREQ, epoch, num_epoch, epoch_iters,
                    train_loader, optimizer, net, train_dataset, criterion, scheduler,
                    writer_dict, loss_factors):
    net.train()  # set model to training mode

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    AM_loss_apnet = AverageMeter()
    AM_loss_image = AverageMeter()
    AM_loss_AD = AverageMeter()
    AM_loss_PD = AverageMeter()
    AM_loss_lovasz = AverageMeter()
    tic = time.time()

    if dist.get_rank() == 0:
        train_loader = tqdm(train_loader, total=len(train_loader))
    for batch_idx, batch_data in enumerate(train_loader):
        for key in batch_data:
            if key == "input_inds" or key == "cloud_inds":
                continue
            elif type(batch_data[key]) is list:
                for i in range(cfg.MODEL.NUM_LAYERS):
                    batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
            else:
                batch_data[key] = batch_data[key].cuda(non_blocking=True)

        optimizer.zero_grad()
        # Forward pass
        torch.cuda.synchronize()
        outputs = net(batch_data)
        # loss, _ = compute_loss(outputs, train_dataset, criterion)
        res_loss, _, _ = compute_loss(outputs, criterion[0], multiloss=True, src_pts=cfg.MODEL.SRC_PTS,
                                      num_classes=train_dataset.num_classes)
        loss_image, _, _ = compute_img_loss(
            outputs['ocr_out'], outputs['img_labels'], criterion[0], train_dataset.num_classes)
        loss_lovasz = compute_lovasz(outputs, criterion[1], src_pts=cfg.MODEL.SRC_PTS)
        # loss_image = 0
        loss = loss_factors[0] * res_loss['fused'] + \
               loss_factors[1] * loss_image + \
               loss_factors[2] * res_loss['a'] + \
               loss_factors[3] * res_loss['p'] + \
               loss_factors[4] * loss_lovasz

        if dist.is_distributed():
            loss = reduce_tensor(loss)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        AM_loss_apnet.update(res_loss['fused'].item())
        AM_loss_image.update(loss_image.item())
        AM_loss_AD.update(res_loss['a'].item())
        AM_loss_PD.update(res_loss['p'].item())
        AM_loss_lovasz.update(loss_lovasz.item())

        if (batch_idx + 1) % cfg.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}'.format(
                epoch, num_epoch, batch_idx, epoch_iters,
                batch_time.value(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

        # if (batch_idx + 1) * cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS) >= cfg.TRAIN.SAMPLES_PER_EPOCH:
        #     break

    if dist.get_rank() <= 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', ave_loss.average(), global_steps)
        writer.add_scalar('train_loss_apnet', AM_loss_apnet.average(), global_steps)
        writer.add_scalar('train_loss_image', AM_loss_image.average(), global_steps)
        writer.add_scalar('train_loss_AD', AM_loss_AD.average(), global_steps)
        writer.add_scalar('train_loss_PD', AM_loss_PD.average(), global_steps)
        writer.add_scalar('train_loss_Lovasz', AM_loss_lovasz.average(), global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    scheduler.step()


def validate(val_loader, net, logger, val_dataset, criterion, writer_dict, loss_factors, output_dir):
    net.eval()  # set model to eval mode (for bn and dp)
    iou_calc = IoUCalculator(cfg.DATASET.NUM_CLASSES)
    iou_calc_P = IoUCalculator(cfg.DATASET.NUM_CLASSES)
    iou_calc_A = IoUCalculator(cfg.DATASET.NUM_CLASSES)
    ave_loss = AverageMeter()
    AM_loss_apnet = AverageMeter()
    AM_loss_image = AverageMeter()
    AM_loss_AD = AverageMeter()
    AM_loss_PD = AverageMeter()
    AM_loss_lovasz = AverageMeter()

    if dist.get_rank() == 0:
        val_loader = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            for key in batch_data:
                if key == "input_inds" or key == "cloud_inds":
                    continue
                elif type(batch_data[key]) is list:
                    for i in range(cfg.MODEL.NUM_LAYERS):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            # Forward pass
            torch.cuda.synchronize()
            outputs = net(batch_data)

            # loss, outputs = compute_loss(outputs, val_dataset, criterion)
            # loss1, loss2, loss3, outputs = compute_loss(outputs, val_dataset, criterion)
            res_loss, logits, labels = compute_loss(
                outputs, criterion[0], multiloss=True, src_pts=cfg.MODEL.SRC_PTS,
                num_classes=val_dataset.num_classes)
            loss_image, _, _ = compute_img_loss(
                outputs['ocr_out'], outputs['img_labels'], criterion[0], val_dataset.num_classes)
            loss_lovasz = compute_lovasz(outputs, criterion[1], src_pts=cfg.MODEL.SRC_PTS)
            # loss_image = 0

            loss = loss_factors[0] * res_loss['fused'] + \
                   loss_factors[1] * loss_image + \
                   loss_factors[2] * res_loss['a'] + \
                   loss_factors[3] * res_loss['p'] + \
                   loss_factors[4] * loss_lovasz

            if dist.is_distributed():
                loss = reduce_tensor(loss)

            acc = compute_acc(logits['fused'], labels['fused'])
            iou_calc.add_data(logits['fused'], labels['fused'])
            iou_calc_A.add_data(logits['a'], labels['a'])
            iou_calc_P.add_data(logits['p'], labels['p'])

            # update average loss
            ave_loss.update(loss.item())
            AM_loss_apnet.update(res_loss['fused'].item())
            AM_loss_image.update(loss_image.item())
            AM_loss_AD.update(res_loss['a'].item())
            AM_loss_PD.update(res_loss['p'].item())
            AM_loss_lovasz.update(loss_lovasz.item())

    if dist.is_distributed():
        mean_iou, iou_list = merge_results_dist(iou_calc, output_dir, val_dataset.num_classes)
        mean_iou_A, iou_list_A = merge_results_dist(iou_calc_A, output_dir, val_dataset.num_classes)
        mean_iou_P, iou_list_P = merge_results_dist(iou_calc_P, output_dir, val_dataset.num_classes)
    else:
        mean_iou, iou_list = iou_calc.compute_iou()
        mean_iou_A, iou_list_A = iou_calc_A.compute_iou()
        mean_iou_P, iou_list_P = iou_calc_P.compute_iou()

    if dist.get_rank() <= 0:
        log_IoU(logger, mean_iou, iou_list)
        log_IoU(logger, mean_iou_A, iou_list_A)
        log_IoU(logger, mean_iou_P, iou_list_P)

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer_dict['max_val_mIoU'] = max(writer_dict['max_val_mIoU'], mean_iou)
        writer.add_scalar('max_valid_mIoU', writer_dict['max_val_mIoU'], global_steps)
        writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
        writer.add_scalar('val_loss_apnet', AM_loss_apnet.average(), global_steps)
        writer.add_scalar('val_loss_image', AM_loss_image.average(), global_steps)
        writer.add_scalar('val_loss_AD', AM_loss_AD.average(), global_steps)
        writer.add_scalar('val_loss_PD', AM_loss_PD.average(), global_steps)
        writer.add_scalar('val_loss_Lovasz', AM_loss_lovasz.average(), global_steps)
        writer.add_scalar('valid_mIoU', mean_iou, global_steps)
        writer.add_scalar('valid_mIoU_A', mean_iou_A, global_steps)
        writer.add_scalar('valid_mIoU_P', mean_iou_P, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 4

    return mean_iou, ave_loss.average(), iou_list


def get_sampler(dataset):
    from lib.utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def loss_factors_scheduler(n_epoch, strategy=1):
    # Generate factors for loss functions in different epoches.
    # Loss functions are in this:
    #   1. APNet (A+P -> F -> D): WCE loss
    #   2. The combined A-branch loss = 0.04 * OCR Loss + 0.1 * 2D SemSeg Loss
    #   3. A-branch -> Decoder: WCE loss
    #   4. P-branch -> Decoder: WCE loss
    #   5. APNet (A+P -> F -> D): Lovasz softmax loss
    f1 = [0.1] * 30 + [1] * (n_epoch-30)
    f2 = [1] * 30 + [0.1] * (n_epoch-30)
    f3 = [1] * 30 + [0.1] * (n_epoch-30)
    f4 = [0.5] * 30 + [0.05] * (n_epoch-30)
    f5 = [0.1] * 30 + [1] * (n_epoch-30)
    loss_factors = [x for x in zip(f1, f2, f3, f4, f5)]

    return loss_factors


def main():
    args = parse_args()

    if args.seed > 0:
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 4,
        'max_val_mIoU': 0,
    }

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    gpus = list(cfg.GPUS)
    distributed = args.local_rank >= 0
    # distributed = False
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    if distributed:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # get_dataset & dataloader
    train_dataset = SensatTrainingSet(
        mode='training',
        cfg_dataset=cfg.DATASET,
        cfg_model=cfg.MODEL,
        cfg_trick=cfg.TRICK,
        data_list_path=cfg.DATASET.TRAIN_SET,
        visualize=cfg.DATASET.VISUALIZE_EACH_SAMPLE,
        debug=args.debug,
    )
    val_dataset=SensatTrainingSet(
        mode='tra_val',
        cfg_dataset=cfg.DATASET,
        cfg_model=cfg.MODEL,
        cfg_trick=cfg.TRICK,
        data_list_path=cfg.DATASET.TEST_SET,
        visualize=cfg.DATASET.VISUALIZE_EACH_SAMPLE,
        debug=args.debug,
    )


    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=True,
    )
    val_sampler = get_sampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler=val_sampler,
        worker_init_fn=worker_init_fn,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=True,
    )

    # Loss Function
    class_weights = torch.from_numpy(train_dataset.get_class_weights()).float().cuda()
    ce_loss = LabelSmoothingCrossEntropy(
        ignore_label=train_dataset.ignored_labels,
        weight=class_weights,
        reduction='mean',
    )
    # Lovasz softmax
    lovasz_softmax = Lovasz_softmax(ignore=train_dataset.ignored_labels)
    criterion = [ce_loss, lovasz_softmax]

    # Loss factor scheduler
    loss_factors = loss_factors_scheduler(cfg.TRAIN.END_EPOCH, 5)

    # Network & Optimizer
    net = APNet(cfg)

    # Load the Adam optimizer
    P_Encoder1 = list(map(id, net.fc0.parameters()))
    P_Encoder2 = list(map(id, net.dilated_res_blocks.parameters()))
    P_Encoder3 = list(map(id, net.decoder_0.parameters()))
    P_Encoder4 = list(map(id, net.decoder_blocks.parameters()))
    base_params = filter(lambda p: id(p) not in P_Encoder1 + P_Encoder2 + P_Encoder3 + P_Encoder4, net.parameters())
    optimizer = optim.AdamW(
        [
            {'params': base_params},
            {'params': net.fc0.parameters(), 'lr': cfg.TRAIN.LR * 5},
            {'params': net.dilated_res_blocks.parameters(), 'lr': cfg.TRAIN.LR * 5},
            {'params': net.decoder_0.parameters(), 'lr': cfg.TRAIN.LR * 5},
            {'params': net.decoder_blocks.parameters(), 'lr': cfg.TRAIN.LR * 5},
        ],
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    if distributed:
        net = net.to(device)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        net = nn.DataParallel(net, device_ids=gpus).cuda()


    # Load module
    best_mIoU = 0
    last_epoch = 0
    if cfg.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})

            last_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            net.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

            writer_dict['train_global_steps'] = last_epoch
            writer_dict['valid_global_steps'] = last_epoch
            writer_dict['max_val_mIoU'] = best_mIoU

        if distributed:
            torch.distributed.barrier()

    start = time.time()
    for epoch in range(last_epoch, cfg.TRAIN.END_EPOCH):
        cur_epoch = epoch
        if args.local_rank <= 0:
            logger.info('**** EPOCH %03d ****' % (epoch))

        if train_loader is not None and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            cfg.PRINT_FREQ, epoch, cfg.TRAIN.END_EPOCH, cfg.TRAIN.EPOCH_ITERS,
            train_loader, optimizer, net, train_dataset,
            criterion, scheduler, writer_dict, loss_factors[epoch]
        )

        if args.local_rank <= 0:
            logger.info('**** EVAL EPOCH %03d ****' % (epoch))

        if (epoch+1)%4 == 0:
            mean_iou, valid_loss, IoU_array = validate(val_loader, net, logger, val_dataset, criterion, writer_dict,
                                                       loss_factors[epoch], final_output_dir)

            if args.local_rank <= 0:
                if cfg.SAVE_WEIGHTS:
                    logger.info('=> saving checkpoint to {}'.format(
                        final_output_dir + '/checkpoint.pth.tar'))

                    # Save checkpoint
                    torch.save({
                        'epoch': cur_epoch + 1,
                        'best_mIoU': best_mIoU,
                        'state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
                # Save best checkpoint
                if mean_iou > best_mIoU:
                    best_mIoU = mean_iou
                    if cfg.SAVE_WEIGHTS:
                        torch.save(net.module.state_dict(), os.path.join(final_output_dir, 'best.pth'))
                msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(valid_loss, mean_iou, best_mIoU)
                logging.info(msg)
                logging.info(IoU_array)

    if args.local_rank <= 0:
        # Save final checkpoint
        if cfg.SAVE_WEIGHTS:
            torch.save(net.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = time.time()
        logger.info('%d Hours %d Minutes' % (np.int((end - start) / 3600), np.int((end - start) % 60)))
        logger.info('Done')


if __name__ == '__main__':
    main()

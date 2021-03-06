# coding=utf-8
import datetime
import os
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from thirdpart.data.dataset import CocoDataset
from thirdpart.optim import create_optimizer
from thirdpart.scheduler import create_scheduler
from thirdpart.data.augmentation import get_augmentation, detection_collate, collater, Augmenter, Resizer, Normalizer
from backbone_multi import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from efficientdet.ghm_loss import GHMLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, PrefetchLoader
from efficientdet.utils import Anchors


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file, encoding='utf-8').read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='wheat', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument("--image_size", type=int, default=None, help="The common width and height for all images")
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    # Optimizer parameters
    parser.add_argument('--opt', default='lookahead_radam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=50, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--data_path', type=str, default='/home/data/detection',
                        help='the root folder of dataset')
    date = 'd1-0525-mul-wheat'
    parser.add_argument('--log_path', type=str, default='logs/%s' % date)
    det_path = r'/home/data/weights/efficientdet/efficientdet-d1-2.pth'
    # log_path = r'outputs/d1-0512/d1_0.4956_0.0709_0.4247_63.pth'
    parser.add_argument('--load_weights', type=str, default=det_path,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='outputs/%s' % date)
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of training, '
                                                                  'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        # self.criterion = GHMLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, anchors, obj_list=None):
        regression, classification = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def create_loader(opt, params, image_size):
    trans = 'albu'  # raw albu
    if trans == 'albu':
        collate = detection_collate
        train_trans = get_augmentation(phase='train', width=image_size, height=image_size)
        valid_trans = get_augmentation(phase='valid', width=image_size, height=image_size)
    else:
        collate = collater
        train_trans = transforms.Compose([Normalizer(), Augmenter(), Resizer(common_size=image_size)])
        valid_trans = transforms.Compose([Normalizer(), Resizer(common_size=image_size)])

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collate,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': False,
                  'collate_fn': collate,
                  'num_workers': opt.num_workers}

    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                               set=params.train_set, resized_size=image_size,
                               transform=train_trans)
    training_generator = DataLoader(training_set, **training_params)
    # training_generator = PrefetchLoader(training_generator)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                          set=params.val_set, resized_size=image_size,
                          transform=valid_trans)
    val_generator = DataLoader(val_set, **val_params)
    # val_generator = PrefetchLoader(val_generator)

    return training_generator, val_generator


def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    created_time = f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    saved_path = opt.saved_path + created_time
    os.makedirs(opt.log_path + created_time, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        weights_path = opt.load_weights
        last_step = 0

        try:
            model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + created_time)

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    optimizer = create_optimizer(opt, model)
    scheduler, num_epochs = create_scheduler(opt, optimizer)
    if scheduler is not None and opt.start_epoch > 0:
        scheduler.step(opt.start_epoch)

    best_valid_loss = 1e5
    best_train_loss = 1e5
    step = 0
    model.train()

    input_sizes = [640, 768, 896, 1024]
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]

    for epoch in range(opt.num_epochs):
        size = np.random.choice(input_sizes)
        print('---------------------input size %d----------------------------' % size)

        training_generator, val_generator = create_loader(opt, params, size)

        anchors = Anchors(anchor_scale=anchor_scale[opt.compound_coef], image_size=size)

        model.train()

        num_iter_per_epoch = len(training_generator)

        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue

        epoch_loss = []
        progress_bar = tqdm(training_generator)
        num_updates = epoch * len(training_generator)
        for iter, data in enumerate(progress_bar):
            if iter < step - last_epoch * num_iter_per_epoch:
                progress_bar.update()
                continue
            imgs = data['img'].float()
            annot = data['annot']

            if params.num_gpus == 1:
                # if only one gpu, just send it to cuda:0
                # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                imgs = imgs.cuda()
                annot = annot.cuda()

            optimizer.zero_grad()
            cls_loss, reg_loss = model(imgs, annot, anchors(imgs), obj_list=params.obj_list)

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()

            loss = cls_loss + reg_loss
            if loss == 0 or not torch.isfinite(loss):
                continue

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            num_updates += 1

            epoch_loss.append(float(loss))
            mean_epoch_loss = np.mean(epoch_loss)

            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if scheduler is not None:
                scheduler.step_update(num_updates=num_updates, metric=mean_epoch_loss)

            progress_bar.set_description(
                'Ep: {}/{} Iter: {}/{} Cls: {:.4f} Reg: {:.4f} Total: {:.4f}'.format(
                    epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                    reg_loss.item(), mean_epoch_loss.item()))
            writer.add_scalar('Train/Total_loss', mean_epoch_loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss,
                              epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Learning rate/LR', lr, epoch)

        mean_epoch_loss = np.mean(epoch_loss)
        if mean_epoch_loss + opt.es_min_delta < best_train_loss:
            best_train_loss = mean_epoch_loss
            save_checkpoint(model, saved_path, 'Best_train_model.pth')

        if epoch % opt.val_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(val_generator):
                with torch.no_grad():
                    imgs = data['img'].float()
                    annot = data['annot']

                    if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    cls_loss, reg_loss = model(imgs, annot, anchors(imgs), obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss_classification_ls.append(cls_loss.item())
                    loss_regression_ls.append(reg_loss.item())

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Val. Epoch: {}/{}. Cla loss: {:1.4f}. Reg loss: {:1.4f}. Total: {:1.4f}'.format(
                    epoch, opt.num_epochs, cls_loss, reg_loss, np.mean(loss)))
            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_valid_loss:
                best_valid_loss = loss
                model_name = "d%d_%.4f_%.4f_%.4f_%d.pth" % (
                    opt.compound_coef, loss, reg_loss.mean(), cls_loss.mean(), epoch)
                save_checkpoint(model, saved_path, model_name)

            if scheduler is not None:
                # step LR for next epoch
                scheduler.step(epoch + 1, np.mean(loss))

    writer.close()


def save_checkpoint(model, saved_path, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)

import argparse
import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models import vision_transformer as vits
from models import vision_transformer2 as vits2


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr * (args.batch_size/128), momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * (args.batch_size/128) * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # inductive
    best_test_acc_lab = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - start

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out, student_ood, pseudo_out = student(images)
                teacher_out = student_out.detach()

                pstr = ''
                loss = 0
                # ---------------------------------- SimGCD ----------------------------------
                # supervised GCD loss
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                loss += args.sup_weight * cls_loss
                pstr += f'cls_loss: {cls_loss.item():.4f} '

                # unsupervised GCD loss
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                loss += (1 - args.sup_weight) * cluster_loss
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                # ---------------------------------- Semantic Distribution Learning ----------------------------------
                # reference: https://github.com/VisionLearningGroup/OP_Match
                student_ood = student_ood / 0.1
                logits_ood = torch.cat([f[mask_lab] for f in student_ood.chunk(2)], dim=0)
                logits_ood_u = torch.cat([f[~mask_lab] for f in student_ood.chunk(2)], dim=0)
                logits_open_u1, logits_open_u2 = logits_ood_u.chunk(2)
                # SDL Loss for labeled samples
                sup_sdl_loss = ova_loss(logits_ood, sup_labels)
                pstr += f'sup_sdl_loss: {sup_sdl_loss.item():.4f} '

                # SDL Loss for unlabeled samples
                # entropy minimization
                L_oem = ova_ent(logits_open_u1) / 2.
                L_oem += ova_ent(logits_open_u2) / 2.
                # Soft consistency regularization
                logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
                logits_open_u1 = F.softmax(logits_open_u1, 1)
                logits_open_u2 = F.softmax(logits_open_u2, 1)
                unsup_sdl_loss = 0.1 * L_oem + torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2) ** 2, 1), 1))
                pstr += f'unsup_sdl_loss: {unsup_sdl_loss.item():.4f} '
                loss += args.sdl_loss_weight * (sup_sdl_loss + unsup_sdl_loss)

                # distribution certainty score from OVA classifier
                ova_scores = F.softmax(student_ood.view(student_ood.size(0), 2, -1), 1).detach()
                pred_close = ova_scores[:, 1, :].data.max(1)[1]
                tmp_range = torch.arange(0, ova_scores.size(0)).long().cuda()
                unk_score = ova_scores[tmp_range, 0, pred_close]
                unk_score_unlabelled = torch.cat([f[~mask_lab] for f in unk_score.chunk(2)], dim=0)
                ood_cer_score = torch.abs(2*unk_score_unlabelled - 1)

                # ---------------------------------- Auxiliary Debiased Learning ----------------------------------
                sup_logits_pseudo = torch.cat([f[mask_lab] for f in (pseudo_out / args.pseudo_temp).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                sup_adl_loss = nn.CrossEntropyLoss()(sup_logits_pseudo, sup_labels)
                pstr += f'sup_adl_loss: {sup_adl_loss.item():.4f} '
                loss += args.adl_loss_weight * (1 - args.pl_loss_weight) * sup_adl_loss

                unsup_logits_pseudo = torch.cat([f[~mask_lab] for f in (pseudo_out / args.pseudo_temp).chunk(2)], dim=0)
                # softmax score
                unsup_logits = torch.cat([f[~mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                pseudo_label = torch.softmax(unsup_logits.detach(), dim=-1)
                # logit
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                # weighting the loss using distribution certainty score
                unsup_adl_loss = (F.cross_entropy(unsup_logits_pseudo, targets_u, reduction='none') * mask * ood_cer_score).mean()
                pstr += f'unsup_adl_loss: {unsup_adl_loss.item():.4f} '
                loss += args.adl_loss_weight * args.pl_loss_weight * unsup_adl_loss

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            whole_time = time.perf_counter() - start
            start = time.perf_counter()
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t time {:.3f} data_time {:.3f} loss {:.3f}\t {}'.format(epoch, batch_idx, len(train_loader), whole_time, data_time, loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Step schedule
        exp_lr_scheduler.step()
        torch.save(student.state_dict(), args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2 = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, all_acc_test2, old_acc_test2, new_acc_test2 = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            if old_acc_test > best_test_acc_lab:
                args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                # inductive
                best_test_acc_lab = old_acc_test
                # transductive
                best_train_acc_lab = old_acc
                best_train_acc_ubl = new_acc
                best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    preds_pseudo = []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits, _, logits_pseudo = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            preds_pseudo.append(logits_pseudo.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds_pseudo = np.concatenate(preds_pseudo)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    all_acc2, old_acc2, new_acc2 = log_accs_from_preds(y_true=targets, y_pred=preds_pseudo, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    return all_acc, old_acc, new_acc, all_acc2, old_acc2, new_acc2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd', type=str)
    parser.add_argument('--class_num', default=0, type=int)
    parser.add_argument('--dino', type=str, default='v1')
    parser.add_argument('--seed', default=0, type=int)

    # auxiliary debiased classifier
    parser.add_argument('--adl_loss_weight', type=float, default=1.0)
    parser.add_argument('--pl_loss_weight', type=float, default=0.5)
    parser.add_argument('--threshold', default=0.95, type=float, help='debiasing threshold')
    parser.add_argument('--pseudo_temp', default=0.1, type=float)
    parser.add_argument('--save_all', action='store_true', default=False)

    # distribution detector
    parser.add_argument('--sdl_loss_weight', type=float, default=0.01)
    parser.add_argument('--num_ood_layers', default=5, type=int)

    # evaluation
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default='debgcd_models/dinov1/scars/model.pt')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    if not args.class_num:
        args.num_unlabeled_classes = len(args.unlabeled_classes)
    else:
        args.num_unlabeled_classes = args.class_num - args.num_labeled_classes

    init_experiment(args, runner_name=[f'DebGCD_{args.dataset_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    # Add a handler for stdout and configure it to log to stdout as well
    args.logger.add(sys.stdout)

    # ----------------------
    # SET SEED
    # ----------------------
    set_random_seed(args.seed)

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # DINO version
    if args.dino == 'v1':
        backbone = vits.__dict__['vit_base']()
    elif args.dino == 'v2':
        backbone = vits2.__dict__['vit_base']()
        args.warmup_model_dir = args.warmup_model_dir.replace('dino_vitb16', 'dinov2_vitb14_reg4')
    else:
        raise AttributeError('Unsupported DINO version')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    from model import DebGCDHead
    projector = DebGCDHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim , ood_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers , noodlayers=args.num_ood_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_path, map_location='cpu'))
        model = model.to(device)
        all_acc, old_acc, new_acc, _, _, _ = test(model, test_loader_unlabelled, epoch=0, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    else:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
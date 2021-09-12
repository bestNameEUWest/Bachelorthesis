import gc
import logging
import sys
import os
import time
import datetime

from collections import defaultdict

import pandas as pd
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

import scripts.modules.Argparser as ap
import scripts.modules.DSUtils as dsu

from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models.TrajectoryGenerator import TrajectoryGenerator
from sgan.models.TrajectoryDiscriminator import TrajectoryDiscriminator
from sgan.models.SGANTrajectoryDiscriminator import SGANTrajectoryDiscriminator

from scripts.modules.Utils import get_total_norm, relative_to_abs, get_dset_path
from sgan.models.Utils import log

torch.backends.cudnn.benchmark = True

FORMAT = '[%(levelname)s: %(filename)-17s %(lineno)3d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def init_weights(m):
    classname = m.__class__.__name__
    if (classname.find('Linear') != -1) and classname.find('LinearEmbedding') == -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.cuda.LongTensor
    float_dtype = torch.cuda.FloatTensor
    if args.cpu or not torch.cuda.is_available():
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor
    return long_dtype, float_dtype


study_counter = None
run_identifier = f"{datetime.datetime.now():%d-%m-%Y__%H:%M:%S}"


def objective(trial):
    args = ap.parse_args()
    device = torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")

    # Define hyperparams
    if args.optuna:
        args.tf_emb_dim = 2 ** trial.suggest_int("tf_emb_dim_exp", 3, 8)  # 8 - 256
        args.tf_ff_size = 2 ** trial.suggest_int("tf_ff_size_exp", 2, 10)  # 4 - 1024
        args.pool_emb_dim = 2 ** trial.suggest_int("pool_emb_dim_exp", 2, 9)  # 4 - 512
        args.bottleneck_dim = 2 ** trial.suggest_int("bottleneck_dim_exp", 2, 8)  # 4 - 256
        args.mlp_dim = 2 ** trial.suggest_int("mlp_dim_exp", 2, 6)  # 4 - 64
        args.noise_dim = (2 ** trial.suggest_int("noise_dim_exp", 2, 5),)  # 4 - 32
        args.layer_count = trial.suggest_int("layer_count", 2, 4)
        args.dropout = trial.suggest_float("dropout", 1e-1, 5e-1)
        args.g_learning_rate = trial.suggest_float("g_learning_rate", 1e-5, 1e-2, log=True)
        args.d_learning_rate = trial.suggest_float("d_learning_rate", 1e-5, 1e-2, log=True)
        args.heads = 2 ** trial.suggest_int("heads_exp", 1, 3)  # 2 - 8
        if args.sgan_d:
            args.sgan_d_emb_dim = 2 ** trial.suggest_int("sgan_d_emb_dim_exp", 2, 8)  # 4 - 256

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    args.train_path = get_dset_path(args.dataset_name, 'train')
    args.val_path = get_dset_path(args.dataset_name, 'val')
    args.test_path = get_dset_path(args.dataset_name, 'test')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    [train_loader, val_loader, test_loader] = dsu.dataset_loader(args)

    generator = TrajectoryGenerator(
        device=device,
        pool_emb_dim=args.pool_emb_dim,
        tf_emb_dim=args.tf_emb_dim,
        mlp_dim=args.mlp_dim,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        pred_len=args.pred_len,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        batch_norm=args.batch_norm,
        layer_count=args.layer_count,
        activation='leakyrelu',
        heads=args.heads,
    )

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    torch.autograd.set_detect_anomaly(True)

    if args.sgan_d:
        discriminator = SGANTrajectoryDiscriminator(
            device=device,
            tf_emb_dim=args.tf_emb_dim,
            d_layer_count=args.sgan_d_layer_count,
            d_emb_dim=args.sgan_d_emb_dim,
            bottleneck_dim=args.bottleneck_dim,
            pool_emb_dim=args.pool_emb_dim,
            dropout=args.dropout,
            mlp_dim=args.mlp_dim,
            activation='leakyrelu',
            batch_norm=args.batch_norm,
            d_type=args.d_type,
        )
    else:
        discriminator = TrajectoryDiscriminator(
            device=device,
            pool_emb_dim=args.pool_emb_dim,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            d_type=args.d_type,
            activation='leakyrelu',
        )

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(f'{args.checkpoint_name}_with_model.pt')

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info(f'Restoring from checkpoint {restore_path}')
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        epoch = checkpoint['counters']['epoch']
    else:
        # Starting from scratch, so initialize checkpoint data structure
        epoch = 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }

    generator_params = count_parameters(generator)
    discriminator_params = count_parameters(discriminator)

    hyperparams = {
        "tf_emb_dim": args.tf_emb_dim,
        "tf_ff_size": args.tf_ff_size,
        "pool_emb_dim": args.pool_emb_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "mlp_dim": args.mlp_dim,
        "noise_dim": args.noise_dim,
        "layer_count": args.layer_count,
        "g_learning_rate": args.g_learning_rate,
        "d_learning_rate": args.d_learning_rate,
        "heads": args.heads,
        "sgan_d_emb_dim": args.sgan_d_emb_dim,
        "dropout": args.dropout,
        "g_param_count": generator_params,
        "d_param_count": discriminator_params
    }

    t_total = time.time()
    t0 = time.time()

    global study_counter
    if study_counter is None:
        dirs = os.listdir('./runs')
        study_counter = len(dirs)

    global run_identifier
    metrics_dir = os.path.join('runs', f'study_{study_counter}_{run_identifier}', f'trial_{trial.number}')
    os.makedirs(metrics_dir)

    pd.DataFrame.from_dict(hyperparams).to_csv(os.path.join(metrics_dir, 'h_params.csv'), index=False)

    train_metrics_pd = None
    train_metrics_path = os.path.join(metrics_dir, 'train_metrics.csv')

    val_metrics_pd = None
    val_metrics_path = os.path.join(metrics_dir, 'val_metrics.csv')

    while epoch < args.num_epochs:
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        if epoch % 100 == 0:
            logger.info(f'Starting epoch {epoch+1}')
        for batch in train_loader:
            gc.collect()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, device)
                checkpoint['norm_d'].append(get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, device)
                checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                g_steps_left -= 1
            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

        # Maybe save a checkpoint
        if epoch % args.checkpoint_every == 0:
            checkpoint['counters']['epoch'] = epoch

            # Check stats on the validation set
            metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn, device, True)
            metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, device, False, limit=True)

            if args.timing == 1:
                cp_time = time.time() - t0
                metrics_train['time'] = cp_time
                metrics_val['time'] = cp_time
                t0 = time.time()
            metrics_train['epoch'] = epoch
            metrics_val['epoch'] = epoch

            train_metrics_pd = pd.DataFrame([metrics_train], columns=metrics_train.keys())
            if not os.path.isfile(train_metrics_path):
                train_metrics_pd.to_csv(train_metrics_path, index=False)
            else:
                train_metrics_pd.to_csv(train_metrics_path, mode='a', header=False, index=False)

            val_metrics_pd = pd.DataFrame([metrics_val], columns=metrics_val.keys())
            if not os.path.isfile(val_metrics_path):
                val_metrics_pd.to_csv(val_metrics_path, index=False)
            else:
                val_metrics_pd.to_csv(val_metrics_path, mode='a', header=False, index=False)

            for k, v in sorted(metrics_val.items()):
                # logger.info('  [val] {}: {:.3f}'.format(k, v))
                checkpoint['metrics_val'][k].append(v)
            for k, v in sorted(metrics_train.items()):
                # logger.info('  [train] {}: {:.3f}'.format(k, v))
                checkpoint['metrics_train'][k].append(v)

            min_mad = min(checkpoint['metrics_val']['mad'])
            min_mad_nl = min(checkpoint['metrics_val']['mad_nl'])

            if metrics_val['mad'] == min_mad:
                # logger.info('New low for avg_disp_error')
                checkpoint['g_best_state'] = generator.state_dict()
                checkpoint['d_best_state'] = discriminator.state_dict()

            if metrics_val['mad_nl'] == min_mad_nl:
                # logger.info('New low for avg_disp_error_nl')
                checkpoint['g_best_nl_state'] = generator.state_dict()
                checkpoint['d_best_nl_state'] = discriminator.state_dict()

            # Save another checkpoint with model weights and
            # optimizer state
            checkpoint['g_state'] = generator.state_dict()
            checkpoint['g_optim_state'] = optimizer_g.state_dict()
            checkpoint['d_state'] = discriminator.state_dict()
            checkpoint['d_optim_state'] = optimizer_d.state_dict()

            # Save a checkpoint with no model weights by making a shallow
            # copy of the checkpoint excluding some items
            key_blacklist = [
                'g_state', 'd_state', 'g_best_state', 'g_best_nl_state', 'g_optim_state', 'd_optim_state',
                'd_best_state', 'd_best_nl_state'
            ]
            small_checkpoint = {}
            for k, v in checkpoint.items():
                if k not in key_blacklist:
                    small_checkpoint[k] = v
            # torch.save(small_checkpoint, checkpoint_path)

            # Handle pruning based on the intermediate value.
            trial.report(metrics_train["mad"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # t_mem = torch.cuda.get_device_properties(0).total_memory
            # r_mem = torch.cuda.memory_reserved(0)
            # a_mem = torch.cuda.memory_allocated(0)
            # f_mem = r-a  # free inside reserved
            # print(f'Total memory: {t_mem}')
            # print(f'Reserved memory: {r_mem}')
            # print(f'Reserved/Total: {r_mem/t_mem:.1f} %')
            # print(f'Allocated/Total: {a_mem/t_mem:.1f} %')
            # print(f'Allocated/Reserved: {a_mem/t_mem:.1f} %')

        if args.print:
            logger.info(f'Epoch {epoch} took {metrics_train["time"]:.2f}s')
            logger.info(f'Train: mad = {metrics_train["mad"]:.3f}  fad = {metrics_train["fad"]:.3f}')
            logger.info(f'Val: mad = {metrics_val["mad"]:.3f}  fad = {metrics_val["fad"]:.3f}')
            for k, v in sorted(losses_d.items()):
                logger.info('  [D] {}: {:.3f}'.format(k, v))
                checkpoint['D_losses'][k].append(v)
            for k, v in sorted(losses_g.items()):
                logger.info('  [G] {}: {:.3f}'.format(k, v))
                checkpoint['G_losses'][k].append(v)
        epoch += 1
    logger.info(f'Total training time: {time.time() - t_total}')
    if not args.optuna:
        exit()
    return min_mad


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, device):
    batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, mean_rel, std_rel) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, pred_traj_gt_rel, seq_start_end, False)

    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, device):
    batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, mean_rel, std_rel) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, pred_traj_gt_rel, seq_start_end, False)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_accuracy(args, loader, generator, discriminator, d_loss_fn, device, predict, limit=False):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, mean_rel, std_rel) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, pred_traj_gt_rel, seq_start_end, predict)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            mad, mad_l, mad_nl = cal_mad(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

            fad, fad_l, fad_nl = cal_fad(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(mad.item())
            disp_error_l.append(mad_l.item())
            disp_error_nl.append(mad_nl.item())
            f_disp_error.append(fad.item())
            f_disp_error_l.append(fad_l.item())
            f_disp_error_nl.append(fad_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['mad'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fad'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['mad_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fad_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['mad_l'] = 0
        metrics['fad_l'] = 0
    if total_traj_nl != 0:
        metrics['mad_nl'] = sum(disp_error_nl) / (total_traj_nl * args.pred_len)
        metrics['fad_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['mad_nl'] = 0
        metrics['fad_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_mad(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    mad = displacement_error(pred_traj_fake, pred_traj_gt)
    mad_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    mad_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return mad, mad_l, mad_nl


def cal_fad(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fad = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fad_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fad_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fad, fad_l, fad_nl


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=80, gc_after_trial=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

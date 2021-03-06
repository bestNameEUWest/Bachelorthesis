import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from scripts.modules.Utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint, device):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        device=device,
        pool_emb_dim=args.pool_emb_dim,
        tf_emb_dim=args.tf_emb_dim,
        mlp_dim=args.mlp_dim,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.to(device)
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    mad_outer, fad_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            mad, fad = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                mad.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fad.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            mad_sum = evaluate_helper(mad, seq_start_end)
            fad_sum = evaluate_helper(fad, seq_start_end)

            mad_outer.append(mad_sum)
            fad_outer.append(fad_sum)
        mad = sum(mad_outer) / (total_traj * args.pred_len)
        fad = sum(fad_outer) / (total_traj)
        return mad, fad


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint, args.device)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        mad, fad = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, MAD: {:.2f}, FAD: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, mad, fad))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

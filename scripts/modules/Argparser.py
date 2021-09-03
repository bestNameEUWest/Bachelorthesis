import argparse
import os

from scripts.modules.Utils import int_tuple, bool_flag

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset options
    parser.add_argument('--dataset_folder', default='datasets', type=str)
    parser.add_argument('--dataset_name', default='zara1', type=str)
    parser.add_argument('--raw_dataset_folder', default='raw_data', type=str)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--loader_num_workers', default=2, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=8, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--step', default=10, type=int)

    # Optimization
    parser.add_argument('--optuna', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    # Model Options
    parser.add_argument('--pool_emb_dim', default=64, type=int)
    parser.add_argument('--tf_emb_dim', default=512, type=int)
    parser.add_argument('--tf_ff_size', default=2048, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--layer_count', default=6, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)
    parser.add_argument('--mlp_dim', default=1024, type=int)

    # Generator Options
    parser.add_argument('--noise_dim', default=[0], type=int_tuple)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise_mix_type', default='ped')
    parser.add_argument('--clipping_threshold_g', default=0, type=float)
    parser.add_argument('--g_learning_rate', default=5e-4, type=float)
    parser.add_argument('--g_steps', default=1, type=int)

    # Pooling Options
    parser.add_argument('--pooling_type', default='pool_net')
    parser.add_argument('--bottleneck_dim', default=1024, type=int)

    # Discriminator Options
    parser.add_argument('--d_type', default='local', type=str)
    parser.add_argument('--d_learning_rate', default=5e-4, type=float)
    parser.add_argument('--d_steps', default=2, type=int)
    parser.add_argument('--clipping_threshold_d', default=0, type=float)

    # Loss Options
    parser.add_argument('--l2_loss_weight', default=0, type=float)
    parser.add_argument('--best_k', default=1, type=int)

    # Output
    parser.add_argument('--output_dir', default=os.getcwd())
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--checkpoint_every', default=10, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    parser.add_argument('--num_samples_check', default=5000, type=int)

    # Misc
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--timing', default=0, type=int)
    parser.add_argument('--gpu_num', default="0", type=str)

    return parser.parse_args()


import logging

from torch.utils.data import DataLoader
from sgan.data.trajectories import TrajectoryDataset, seq_collate

logger = logging.getLogger(__name__)


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    logger.info('DISABLED SHUFFLE IN LOADER.PY')    
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        #shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader

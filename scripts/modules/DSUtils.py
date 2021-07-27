import logging
import os

import torch

from datetime import datetime
from sgan.data.loader import data_loader

logger = logging.getLogger(__name__)


def is_data_prepared(save_dir, dataset_info):
  logger.info('Checking if data is already prepared')
  save_file_name = 'info.pt'
  
  for (root, dirs, files) in os.walk(save_dir):
    for file in files:
      if save_file_name == file:
        save_file_path = os.path.join(root, save_file_name)        
        saved_info = torch.load(save_file_path)
        if saved_info == dataset_info:
          return True, root  
  return False, None


def dataset_loader(args):
    pytorch_data_save_dir = 'pytorch_data_save'
    try:
        os.makedirs(f'{pytorch_data_save_dir}/{args.dataset_name}')
    except:
        pass
    
    dataset_info  = {
        'dataset_name': args.dataset_name,
        'obs_len': args.obs_len,
        'pred_len': args.pred_len,
        # 'steps': args.steps,
        'skip': args.skip,
        # 'col_names': args.col_names,    
    }

    # data preparation check
    if args.dataset_name == 'rounD':
        datasets_formatted_check(args)

    save_dir_name = datetime.now().strftime("%d-%m-%Y_%Ss-%Mm-%Hh")
    available, path = is_data_prepared(pytorch_data_save_dir, dataset_info)

    labels = ['train', 'val', 'test']
    paths = [args.train_path, args.val_path, args.test_path]
    dss = [None, None, None]
    loaders = [None, None, None]
    for i in range(len(dss)):    
        if available:
            dss[i] = torch.load(os.path.join(path, f'{labels[i]}_dset.pt'))        
            loaders[i] = torch.load(os.path.join(path, f'{labels[i]}_loader.pt'))                  
            logger.info(f'Loaded prepared {args.dataset_name} {labels[i]} data')          
            logger.info(f'Data info: {dataset_info}')
        else:
            logger.info(f'Prepared {args.dataset_name} {labels[i]} data not found')
            logger.info('Preparing data...')
            dss[i], loaders[i] = data_loader(args, paths[i])         
            save_dir_path = os.path.join(pytorch_data_save_dir, args.dataset_name,
            save_dir_name)  
            try:
                os.makedirs(save_dir_path)
            except:
                pass
            torch.save(dss[i], os.path.join(save_dir_path, f'{labels[i]}_dset.pt'))
            torch.save(loaders[i], os.path.join(save_dir_path, f'{labels[i]}_loader.pt'))
    if not available:
        torch.save(dataset_info, os.path.join(save_dir_path, 'info.pt'))
        logger.info(f'Prepared and saved data with: {dataset_info}')

    return dss, loaders


def datasets_formatted_check(args):
    try:
        os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "train"))
        os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "val"))
        os.listdir(os.path.join(args.dataset_folder, args.dataset_name, "test"))    
    except FileNotFoundError:
        format_raw_dataset(args)  


def format_raw_dataset(args):
    test_and_val_ratio = 0.15
    relevant_cols = ['frame', 'obj', 'x', 'y', 'heading', 'width', 'length', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']  # all relevant data
    files = [dir for dir in os.listdir(os.path.join(args.raw_dataset_folder, args.dataset_name))]
    metasets_list = [dir for dir in files if 'Meta' in dir]
    datasets_list = [dir for dir in files if 'Meta' not in dir]
    datasets_list.sort()  
    metasets_list.sort()
    
    try:
        os.makedirs(os.path.join(args.dataset_folder, args.dataset_name))      
    except:
        pass

    for i, (dataset, metaset) in enumerate(zip(datasets_list, metasets_list)):
        data = pd.read_csv(os.path.join(args.raw_dataset_folder, args.dataset_name, dataset))
        data = data.rename(columns={'trackId': 'obj', 'xCenter': 'x', 'yCenter': 'y'})
        data = data[relevant_cols]
        
        if args.use_classes:
           data = add_classes(data, args, metaset)
          
        raw_data = data.astype(np.float32)    
        objs = raw_data.obj.unique()
        np.random.seed(42)
        np.random.shuffle(objs)

        val_i = int((1 - 2*test_and_val_ratio)*len(objs))
        test_i = int((1 - test_and_val_ratio)*len(objs))            
        obj_train, obj_val, obj_test = np.split(objs, [val_i, test_i])

        set_types = ['train', 'val', 'test']
        srcs = [obj_train, obj_val, obj_test] 
        pd.options.mode.chained_assignment = None  # default='warn'
        for src, set_type in zip(srcs, set_types):
            trg = []
            for obj in src:
                obj_data = raw_data[raw_data.obj == obj]
                obj_data = add_synthetic_values(obj_data)
                trg.append(obj_data)        
            trg = pd.concat(trg, ignore_index=True)
            try:
                path = os.path.join(args.dataset_folder, args.dataset_name, set_type)
                os.makedirs(path)   	
            except:
                pass
            data_path = os.path.join(args.dataset_folder, args.dataset_name, set_type, f'{i:02d}_{set_type}.csv')
            trg.to_csv(path_or_buf=data_path, index=False, sep='\t')  
        pd.options.mode.chained_assignment = 'warn'

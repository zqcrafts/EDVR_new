"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler,
                                           pin_memory=True)
    else:
        num_workers = dataset_opt['n_workers']
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                           pin_memory=False)


def create_dataset(opt,dataset_opt):

    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'Vimeo90K':
        from data.Vimeo90K_dataset import DatasetFromFolder as D
        dataset = D(upscale_factor=opt['scale'], data_augmentation=dataset_opt['augment'],
                    group_file=dataset_opt['filelist'],
                    patch_size=dataset_opt['LQ_size'], black_edges_crop=False, hflip=True, rot=True)

    elif mode == 'video_test':
        from data.Vimeo90K_dataset import DatasetFromFolder as D
        # dataset = D(upscale_factor=opt['scale'], data_augmentation=dataset_opt['augment'],
        #             group_file=dataset_opt['filelist'],
        #             patch_size=None, black_edges_crop=False, hflip=False, rot=False)
        dataset = D(upscale_factor=opt['scale'], data_augmentation=None,
            group_file=dataset_opt['filelist'],
            patch_size=None, black_edges_crop=False, hflip=False, rot=False)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))


    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

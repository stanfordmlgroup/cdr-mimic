import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
import util


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Head and Spine CT')
        self.parser.add_argument('--model', type=str, choices=('QANet', 'SimpleNN'), default='SimpleNN',
                                 help='Model name.')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
        self.parser.add_argument('--ckpt_path', type=str, default='',
                                 help='Path to checkpoint to load. If empty, start from scratch.')
        self.parser.add_argument('--data_dir', type=str, default='/deep/group/data/squad_question_generation/',
                                 help='Path to data directory with question answering dataset.')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'),
                                 help='Initialization method to use for conv kernels and linear weights.')
        self.parser.add_argument('--model_depth', default=50, type=int,
                                 help='Depth of the model. Meaning of depth depends on the model.')
        self.parser.add_argument('--name', type=str, required=True, help='Experiment name.')
        self.parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in the input.')
        self.parser.add_argument('--num_classes', default=1, type=int, help='Number of classes to predict.')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--save_dir', type=str, default='ckpts/',
                                 help='Directory in which to save model checkpoints.')
        self.parser.add_argument('--dataset', type=str, default='mimic', choices=('mimic',),
                                 help='Dataset to use. Gets mapped to dataset class name.')
        self.parser.add_argument('--verbose', action="store_true")

        self.is_training = None

    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        save_dir = os.path.join(args.save_dir, args.name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        args.start_epoch = 1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.epochs_per_save % args.epochs_per_eval != 0:
            raise ValueError('epochs_per_save must be divisible by epochs_per_eval.')
        if args.is_training:
            args.maximize_metric = not args.metric_name.endswith('loss')
            if args.lr_scheduler == 'multi_step':
                args.lr_milestones = util.args_to_list(args.lr_milestones, allow_empty=False)

        # Set up available GPUs
        args.gpu_ids = util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            cudnn.benchmark = True
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        if args.dataset == 'mimic':
            args.dataset = 'MimicDataset'

        # Set up output dir (test mode only)
        if not self.is_training:
            args.results_dir = os.path.join(args.results_dir, args.name)
            os.makedirs(args.results_dir, exist_ok=True)

        return args

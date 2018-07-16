import numpy as np
import os
import torch.utils.data as data


class SQuID(data.Dataset):
    """Question inference dataset based on SQuAD."""
    def __init__(self, args, phase, data_dir, is_training_set):
        """
        Args:
            phase: One of 'train', 'dev', 'test'.
            data_dir: Root data directory.
            is_training_set: If true, this set will be used for training. Else for inference.
        """
        self.phase = phase
        self.data_dir = data_dir
        self.is_training_set = is_training_set

        # Load arrays filled with IDs
        src_path = os.path.join(args.data_dir, phase, 'src_ids.npy')
        self.src_ids = np.load(src_path)
        src_c_path = os.path.join(args.data_dir, phase, 'src_c_ids.npy')
        self.src_c_ids = np.load(src_c_path)

        bio_path = os.path.join(args.data_dir, phase, 'bio_ids.npy')
        self.bio_ids = np.load(bio_path)

        tgt_path = os.path.join(args.data_dir, phase, 'tgt_ids.npy')
        self.tgt_ids = np.load(tgt_path)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        src = self.src_ids[idx]
        src_c = self.src_c_ids[idx]
        bio = self.bio_ids[idx]
        tgt = self.tgt_ids[idx]

        return src, src_c, bio, tgt

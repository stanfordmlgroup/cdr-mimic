import torch
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import TestArgParser
from dataset import SQuAD
from logger import TestLogger
from saver import ModelSaver
from tqdm import tqdm


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    data_loader = data.DataLoader(CoronaryCTADataset(args.data_dir, is_training_set=True),
                                  args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    logger = TestLogger(args, len(data_loader.dataset))

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, info_dict) in enumerate(data_loader):
            with torch.no_grad():
                logits = model.forward(inputs.to(args.device))
                probs = F.sigmoid(logits)

            # TODO: Test script is incomplete. Does nothing with the masks.

            progress_bar.update(inputs.size(0))


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    test(parser.parse_args())

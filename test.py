import torch
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver
from tqdm import tqdm


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    data_loader = get_loader(args, phase=args.phase, is_training=False)
    logger = TestLogger(args, len(data_loader.dataset))

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))
    outputs = []
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, info_dict) in enumerate(data_loader):
            with torch.no_grad():
                pred_params = model.forward(inputs.to(args.device))
                outputs.append(pred_params.cpu().numpy()

            progress_bar.update(inputs.size(0))

    # print pred_params (mu, s) to file
    np.save('outputs.npy', outputs)
    np.savetxt('outputs.csv', outputs)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    test(parser.parse_args())

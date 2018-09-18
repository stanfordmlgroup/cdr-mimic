import torch
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver
from tqdm import tqdm
from data import Dataset, get_loader


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    data_loader = get_loader(args, phase=args.phase, is_training=False)
    logger = TestLogger(args, len(data_loader.dataset))

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))

    all_gender = []
    all_age = []
    all_tte = []
    all_is_alive = []
    all_mu = []
    all_s2 = []
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (src, tgt) in enumerate(data_loader):
            #import pdb
            #pdb.set_trace()
            all_gender.extend([int(x) for x in src[:,0]])
            all_age.extend([float(x) for x in src[:,1]])
            all_tte.extend([float(x) for x in tgt[:,0]])
            all_is_alive.extend([int(x) for x in tgt[:,1]])
            with torch.no_grad():
                pred_params = model.forward(src.to(args.device))

                outputs = pred_params.cpu().numpy()
                all_mu.extend([float(x) for x in outputs[0]])
                all_s2.extend([float(x) for x in outputs[1]])

            progress_bar.update(src.size(0))

    # print pred_params (mu, s) to file
    fd = open(args.results_dir + '/test_stats.csv', 'w')
    fd.write('gender, age, tte, is_alive, mu, s2\n')
    for gender, age, tte, is_alive, mu, s2 \
        in zip(all_gender, all_age, all_tte, all_is_alive, all_mu, all_s2):
        
        fd.write('%d, %f, %f, %d, %f, %f\n' % (gender, age, tte, is_alive, mu, s2))
    fd.close()

    #np.save('outputs.npy', outputs)
    #np.savetxt('outputs.csv', outputs)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    test(parser.parse_args())

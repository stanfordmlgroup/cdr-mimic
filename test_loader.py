import argparse
from data.loader import get_loader

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)

    return parser


args = get_parser().parse_args()

loader = get_loader(args)
# print(iter(loader).next())
import pdb
pdb.set_trace()
for src, tgt in loader:
	print(src[0]) #, tgt)

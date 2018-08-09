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

loader, D_in = get_loader(args)
print(f'D_in is {D_in}')
# print(iter(loader).next())
for src, tgt in loader:
	print(src, tgt)

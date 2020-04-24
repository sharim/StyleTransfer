import sys
import torch
from utils import getArgument
from stylize import stylize
from train import train


def main():
    args = getArgument()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    train(args) if args.subcommand == "train" else stylize(args)


if __name__ == "__main__":
    main()

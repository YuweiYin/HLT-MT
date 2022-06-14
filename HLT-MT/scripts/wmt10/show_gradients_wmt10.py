import argparse
import xlwt
import xlrd
import os
import numpy as np

LANGS = "fr cs de fi lv et ro hi tr gu".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gradients', '-gradients', type=str,
                        default=r'/path/to/Evaluation/wmt10/BLEU/',
                        help='input stream')
    parser.add_argument('--checkpoint', '-checkpoint', type=str,
                        default=r'/path/to/Evaluation/wmt10/BLEU/',
                        help='input stream')
    args = parser.parse_args()
    return args


def _lang_pair(src, tgt):
    return "{}-{}".format(src, tgt)


if __name__ == "__main__":
    args = parse_args()
    x2x = {}
    print("MODEL: {}".format(checkpoint_name))
    for i, src in enumerate(LANGS):
        grad_path = os.path.join(args.log, "{}-{}.grad".format("en", src))
        x2x["{}-{}".format(src, "en")] = np.load(grad_path)
        grad_path = os.path.join(args.log, "{}-{}.grad".format(src, "en"))
        x2x["{}-{}".format(src, "en")] = np.load(grad_path)

import numpy as np
import argparse
import os

LANGS = "fr cs de fi lv et ro hi tr gu".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-path', type=str,
                        default=r'/path/to/checkpoint/grads/',
                        help='input stream')
    args = parser.parse_args()
    return args


def cosine_similarity(x, y):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    xy = sum(x * y)
    x2 = np.linalg.norm(x)
    y2 = np.linalg.norm(y)
    return xy / (x2 * y2)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    PATH = args.path
    results = []
    w_results = []
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    with open(os.path.join(args.path, "cosine.txt"), "w", encoding="utf-8") as w:
        for src in LANGS:
            tmp = []
            for tgt in LANGS:
                src_grads = np.load("{}/en-{}.grad.npy".format(PATH, src))
                tgt_grads = np.load("{}/en-{}.grad.npy".format(PATH, tgt))
                cosine_score = cosine_similarity(src_grads.astype(np.float32), tgt_grads.astype(np.float32))
                results.append(cosine_score)
                output_str = "{}: [en->{}]-[en->{}]: {}".format(PATH, src, tgt, round(cosine_score, 2))
                print(output_str)
                w.write("{}\n".format(output_str))
                tmp.append(cosine_score)
            w_results.append(tmp)
        output_str = "Avg: {}\n".format(sum(results) / len(results))
        print(output_str)
        w.write(output_str)
        print(results)
        w.write("{}\n".format(w_results))

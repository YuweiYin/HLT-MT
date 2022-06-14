import argparse
import os

LANGS = "en fr cs de fi lv et ro hi tr gu".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/alignment-train/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/alignment-train/overlaps.txt',
                        help='input stream')
    parser.add_argument('--workers', '-workers', type=int,
                        default=80, help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lang_dicts = {}
    for lang in LANGS:
        lang_dicts[lang] = set()
        if lang == "en":
            file_name = os.path.join(args.input, "train.en-de.en")
        else:
            file_name = os.path.join(args.input, "train.en-{}.{}".format(lang, lang))
        with open(file_name, "r", encoding="utf-8") as r:
            count = 0
            for line in r:
                words = set(line.strip().split())
                lang_dicts[lang].update(words)
                count += 1
                if count % 1000000 == 0:
                    print(count)
        print("Successfully Loading Language {} dictionary: {}".format(lang, len(lang_dicts[lang])))
    results = []
    for src in LANGS:
        result = []
        for tgt in LANGS:
            if src != tgt:
                subsection = lang_dicts[src] & lang_dicts[tgt]
                union = lang_dicts[src] | lang_dicts[tgt]
                score = float(len(subsection)) / len(union)
            else:
                score = 1.0
            print("{}-{}: {}".format(src, tgt, score))
            result.append(score)
        results.append(result)
    print(results)
    with open(args.output, "w", encoding="utf-8") as w:
        w.write("{}\n".format(results))

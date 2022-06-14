import argparse
import xlwt
import xlrd
import os
from collections import OrderedDict

TOTAL_DIRECTION = 30


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


LANGS = "fr cs de fi lv et ro hi tr gu".split()
HIGH_LANGS = "fr cs de fi lv et".split()
LOW_LANGS = "ro hi tr gu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/path/to/Evaluation/wmt10/BLEU/',
                        help='input stream')
    parser.add_argument('--checkpoint-name', '-checkpoint-name', type=str,
                        default=r'/path/to/checkpoint.pt',
                        help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name, save_dir='/path/to/result/'):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(name, cell_overwrite_ok=True)
    worksheet.write(1, 0, label="DeltaLM-Postnorm (Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(i + 1, 0, label=LANGS[i])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            worksheet.write(i + 1, j + 1, label=results[i][j])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    workbook.save('{}/{}.xls'.format(save_dir, name))
    return workbook


def _lang_pair(src, tgt):
    return "{}->{}".format(src, tgt)


def calculate_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        results.append(round(avg, 2))
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    else:
        for key in x2x.keys():
            results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: all: {:.2f}".format(model_name, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)


def calculate_high_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        results.append(round(avg, 2))
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key and key.split("->")[1] in HIGH_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key and key.split("->")[0] in HIGH_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    else:
        for key in x2x.keys():
            if key.split("->")[0] in HIGH_LANGS or key.split("->")[1] in HIGH_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: all: {:.2f}".format(model_name, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)


def calculate_low_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        results.append(round(avg, 2))
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key and key.split("->")[1] in LOW_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key and key.split("->")[0] in LOW_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    else:
        for key in x2x.keys():
            if key.split("->")[0] in LOW_LANGS or key.split("->")[1] in LOW_LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: all: {:.2f}".format(model_name, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)


if __name__ == "__main__":
    args = parse_args()
    x2x = {}
    results = []
    checkpoint_name = args.checkpoint_name
    print("MODEL: {}".format(checkpoint_name))
    for i, src in enumerate(LANGS):
        results.append([])
        with open(os.path.join(args.log, "{}-{}.BLEU".format(src, "en")), "r", encoding="utf-8") as r:
            result_lines = r.readlines()
            for i in range(len(result_lines) - 1, -1, -1):  # reversed search
                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace(
                        "MODEL: ", ""):
                    last_line = result_lines[i + 1]  # read the latest results
                    if 'BLEU+case.mixed' in last_line:
                        score = float(last_line.split()[2])
                        x2x["{}->{}".format(src, "en")] = score
                        results[-1].append(score)
                        break
                    else:
                        print(os.path.join(args.log, "{}-{}.BLEU".format(src, "en")))
                        break
                if i == 0:
                    print(result_lines)

        with open(os.path.join(args.log, "{}-{}.BLEU".format("en", src)), "r", encoding="utf-8") as r:
            result_lines = r.readlines()
            for i in range(len(result_lines) - 1, -1, -1):  # reversed search
                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace(
                        "MODEL: ", ""):
                    last_line = result_lines[i + 1]  # read the latest results
                    if 'BLEU+case.mixed' in last_line:
                        score = float(last_line.split()[2])
                        x2x["{}->{}".format("en", src)] = score
                        results[-1].append(score)
                        break
                    else:
                        print(os.path.join(args.log, "{}-{}.BLEU".format("en", src)))

    calculate_avg_score(x2x, src="en", model_name="our")
    calculate_avg_score(x2x, tgt="en", model_name="our")
    calculate_avg_score(x2x, model_name="our")

    calculate_high_avg_score(x2x, src="en", model_name="our_high")
    calculate_high_avg_score(x2x, tgt="en", model_name="our_high")
    calculate_high_avg_score(x2x, model_name="our_high")

    calculate_low_avg_score(x2x, src="en", model_name="our_low")
    calculate_low_avg_score(x2x, tgt="en", model_name="our_low")
    calculate_low_avg_score(x2x, model_name="our_low")

    # name = "wmt10"
    # create_excel(results, name=name)

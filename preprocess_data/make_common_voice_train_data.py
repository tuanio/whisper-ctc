import json
import os
import argparse
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in tqdm(data, desc=f'Dumping data, file: "{path}"'):
            line["label"] = " ".join(line["phoneme"])
            del line["type"]
            del line["grapheme"]
            del line["phoneme"]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main(args):
    for subset in ["train", "test", "dev"]:
        src_path = os.path.join(args.cv_path, f"common_voice_{subset}.jsonl")
        des_path = os.path.join(args.dest_path, f"cv_{subset}.jsonl")
        f = open(des_path, "w", encoding="utf-8")
        with open(src_path, "r", encoding="utf-8") as ff:
            for line_data in tqdm(ff, desc=f'Dumping data, file: "{des_path}"'):
                line = json.loads(line_data)
                line["label"] = " ".join(line["phoneme"])
                del line["grapheme"]
                del line["phoneme"]
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-path",
        default="raw_data/",
        type=str,
        help="Json line file of indict tts created by `indict_tts.py`",
    )
    parser.add_argument(
        "--dest-path",
        default="train_data",
        help="Destination folder of indict tts subsets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Control random state, different train/test/valid for each seed",
    )

    args = parser.parse_args()
    main(args)

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
    data = [json.loads(i) for i in open(args.indict_tts_path, "r", encoding="utf-8")]
    _type = [i["type"] for i in data]
    eval_ratio = args.test_ratio + args.dev_ratio
    train_data, test_data, train_type, test_type = train_test_split(
        data, _type, test_size=eval_ratio, random_state=args.seed, stratify=_type
    )
    test_data, dev_data = train_test_split(
        test_data,
        test_size=args.test_ratio / eval_ratio,
        random_state=args.seed,
        stratify=test_type,
    )

    dump_json(train_data, os.path.join(args.dest_path, "indict_tts_train.jsonl"))
    dump_json(test_data, os.path.join(args.dest_path, "indict_tts_test.jsonl"))
    dump_json(dev_data, os.path.join(args.dest_path, "indict_tts_dev.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indict-tts-path",
        default="raw_data/indict_tts.jsonl",
        type=str,
        help="Json line file of indict tts created by `indict_tts.py`",
    )
    parser.add_argument(
        "--dest-path",
        default="train_data",
        help="Destination folder of indict tts subsets",
    )
    parser.add_argument("--test-ratio", type=float, default="0.1")
    parser.add_argument("--dev-ratio", type=float, default="0.1")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Control random state, different train/test/dev for each seed",
    )

    args = parser.parse_args()
    main(args)

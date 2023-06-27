import os
import json
import pandas as pd
from tqdm.auto import tqdm
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

cv_path = "cv-corpus-13.0-2023-03-09/en"
audio_path = os.path.join(cv_path, "clips/")

backend = EspeakBackend("en-us")

# separate phones by a space and ignoring words boundaries
separator = Separator(phone=",", word=" ")


def get_phoneme(text):
    phoneme = backend.phonemize([text], separator=separator, strip=True)[0]
    phoneme = [j for i in phoneme.split() for j in i.split(",")]
    return phoneme


for subset in tqdm(["train", "dev", "test"], desc="Loading subset"):
    df = pd.read_csv(os.path.join(cv_path, subset + ".tsv"), sep="\t")[
        ["path", "sentence"]
    ]
    df["path"] = df.path.apply(lambda x: os.path.join(audio_path, x))
    df = df.rename(columns={"path": "audio_path", "sentence": "grapheme"})
    df["phoneme"] = df.grapheme.apply(get_phoneme)
    f = open("common_voice_" + subset + ".jsonl", "w")
    for d in df.to_dict("records"):
        f.write(json.dumps(d) + "\n", ensure_ascii=False)
    f.close()

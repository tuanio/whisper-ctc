import os
import re
from tqdm.auto import tqdm
import numpy as np
import json
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

indict_tts_path = "/kaggle/input/indictts-english/IndicTTS/"

wav_files = []
wav_types = []
transcripts = {}
for root, dirs, files in tqdm(os.walk(indict_tts_path)):
    for file in files:
        if "english" in root:
            filepath = os.path.join(root, file)
            if file == "txt.done.data":
                f = open(filepath, "r", encoding="ISO-8859-1")
                try:
                    for line in f:
                        key, trans = re.findall(r"\(\s?(.+) \"\s?(.+)\s?\" \)", line)[0]
                        transcripts[key.strip()] = trans.strip()
                except UnicodeDecodeError:
                    pass
            elif file.endswith(".wav"):
                f_type = filepath.rsplit("_", 2)[1]
                wav_types.append(f_type)
                wav_files.append(filepath)

wav_files_processed = [i.split(indict_tts_path)[-1] for i in wav_files]

# punctuation marks
text = "They speak English at work."

text = Punctuation(';:,.!"?()-').remove(text)

# build the set of all the words in the text
words = {w.lower() for w in text.strip().split(" ")}

# initialize the espeak backend for English
backend = EspeakBackend("en-us")

# separate phones by a space and ignoring words boundaries
separator = separator = Separator(phone=",", word=" ")


def get_phoneme(text):
    phoneme = backend.phonemize([text], separator=separator, strip=True)[0]
    phoneme = [j for i in phoneme.split() for j in i.split(",")]
    return phoneme


data = []
for i in tqdm(np.arange(len(wav_types))):
    path = wav_files_processed[i]
    _type = wav_types[i]
    key = path.rsplit(os.sep)[-1].split(".")[0]
    grapheme = transcripts[key]
    phoneme = get_phoneme(grapheme)

    datum = {
        "audio_path": path,
        "type": _type,
        "grapheme": grapheme,
        "phoneme": phoneme,
    }
    data.append(datum)

with open("indict_tts.jsonl", "w") as f:
    for datum in data:
        f.write(json.dumps(datum, ensure_ascii=False) + "\n")

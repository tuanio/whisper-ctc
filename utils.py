import torchaudio
from transformers import Wav2Vec2PhonemeCTCTokenizer, WhisperFeatureExtractor

BASE_SAMPLE_RATE = 16000
ENCODER_MAX_LENGTH = 3000
ENCODER_OUTPUT_LENGTH = ENCODER_MAX_LENGTH // 2


def load_and_resample(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != BASE_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, BASE_SAMPLE_RATE)
    return wav.mean(axis=0).numpy()  # mean the channel


def make_feat_extractor_and_tokenizer(feat_extractor_id, tokenizer_id):
    feat_extractor = WhisperFeatureExtractor.from_pretrained(feat_extractor_id)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(tokenizer_id)
    return feat_extractor, tokenizer


def greedy_decode(ids, tokenizer):
    return tokenizer.decode(ids)

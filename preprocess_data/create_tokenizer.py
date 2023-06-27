from transformers import Wav2Vec2PhonemeCTCTokenizer

# change your tokenizer here
tokenizer = Wav2Vec2PhonemeCTCTokenizer(
    vocab_file="vocab/phones_vocab.json", do_phonemize=False
)

tokenizer.save_pretrained("wav2vec2-phoneme-ipa-ctc")
api.upload_folder(
    folder_path="wav2vec2-phoneme-ipa-ctc",
    repo_id=f"tuanio/wav2vec2-phoneme-ipa-ctc",
    create_pr=1,
    repo_type="model",
    multi_commits=True,
    multi_commits_verbose=True,
)

from huggingface_hub import HfApi
from transformers import WhisperFeatureExtractor, WhisperModel

# need login
api = HfApi()

for submodel in ["tiny", "base", "small", "medium"]:
    model_id = model_id = f"openai/whisper-{submodel}.en"

    feat_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    encoder = WhisperModel.from_pretrained(model_id).get_encoder()

    feat_extractor.save_pretrained(f"openai/whisper-encoder.{submodel}.en")
    encoder.save_pretrained(f"openai/whisper-encoder.{submodel}.en")

    api.upload_folder(
        folder_path=f"openai/whisper-encoder.{submodel}.en",
        repo_id=f"tuanio/whisper-encoder.{submodel}.en",
        create_pr=1,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )

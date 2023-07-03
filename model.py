import torch
from torch import nn
from transformers import get_scheduler, Wav2Vec2PhonemeCTCTokenizer
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import lightning as L
from torch import Tensor
from utils import ENCODER_OUTPUT_LENGTH
from jiwer import wer as cal_wer


class WhisperCTC(nn.Module):
    def __init__(
        self, encoder_id: str = "tuanio/whisper-encoder.tiny.en", vocab_size: int = 47
    ):
        super().__init__()
        self.encoder = WhisperEncoder.from_pretrained(encoder_id)
        print("Freezing Whisper Encoder...")
        self.encoder._freeze_parameters()
        print("Freezed!")
        self.ctc_head = nn.Linear(self.encoder.config.d_model, vocab_size)

    def forward(self, feat: Tensor, attn_mask: Tensor):
        enc = self.encoder(
            input_features=feat, attention_mask=attn_mask
        ).last_hidden_state
        logits = self.ctc_head(enc)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs


class WhisperModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = WhisperCTC(**cfg.model)
        self.ctc_loss = nn.CTCLoss(blank=3)  # as in vocab json
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(cfg.tokenizer_id)
        self.cfg = cfg

    def forward(self, feat: Tensor, attn_mask: Tensor):
        return self.model(feat, attn_mask)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.cfg.optim)
        scheduler = {
            "scheduler": get_scheduler(
                name=self.cfg.scheduler.name,
                optimizer=optimizer,
                num_warmup_steps=round(
                    self.cfg.scheduler.warmup_ratio * self.cfg.scheduler.total_steps
                ),
                num_training_steps=self.cfg.scheduler.total_steps,
            ),
            "interval": self.cfg.scheduler.interval,
            "frequency": self.cfg.scheduler.frequency,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        feat, attn_mask, target, target_length = batch
        N = feat.size(0)
        output = self(feat, attn_mask)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )
        loss = self.ctc_loss(
            output.permute(1, 0, 2), target, output_length, target_length
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, attn_mask, target, target_length = batch
        N = feat.size(0)
        output = self(feat, attn_mask)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )
        loss = self.ctc_loss(
            output.permute(1, 0, 2), target, output_length, target_length
        )

        target_sequences = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        predict_sequences = self.tokenizer.batch_decode(
            output.argmax(-1), skip_special_tokens=True
        )
        wer = cal_wer(target_sequences, predict_sequences)

        self.log("valid/loss", loss)
        self.log("valid/batch-wer", wer)

        return {"wer": wer, "loss": loss}

    def test_step(self, batch, batch_idx):
        feat, attn_mask, target, target_length = batch
        N = feat.size(0)
        output = self(feat, attn_mask)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )
        loss = self.ctc_loss(
            output.permute(1, 0, 2), target, output_length, target_length
        )

        target_sequences = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        predict_sequences = self.tokenizer.batch_decode(
            output.argmax(-1), skip_special_tokens=True
        )
        wer = cal_wer(target_sequences, predict_sequences)

        self.log("test/loss", loss)
        self.log("test/batch-wer", wer)

        return {"wer": wer, "loss": loss}

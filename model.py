import torch
from torch import nn
from transformers import get_scheduler
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import lightning as L
from utils import ENCODER_OUTPUT_LENGTH
from jiwer import wer as cal_wer


class WhisperCTC(nn.Module):
    def __init__(
        self, encoder_id: str = "tuanio/whisper-encoder.tiny.en", vocab_size: int = 47
    ):
        super().__init__()
        self.encoder = WhisperEncoder.from_pretrained(encoder_id)
        self.ctc_head = nn.Linear(self.encoder.config.d_model, vocab_size)

    def forward(self, **x):
        logits = self.ctc_head(self.encoder(**x).last_hidden_state)
        log_probs = nn.functional.log_softmax(logits)
        return log_probs


class WhisperModel(L.LightningModule):
    def __init__(self, cfg):
        self.model = WhisperCTC(**cfg.model)
        self.ctc_loss = nn.CTCLoss(blank=3)  # as in vocab json

    def forward(self, x: Tensor):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optim)
        scheduler = {
            "scheduler": get_scheduler(
                name=self.scheduler.name,
                optimizer=optimizer,
                num_warmup_steps=round(
                    self.scheduler.warmup_ratio * self.scheduler.total_steps
                ),
                num_training_steps=self.scheduler.total_steps,
            ),
            "interval": self.scheduler.interval,
            "frequency": self.scheduler.frequency,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        feat, target, target_length = batch
        N = feat.size(0)
        output = self(x)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )
        loss = self.ctc_loss(output, target, output_length, target_length)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, target, target_length = batch
        N = feat.size(0)
        output = self(x)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )

        loss = self.ctc_loss(output, target, output_length, target_length)

        target_sequences = tokenizer.batch_decode(target, skip_special_tokens=True)
        predict_sequences = tokenizer.batch_decode(
            output.argmax(-1), skip_special_tokens=True
        )
        wer = cal_wer(target_sequences, predict_sequences)

        self.log("valid/loss", loss)
        self.log("valid/batch-wer", wer)

        return {"wer": wer, "loss": loss}

    def test_step(self, batch, batch_idx):
        feat, target, target_length = batch
        N = feat.size(0)
        output = self(x)
        output_length = torch.full(
            size=(N,),
            fill_value=ENCODER_OUTPUT_LENGTH,
            dtype=torch.long,
            device=feat.device,
        )

        loss = self.ctc_loss(output, target, output_length, target_length)

        target_sequences = tokenizer.batch_decode(target, skip_special_tokens=True)
        predict_sequences = tokenizer.batch_decode(
            output.argmax(-1), skip_special_tokens=True
        )
        wer = cal_wer(target_sequences, predict_sequences)

        self.log("test/loss", loss)
        self.log("test/batch-wer", wer)

        return {"wer": wer, "loss": loss}

import torch
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from dataset import SpeechDataModule
from model import WhisperModel


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = SpeechDataModule(cfg.data_cfg)

    print("Change [total_steps] params.")
    cfg.model_cfg.scheduler.total_steps = (
        len(datamodule.train_dataloader()) * cfg.trainer_cfg.arguments.max_epochs
    )
    model = WhisperModel(cfg.model_cfg)

    logger = WandbLogger(**cfg.trainer_cfg.logger_wandb)

    trainer = pl.Trainer(**cfg.trainer_cfg.arguments, logger=logger)

    ckpt_path = None
    if cfg.experiment_cfg.ckpt.resume_ckpt:
        ckpt_path = cfg.experiment_cfg.ckpt.ckpt_path

    if cfg.experiment_cfg.train:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    if cfg.experiment_cfg.test:
        results = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        pd.DataFrame(results).to_csv("%s/test.csv" % trainer.log_dir)


if __name__ == "__main__":
    main()

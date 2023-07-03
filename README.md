# Whisper Encoder + CTC

## Install dependencies

```bash
apt install sox
apt-get install espeak-ng -y
pip install -r requirements.txt + sox
 ```

## Usage

```bash
python main.py
```

for more configuration, please refer to https://hydra.cc/docs/intro/


```bash
python finetuning.py \
    data_cfg.dataloader.batch_size=16 \
    experiment_cfg.train=True
```
# Prepare environment
Source instruction https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb

## Install UV and initialize virtual environment

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv .venv-fine-tuning --python 3.13 --seed
source .venv-fine-tuning/bin/activate
```

## Install dependencies

```shell
uv pip install -q -U bitsandbytes peft trl tensorboardX wandb
```

## Define huggingface token

```shell
export HF_TOKEN=""
```
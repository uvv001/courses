# Prepare environment
Source instruction https://huggingface.co/learn/agents-course/unit2/smolagents/introduction

## Install UV

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Install dependencies and initialize virtual environment

```shell
uv sync
```

## Define access tokens

```shell
export HF_TOKEN=""
export OPENROUTER_API_KEY=""
```

## Run script
```shell
# use default huggingface provider
uv run playlist.py
# use openrouter provider
uv run playlist.py --provider openrouter
```
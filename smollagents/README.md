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

## Define huggingface token

```shell
export HF_TOKEN=""
```
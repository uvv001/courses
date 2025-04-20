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

## Define .env file

```shell
HF_TOKEN=
OPENROUTER_API_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
OTEL_EXPORTER_OTLP_ENDPOINT=
```

## Run script
```shell
# use default huggingface provider
uv run playlist.py
# use openrouter provider
uv run playlist.py --provider openrouter
```
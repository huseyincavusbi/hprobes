# Installation

## Requirements

- Python 3.11+
- PyTorch 2.0+
- A HuggingFace causal language model

## From PyPI

```bash
pip install hprobes
```

```bash
uv add hprobes
```

## From Source

```bash
git clone https://github.com/huseyincavusbi/hprobes.git
cd hprobes
uv venv .venv
source .venv/bin/activate
uv sync --dev
```


## Verification

```python
from hprobes import HProbes
print("hprobes imported successfully")
```

## CLI

The `hprobes` CLI is included automatically:

```bash
hprobes --help
```

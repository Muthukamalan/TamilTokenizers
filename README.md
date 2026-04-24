# TamilTokenizer

TamilTokenizer is a byte-level BPE tokenizer project focused on Tamil text.  
It includes:
- A reusable tokenizer package (`minbpe`)
- A corpus preparation pipeline (`data/`)
- A training script to produce tokenizer artifacts (`train.py`)
- A FastAPI inference service (`server.py`)
- A Gradio visualization app (`app.py`)
- A Docker setup for API deployment


```mermaid
flowchart TD

subgraph group_core["Tokenizer core"]
  node_minbpe_init["API<br/>package export<br/>[__init__.py]"]
  node_minbpe_base["Base<br/>tokenizer core<br/>[base.py]"]
  node_minbpe_basic["Basic BPE<br/>tokenizer variant<br/>[basic.py]"]
  node_minbpe_regexs["Regex BPE<br/>tokenizer variant<br/>[regexs.py]"]
  node_minbpe_tests["Tests<br/>[test_tokenizer.py]"]
end

subgraph group_pipeline["Data pipeline"]
  node_download_html["Download HTML<br/>data ingest<br/>[download_html.py]"]
  node_corpus_create["Build Corpus<br/>data transform<br/>[corpus_create.py]"]
end

subgraph group_artifacts["Artifacts"]
  node_raw_data[("Raw data<br/>staging dir")]
  node_processed_data[("Processed data<br/>staging dir")]
  node_model_files[("Models<br/>artifact store")]
  node_regex_model["regex.model<br/>trained model<br/>[regex.model]"]
  node_regex_vocab["regex.vocab<br/>vocab artifact<br/>[regex.vocab]"]
end

subgraph group_entrypoints["Entrypoints"]
  node_train_py["Train<br/>orchestrator<br/>[train.py]"]
  node_app_py["App<br/>runtime entrypoint<br/>[app.py]"]
  node_server_py["Server<br/>runtime entrypoint<br/>[server.py]"]
end

subgraph group_tooling["Tooling"]
  node_tooling{{"Tooling"}}
end

node_minbpe_init -->|"exports"| node_minbpe_base
node_minbpe_init -->|"exports"| node_minbpe_basic
node_minbpe_init -->|"exports"| node_minbpe_regexs
node_minbpe_basic -->|"extends"| node_minbpe_base
node_minbpe_regexs -->|"extends"| node_minbpe_base
node_minbpe_tests -.->|"validates"| node_minbpe_init
node_download_html -->|"writes"| node_raw_data
node_raw_data -->|"feeds"| node_corpus_create
node_corpus_create -->|"writes"| node_processed_data
node_processed_data -->|"feeds"| node_train_py
node_train_py -->|"uses"| node_minbpe_init
node_train_py -->|"produces"| node_model_files
node_model_files -->|"contains"| node_regex_model
node_model_files -->|"contains"| node_regex_vocab
node_regex_model -->|"loads"| node_app_py
node_regex_vocab -->|"loads"| node_app_py
node_regex_model -->|"loads"| node_server_py
node_tooling -.->|"supports"| node_train_py
node_tooling -.->|"supports"| node_minbpe_tests

click node_minbpe_init "https://github.com/muthukamalan/tamiltokenizers/blob/main/minbpe/__init__.py"
click node_minbpe_base "https://github.com/muthukamalan/tamiltokenizers/blob/main/minbpe/base.py"
click node_minbpe_basic "https://github.com/muthukamalan/tamiltokenizers/blob/main/minbpe/basic.py"
click node_minbpe_regexs "https://github.com/muthukamalan/tamiltokenizers/blob/main/minbpe/regexs.py"
click node_minbpe_tests "https://github.com/muthukamalan/tamiltokenizers/blob/main/minbpe/test_tokenizer.py"
click node_download_html "https://github.com/muthukamalan/tamiltokenizers/blob/main/data/download_html.py"
click node_corpus_create "https://github.com/muthukamalan/tamiltokenizers/blob/main/data/corpus_create.py"
click node_raw_data "https://github.com/muthukamalan/tamiltokenizers/tree/main/data/raw"
click node_processed_data "https://github.com/muthukamalan/tamiltokenizers/tree/main/data/processed"
click node_train_py "https://github.com/muthukamalan/tamiltokenizers/blob/main/train.py"
click node_app_py "https://github.com/muthukamalan/tamiltokenizers/blob/main/app.py"
click node_server_py "https://github.com/muthukamalan/tamiltokenizers/blob/main/server.py"
click node_model_files "https://github.com/muthukamalan/tamiltokenizers/tree/main/models"
click node_regex_model "https://github.com/muthukamalan/tamiltokenizers/blob/main/models/regex.model"
click node_regex_vocab "https://github.com/muthukamalan/tamiltokenizers/blob/main/models/regex.vocab"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_minbpe_init,node_minbpe_base,node_minbpe_basic,node_minbpe_regexs,node_minbpe_tests toneBlue
class node_download_html,node_corpus_create toneAmber
class node_raw_data,node_processed_data,node_model_files,node_regex_model,node_regex_vocab toneMint
class node_train_py,node_app_py,node_server_py toneRose
class node_tooling toneIndigo
```

## Features

- Byte Pair Encoding (BPE) tokenization with custom merge training
- Tamil-specific regex chunking pattern (`[\u0B80-\u0BFF]+`) for training/encoding
- Save/load tokenizer artifacts (`.model`, `.vocab`)
- REST API endpoint for tokenization
- Interactive Gradio UI for token visualization
- Optional data acquisition and corpus generation from HTML sources

## Folder structure
```sh
.
├── app.py                  # Gradio tokenizer visualizer
├── data
│   ├── corpus_create.py
│   ├── download_html.py
│   ├── processed
│   └── raw
├── Dockerfile         # Containerized API runtime
├── EDA.ipynb
├── LICENSE
├── Makefile
├── minbpe
│   ├── base.py
│   ├── basic.py
│   ├── __init__.py
│   ├── regexs.py
│   └── test_tokenizer.py
├── models
│   ├── regex.model      # Serialized tokenizer merges/config
│   └── regex.vocab      # Human-readable vocabulary dump
├── pyproject.toml
├── README.md
├── requirements.txt      # Project metadata and dependencies (uv)
├── server.py             #  FastAPI API server (/encode)
└── train.py              # Tokenizer training script

6 directories, 19 files
```

## How It Works

### 1) Data Pipeline
1. `data/download_html.py` downloads source HTML files into `data/raw/`.
2. `data/corpus_create.py` parses HTML, extracts text with BeautifulSoup, and keeps Tamil Unicode range characters.
3. Final corpus is written to `data/processed/tamil_corpus.txt`.


### 2) Training
- `train.py` reads `data/processed/tamil_corpus.txt`
- Trains `RegexTokenizer` with `vocab_size=1000`
- Writes artifacts to `models/regex.model` and `models/regex.vocab`

### 3) Inference and Visualization
- `server.py` loads `models/regex.model` and exposes `POST /encode`
- `app.py` loads the same model and renders tokenized text in color via Gradio

## Requirements
- Python 3.10+ recommended
- [uv](https://docs.astral.sh/uv/) (recommended package manager), or `pip`
- Docker (optional, for containerized API)

> Note: `pyproject.toml` currently specifies `>=3.14`, but most tooling and dependencies are compatible with mainstream Python 3.10+ environments.

## Quick Start (Local)
### Option A: Using uv (recommended)
```bash
uv sync
```
Run API server:
```bash
uv run python server.py
```
Run Gradio app:
```bash
uv run python app.py
```

![Gradio UI](./assets/Tokenizer%20Vis.png)

### Option B: Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run API server:
```bash
python server.py
```
Run Gradio app:
```bash
python app.py
```

## API Usage

Start the server (`server.py`) and call:
- Endpoint: `POST /encode`
- Default URL: `http://localhost:8000/encode`

Request:
```json
{ "text": "ஆனந்த சிலை மனம் நெகிழ கண்டார்"}
```

Response (shape):
```json
{
  "token_ids": [ ... ],
  "token_details": [
    {
      "token_id": 123,
      "token_bytes": "b'...'",
      "token_text": "..."
    }
  ],
  "full_text": "..."
}
```

Example with curl:

```bash
curl -X POST "http://localhost:8000/encode" -H "Content-Type: application/json" -d '{"text":"ஆதி அந்தமில்லாத காலம்"}'
```

## Docker
Build image:
```bash
docker build -t tamil-tokenizer:latest .
```

Run container:
```bash
docker run --rm -p 8000:8000 tamil-tokenizer:latest
```
Test the API:
```bash
curl -X POST "http://localhost:8000/encode" \
  -H "Content-Type: application/json" \
  -d '{"text":"தமிழ் மொழி அழகு"}'
```
![Endpoint](./assets/FastAPI%20Endpoint.png)

## Training Your Own Model
If you want to retrain from fresh data:
1. Download raw HTML corpus:
   ```bash
   python data/download_html.py
   ```
2. Build processed corpus:
   ```bash
   python data/corpus_create.py
   ```
3. Train tokenizer and generate model files:
   ```bash
   python train.py
   ```

## Development Commands (Makefile)

```bash
make download-data    # fetch raw corpus sources
make get-corpous      # build processed corpus (spelling kept as in Makefile)
make train-tokenizer  # train tokenizer model
make build-image      # build Docker image (tag: tamil-tiktok:latest)
make gradio           # run Gradio app
make del-model        # remove files in models/
make clean            # remove caches and generated artifacts
```

## Python Usage Example

```python
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.load("./models/regex.model")

text = "சிந்தாமணி சிலப்பதிகாரம்"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
```

## Testing

Run tests with:

```bash
pytest -q
```

## Notes and Caveats

- `models/regex.model` must exist before running `server.py` or `app.py`.
- The tokenizer currently emphasizes Tamil Unicode block text splitting.
- Data download uses external sources and may fail due to site/network changes.




## License

This project is licensed under the terms in `LICENSE`.
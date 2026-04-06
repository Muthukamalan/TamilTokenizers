# BPE Tokenizers


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

## Folder structure
```sh
.
├── app.py
├── data
│   ├── corpus_create.py
│   ├── download_html.py
│   ├── processed
│   └── raw
├── Dockerfile
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
│   ├── regex.model
│   └── regex.vocab
├── pyproject.toml
├── README.md
├── requirements.txt
├── server.py
└── train.py

6 directories, 19 files
```
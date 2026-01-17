tunisian_rag_pipeline/
# Tunisian Heritage RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system for question-answering over Tunisian heritage data, leveraging a local Large Language Model (LLM) for private, high-quality responses.

## 🏗️ Project Structure

```
├── README.md              # Project documentation
├── scrapper.py            # Web scraper for data collection
├── Ai_Promptes_Caps/      # AI prompt capts
├── Architecture/          # Architecture documentation
├── tunisian_heritage_data/        # Heritage datasets
│   ├── dataset_index.json
│   ├── metadata/          # Metadata JSON files
│   ├── pdfs/              # PDF documents
│   ├── raw_html/          # Raw HTML files
│   └── texts/             # Text documents
├── tunisian_rag_pipeline/ # Main RAG pipeline
│   ├── chat.py            # Interactive chat interface
│   ├── requirements.txt    # Python dependencies
│   ├── test_retrieval.py   # Retrieval tests
│   ├── config/            # Configuration files
│   ├── scripts/           # Utility scripts (build, query, fine-tune, diagnose)
│   ├── src/               # Source code
│   │   ├── data/          # Data processing (chunking, ingestion, preprocessing)
│   │   ├── embeddings/    # Embedding models
│   │   ├── llm/           # LLM generation and prompts
│   │   ├── pipeline/      # RAG pipeline and intent detection
│   │   ├── retrieval/     # Vector store and retriever
│   │   └── utils/         # Helper utilities
│   ├── tests/             # Unit tests
│   └── vector_db/         # ChromaDB persistent storage
└── vector_db/             # Alternative vector database location
```

## 🚀 Quick Start(first u shood download an LLM Localy )

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the Vector Database
```bash
python scripts/build_vector_db.py
```

### 3. Query the System
```bash
python scripts/query.py "What caused the Tunisian revolution?"
```

### 4. Interactive Chat
```bash
python chat.py
```

## 📊 Features

- **Local LLM**: All answers are generated using a local Large Language Model for speed (no cloud required)
- **Multi-language Support**: English, French, Arabic
- **Intelligent Chunking**: Paragraph-aware, topic-aware splitting
- **Semantic Search**: Vector similarity with metadata filtering
- **Source Attribution**: Every answer includes supporting sources
- **Confidence Scoring**: Reliability indicators for answers
- **GPU Optimization**: Batched processing, 8-16GB VRAM friendly

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Embedding model selection
- Chunk sizes and overlap
- LLM parameters
- Retrieval settings

## 📖 Adding New Data

1. Place new files in `data/` folder
2. Run: `python scripts/build_vector_db.py --update`
3. The system will process only new documents

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📈 Diagnostics

```bash
python scripts/diagnose.py --full
```

## 🤖 Chat Interface

A simple command-line chat interface is provided via `chat.py` for interactive Q&A. All responses are generated locally, ensuring data privacy and low latency.

## License

MIT License
- **Source Attribution**: Every answer includes supporting sources

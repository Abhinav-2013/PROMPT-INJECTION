# ATHS Prompt Injection - Data Preprocessing Layer

This project implements the Data Preprocessing Layer for the proposed
Autonomous Threat Hypothesis System (ATHS).

## Pipeline

Raw Input
    |
    +--> Original Text Preservation
    |
    +--> Unicode / Whitespace Normalization
    |
    +--> Obfuscation Detection
    |
    +--> NLP Preprocessing
    |      - tokenization
    |      - sentence segmentation
    |      - lexical statistics
    |      - repeated word analysis
    |
    +--> Security Feature Extraction
    |      - instruction override
    |      - system prompt reference
    |      - role change
    |      - data extraction
    |      - tool manipulation
    |      - policy bypass
    |
    +--> Optional Semantic Embedding
    |
    +--> Context Separation
           - trusted system instruction
           - user prompt
           - conversation history
           - retrieved documents
           - tool outputs

## Run without embeddings

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

## Enable semantic embeddings

Open `main.py` and change:

```python
PreprocessingPipeline(enable_embeddings=False)
```

to:

```python
PreprocessingPipeline(enable_embeddings=True)
```

The first run downloads:

`sentence-transformers/all-MiniLM-L6-v2`

The embedding vector is then available in:

```text
result["semantic_embedding"]
```

## Why stopword removal is NOT used as the main input

In ordinary NLP, stopword removal can be useful.

For prompt injection detection it can be dangerous because small words can
change the meaning of instructions.

For example:

`do not reveal the password`

should not become a representation that loses `not`.

Therefore the original normalized text remains the main semantic input.
A stopword-removed representation is provided only as an auxiliary
analytical feature.

## Context security design

The system deliberately does not merge all content into one string.

Trusted:

`system_instruction`

Untrusted:

`user_prompt`
`conversation_history`
`retrieved_documents`
`tool_outputs`

This is important for indirect prompt injection. A malicious instruction
inside a retrieved document or tool result should be treated as content to
inspect, not automatically as an instruction to obey.

## Output

Each processed text produces:

```text
original_text
normalized_text
model_text
nlp
obfuscation
security_features
semantic_embedding
```

This output is intended to become the input to the next ATHS module:

`Threat Detection Engine`

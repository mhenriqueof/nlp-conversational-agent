# Conversational Agent
A multi-user conversational agent with persistent episodic memory, dynamic user profiling and
semantic retrieval. Built with Hugging Face Inference API, sentence-transformers, ChromaDB and Gradio.

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/mhenriqueof/conversational-agent)**


## Objective
The goal of this project was to build a conversational agent that goes beyond a simple chatbot by
integrating NLP concepts I have recently studied (RAG, LLM, HF) into a real, deployed system.

The project was also an opportunity to practice software engineering principles, including modular design,
object-oriented programming, clean code practices, structured documentation and proper Git workflow.


## Overview
Instead of relying solely on the LLM's context window for memory, the system utilizes:

* Retrieval-Augmented Generation (RAG) for episodic memory
* Vector embeddings for semantic similarity search
* Persistent ChromaDB collections for long-term storage
* LLM-based extraction for dynamic user profiling
* Per-user isolation for multi-user support

The agent remembers past interactions, builds a structured model of each user and adjusts responses
accordingly.


## How It Works

### Full Pipeline (per message)

```
User Message
     │
     ▼
Embed Message (sentence-transformers)
     │
     ▼
Retrieve Similar Episodes (ChromaDB)
     │
     ▼
Load User Profile (JSON)
     │
     ▼
Build Enriched Prompt (memory + profile)
     │
     ▼
Generate Response (HF Inference API)
     │
     ├──► Save Episode to Memory (ChromaDB)
     │
     └──► Extract & Update User Profile (LLM + JSON)
```

### Multi-User Flow

```
Username Input
     │
     ▼
Load user-specific ChromaDB collection
Load user-specific profile JSON
     │
     ▼
Scoped RAGPipeline instance
     │
     ▼
Isolated memory and profile per user
```


## Core Concepts

### Model
* **LLM:** `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Inference API
* **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers

### Embeddings
Each user message is converted into a high-dimensional vector using
**sentence-transformers**:

* Output: **384-dimensional L2-normalized vector**
* Runs locally, no API call required
* Enables semantic similarity between messages

### Episodic Memory (RAG)
Each conversation episode is stored in ChromaDB with its embedding:

$$
\text{episode} = \\{ \text{user\\_message}, \text{agent\\_response}, \vec{e}, \text{timestamp} \\}
$$

Where $\vec{e}$ is the embedding of the user message.

At each new interaction, the system retrieves the **top-k most similar past episodes** using
cosine similarity:

$$
\text{sim}(q, e_i) = \frac{\vec{q} \cdot \vec{e_i}}{|\vec{q}|\ |\vec{e_i}|}
$$

These episodes are injected into the prompt as context, enabling the agent to "remember".

### Dynamic User Profile
After each interaction, the LLM extracts structured information from the user's message:

```json
{
  "name": "Henrique",
  "interests": ["AI", "Math", "ESO"],
  "goals": ["become an AI engineer", "finish graduation"],
  "values": ["gratitude"]
}
```

New information is **merged** into the existing profile without overwriting previous data.
The profile is injected into every prompt, giving the agent a stable model of who it's talking to.

### Prompt Structure

```
[System Instructions]

[User Profile]
Name: Henrique
Interests: AI, Math, ESO,
Goals: become an AI engineer, finish graduation

[Relevant memories from past interactions]
- User: I love chess / Agent: Chess is a great game!
- User: I'm studying NLP / Agent: That's exciting!

User: <current message>
```


## Architecture

### Phase 1 - Chat with Vector Memory
* LLM Layer: Hugging Face Inference API (Qwen)
* Embedding Layer: sentence-transformers (local)
* Memory Layer: ChromaDB persistent storage
* Retrieval Layer: RAG pipeline
* UI: Gradio + HF Spaces deploy

### Phase 2 - Dynamic User Profile
* Pydantic schema for structured user model
* LLM-based information extraction
* JSON persistence across sessions
* Profile-aware prompt injection
* Real-time profile viewer in UI

### Phase 3 - Multi-User Support
* Username-based login screen
* Per-user ChromaDB collections
* Per-user profile JSON files
* Isolated sessions


## Project Structure
```
nlp-conversational-agent/
├── agent/
│   ├── llm/
│   │   ├── client.py   # HF Inference API client
│   │   └── prompt.py   # Prompt builder (memory + profile injection)
│   ├── memory/
│   │   ├── embedder.py # sentence-transformers embedding
│   │   └── store.py    # ChromaDB read/write (per-user)
│   ├── retrieval/
│   │   └── rag.py      # Full RAG pipeline (per-user)
│   ├── profile/
│   │   ├── schema.py   # Pydantic user profile + persistence
│   └── └── updater.py  # LLM-based profile extraction
├── app.py              # Gradio UI entry point
├── requirements.txt
└── README.md
```


## Installation

### Requirements
- Python 3.13
- Hugging Face account with API token (Write access)

### Clone the Repository
```bash
git clone https://github.com/mhenriqueof/nlp-conversational-agent.git
cd nlp-conversational-agent
```

### Create Virtual Environment
```bash
py -3.13 -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment
Create a `.env` file in the root folder:
```env
HF_TOKEN=your_huggingface_token_here
```

### Run Locally
```bash
py app.py
```


## Usage
1. Open the app in your browser
2. Enter a username to start your personal session
3. Start chatting, the agent will remember you across sessions
4. Watch your profile being built in real time on the right panel


## Known Limitations
* **HF Spaces free tier** - resets the filesystem on restart, clearing all memories and profiles
* **No authentication** - usernames are not password protected
* **Shared compute** - response time depends on HF Inference API availability


---

This project connected several fields I have been studying into something I am genuinely proud of. Building a system that remembers, learns and adapts was a deeply rewarding experience.

Thanks!

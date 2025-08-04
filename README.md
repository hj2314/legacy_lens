### Legacy Lens - Media Processing & RAG Implementation

![](https://i.imgur.com/Iay4pW3.png)

This repository demonstrates the core implementation of Legacy Lens, an AI-powered system that transforms scattered family photos and videos into an intelligent, searchable archive. The system consists of two parts: a media processing pipeline (`write_vault.py`) and a Retrieval-Augmented Generation (RAG) query system (`read_vault.py`).

---

#### Part 1: Media Processing Pipeline (`write_vault.py`)

*Input -> Processing -> Family Home Server*

This pipeline addresses the challenge that the most valuable context for family media often exists only in human memory. It combines irreplaceable human stories with AI visual analysis to create a rich, structured archive.

The system takes family photos, videos, and crucial human context (e.g., "Grandpa at 8 years old in Osaka with his friends") as input. Using a local AI model (Gemma 3n via Ollama) to ensure privacy, it analyzes the visual content. The AI is prompted with the human story to generate complementary, context-aware visual details, rather than generic captions. The output is a `vault.txt` file where each entry links the combined human story and AI analysis to the corresponding media file, creating a foundation for intelligent retrieval.

---

#### Part 2: RAG Query System (`read_vault.py`)

*Family Home Server <-> Query*

This system transforms the static archive into an interactive, conversational experience. The entries in `vault.txt` are converted into vector embeddings (`mxbai-embed-large`) to enable semantic search that goes beyond simple keywords.

When a user asks a natural language question in English or Japanese (e.g., "show me when grandpa was young"), the system uses cosine similarity to find the most relevant entries. This context is then fed to the local Gemma 3n model, which generates a conversational response, explains the media's relevance, and provides direct links to the photos or videos. The entire system, including media hosting, runs on a local web server to maintain complete privacy.

---

### Public Demo

A demonstration of this implementation will be available at **[TBD]**.

Please note that the public demo is hosted on a local server in Japan, and response times may take upwards of 5 to 10 seconds. For privacy purposes, the searchable content is limited to the specific media shown in the demo.

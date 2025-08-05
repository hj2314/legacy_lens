### Legacy Lens - Media Processing & RAG Implementation

![](https://i.imgur.com/Iay4pW3.png)

[Kaggle Gemma 3n contest details](https://www.kaggle.com/competitions/google-gemma-3n-hackathon)

[Video submission](https://youtu.be/VhhZRXfvW0U?si=V0rOfCxOTasE3ter)

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

### Implementation Notes & Scope

This implementation is a proof-of-concept designed to clearly demonstrate the core vision of Legacy Lens for the Gemma 3n Impact Challenge. The technical choices were made to highlight the "Privacy-First" and "Offline Ready" capabilities of running Gemma 3n locally with Ollama.
- Focused Example: To ensure a clear and reliable demonstration, the write_vault.py script uses a small, hardcoded set of media files. This allows for a perfect showcase of the "human + AI" context generation. The next step would be to build a file scanner that can automatically process an entire user-specified directory.
- Simplified Video Analysis: For efficiency, the system analyzes the first frame of video files. This is a strategic simplification to demonstrate multimodal capabilities without the overhead of full video processing, which could be implemented in a future version.
- No Public Demo: As a deliberate design choice reinforcing the project's privacy-first mission, there is no public-facing demo. The entire system is designed to run on a user's private network, ensuring family media is never exposed to the internet. The video demo shows the system in its intended, local environment.


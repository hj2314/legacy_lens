# Legacy Lens - Media Processing & RAG Implementation
**AI-Powered Family Archive: From Raw Media to Searchable Vault**

## About This Repository

This repository demonstrates simplified **core technical implementation** of Legacy Lens's media processing pipeline and RAG system. It showcases how we transform scattered family photos and videos into an intelligent, searchable archive that preserves both visual content and irreplaceable family stories.

The implementation consists of two complementary systems:

1. **`write_vault.py`** - The media ingestion and processing pipeline that creates our searchable archive
2. **`read_vault.py`** - The RAG-powered query system that enables natural language interaction with the archive

---

## Part 1: Media Processing Pipeline (`write_vault.py`)

### The Challenge: Bridging Human Memory and Digital Media

Family archives present a unique challenge: the most valuable information often exists only in human memory, not in the files themselves. A photo from 1950 might show "a group of boys," but only grandpa knows that he's "8 years old, on the far right, in Osaka with his childhood friends." Our processing pipeline solves this by combining **irreplaceable human context with AI visual understanding**.

### Architecture: From Scattered Media to Structured Knowledge

The system begins with raw inputs: family photos, video files, and crucially, the human stories that give them meaning. Each piece of media is processed through Ollama running Gemma 3n locally, ensuring complete privacy for sensitive family content.

**The Human Context Layer:** Before any AI processing begins, we capture the stories that only family members know. These aren't just metadata tags—they're rich narratives like "when grandpa was young he played catchball and caught fish with his friends" or "this is the first time Naoto walked on September 4th, 2024." This human knowledge forms the foundation that no AI system could recreate.

**Multimodal AI Analysis:** The system processes both images and videos through Gemma 3n's vision capabilities. For images, we convert them to base64 format for efficient processing. For videos, we extract representative frames to analyze. The AI doesn't just describe what it sees—it's specifically prompted with the human context to provide complementary visual details.

**Context-Aware Description Generation:** Rather than generic image captioning, our approach uses contextual prompting. When analyzing a 1950s photo, the system already knows it shows "grandpa at 8 years old in Osaka with friends," so it focuses on visual elements that enhance this story: "young boys in school uniforms standing against a tiled wall."

**Structured Output Generation:** Each processed entry combines three layers of information: the human story, the factual context, and the AI visual analysis. These are formatted into vault.txt entries that link descriptions to their corresponding media files, creating a rich foundation for later retrieval.

### The Innovation: Human-AI Collaboration for Heritage Preservation

This approach represents a fundamental shift from traditional digital asset management. Instead of relying solely on automated tagging or manual organization, we create a symbiotic relationship between human memory and AI analysis. The result is an archive that captures not just what's visible, but what's meaningful—preserving the stories that make family media truly valuable.

---

## Part 2: RAG Query System (`read_vault.py`)

### From Static Archive to Interactive Conversation

The second phase transforms our enriched vault.txt into a living, queryable system. This isn't just keyword search—it's semantic understanding that allows family members to ask natural questions and receive contextually appropriate responses.

### Semantic Understanding Through Vector Embeddings

The system begins by converting vault.txt entries into high-dimensional vector representations using the mxbai-embed-large model. These embeddings capture semantic relationships that simple keyword matching would miss. When someone asks "show me when grandpa was young," the system understands this relates to childhood photos, even if those exact words don't appear in the descriptions.

**Intelligent Caching Architecture:** Since embedding generation is computationally expensive, the system implements a sophisticated caching mechanism. Once embeddings are created for vault entries, they're stored locally and only regenerated when content changes. This ensures responsive performance for family members using the system.

**Cosine Similarity Search:** When a query comes in, the system converts it into the same vector space and uses cosine similarity to find the most semantically relevant vault entries. This mathematical approach to relevance ensures that queries like "grandpa's childhood" successfully surface photos from the 1950s, even with varied language or phrasing.

### RAG-Powered Natural Language Responses

The retrieved context is then fed to Gemma 3n running locally through Ollama. The system is designed with family relationships in mind—it knows when someone is asking about grandchildren versus grandparents and adjusts its responses accordingly.

**Context-Enriched Generation:** Rather than just returning search results, the system generates natural language responses that feel conversational. It can explain why a particular photo or video is relevant to the query, drawing connections between the human stories and visual content we captured during processing.

**Multilingual Heritage Support:** Recognizing that many families span multiple cultures and languages, the system supports queries and responses in both English and Japanese. This ensures that cultural nuances and authentic family expressions are preserved rather than lost in translation.

**Media-Rich Responses:** Beyond text responses, the system returns direct links to the relevant media files. Users don't just read about "Naoto's first steps"—they see the actual video, creating an immediate emotional connection to the preserved memory.

### Web Interface and Local Privacy

The system serves both the query interface and media files through a local web server, ensuring that sensitive family content never leaves the family's device. The RESTful API design makes it easy to build rich web interfaces while maintaining the privacy-first architecture that families require.

**Responsive Media Serving:** The server intelligently handles different media types—photos, videos, and audio files—with appropriate MIME types and security measures. Path traversal attacks are prevented while ensuring legitimate family content is easily accessible.

**Real-Time Query Processing:** The asynchronous architecture ensures that even complex semantic searches and AI generation don't block the user interface. Family members experience responsive interactions that feel natural and immediate.

---

## Technical Architecture Philosophy

### Privacy as a Core Design Principle

Every technical decision prioritizes family privacy. By running Gemma 3n through Ollama locally, processing media files on the family's own hardware, and serving content through localhost, we ensure that intimate family moments remain truly private. No external APIs, no cloud dependencies—just local AI serving local families.

### Scalability Through Simplicity

The vault.txt format provides an elegant balance between simplicity and power. It's human-readable for manual review and editing, yet structured enough to support sophisticated RAG operations. This design allows families to start with basic functionality and expand to more complex vector databases and search systems as their needs grow.

### Cultural and Temporal Sensitivity

The system recognizes that family archives span generations, cultures, and languages. Rather than imposing a single organizational structure, it preserves the authentic voice of family stories while making them searchable and accessible to future generations.

---

## From Implementation to Legacy

This technical implementation represents more than just code—it's a bridge between human memory and digital preservation. By combining local AI capabilities with thoughtful human-computer interaction design, we create tools that don't just organize files, but preserve the stories that make them meaningful.

The result is a system where a grandparent can naturally ask "show me when I was young" and immediately see not just photos, but the stories that bring those moments to life. It's where future generations can maintain connection with ancestors through natural conversation, accessing both visual memories and the voices that explain their significance.

## Public Demo

A complete demonstration of this technical implementation will be available at **[TBD]**, showcasing how natural language queries retrieve contextually relevant family media through the AI-enhanced vault system described here.

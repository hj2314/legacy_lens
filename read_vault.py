"""
A backend server for a Retrieval-Augmented Generation (RAG) chat application.

This script performs the following functions:
1.  Initializes and loads a vector database (embeddings) from a text file ('vault.txt').
    It uses a cache ('embedding_cache.json') to speed up startup.
2.  Implements a RAG pipeline:
    - Takes a user query.
    - Finds relevant documents from the vault using semantic search.
    - Injects the relevant context into a prompt for a large language model (LLM).
    - Gets a response from the LLM (powered by Ollama).
3.  Serves a simple web interface ('index.html') and media files.
4.  Provides an API endpoint ('/api/query') to handle chat queries from the web interface.
5.  Parses responses to extract text captions and associated media file URLs.
"""
import torch
import ollama
import os
import asyncio
from openai import OpenAI
import json
import re
import http.server
import socketserver
import urllib.parse
import mimetypes

# --- ANSI escape codes for colored console output ---
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


# --- Configuration ---
class Config:
    """Holds all static configuration variables for the application."""
    # Base directory of the project.
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    # AI model names used for language generation and embedding.
    MODEL_LLM = "gemma3n"
    EMBED_MODEL = "mxbai-embed-large"

    # --- Paths relative to the project directory ---
    # The source file with text entries.
    VAULT_FILE = os.path.join(PROJECT_DIR, "vault.txt")
    # Cache file to store pre-computed embeddings for faster startup.
    CACHE_FILE = os.path.join(PROJECT_DIR, "embedding_cache.json")
    # Directory where media files (images, videos) are stored.
    MEDIA_BASE_PATH = os.path.join(PROJECT_DIR, "data")

    # API endpoint for the local Ollama server.
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    # Number of top relevant documents to retrieve for context.
    TOP_K = 1 
    # Default port for the web server.
    SERVER_PORT = 8000
    # Directory to serve web files from (e.g., index.html).
    WEB_BASE_PATH = PROJECT_DIR


# --- Global variables ---
# These are initialized once by initialize_app() and used throughout.
client = None
vault_content = None
vault_embeddings = None
system_message = ""
conversation_history = []


def extract_media_links(text):
    """
    Extracts a list of media file links from a string using the [[link1, link2]] format.

    Args:
        text (str): The text containing the media link block.

    Returns:
        list[str]: A list of cleaned media file names, or an empty list if none are found.
    """
    match = re.search(r"\[\[([^\]]+)\]\]", text)
    if not match:
        return []
    # Extracts the content inside [[...]]
    links_str = match.group(1)
    # Splits by comma and strips whitespace/quotes from each link.
    return [link.strip().strip("\"'") for link in links_str.split(",")]


def convert_path_to_http_url(path):
    """
    Converts a local file path into a web-accessible URL under the /media/ route.

    This allows the frontend to request media files from the server.

    Args:
        path (str): The local file path (e.g., 'data/grandpa.jpg').

    Returns:
        str: A URL-encoded path (e.g., '/media/grandpa.jpg').
    """
    try:
        file_name = os.path.basename(path)
        # URL-encode the filename to handle spaces and special characters safely.
        encoded_path = urllib.parse.quote(file_name)
        return f"/media/{encoded_path}"
    except Exception:
        # Return the original path if it's not a processable file path.
        return path


def load_vault_content(filepath):
    """
    Loads the content from the vault file into a list of strings.

    Args:
        filepath (str): The path to the vault.txt file.

    Returns:
        list[str]: A list where each element is a line from the file.
    """
    if not os.path.exists(filepath):
        print(f"{YELLOW}Warning: {filepath} not found. Creating empty vault.{RESET}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        # Reads all lines, stripping whitespace, and ignoring any empty lines.
        return [line.strip() for line in f if line.strip()]


def load_or_create_embeddings(content_list, cache_path, embed_model):
    """
    Loads embeddings from a cache or generates them if they don't exist.

    This function checks a JSON cache file first. For any content not in the cache,
    it generates a new embedding using the specified Ollama model and updates the cache.

    Args:
        content_list (list[str]): The list of documents to get embeddings for.
        cache_path (str): Path to the JSON file used for caching embeddings.
        embed_model (str): The name of the embedding model to use.

    Returns:
        torch.Tensor: A tensor containing the embeddings for all content.
    """
    cached_embeddings = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        cached_embeddings = {item["content"]: item["embedding"] for item in cache}

    new_embeddings, embeddings_to_cache = [], []
    for content in content_list:
        if content in cached_embeddings:
            new_embeddings.append(cached_embeddings[content])
        else:
            # Clean the content by removing media links before generating the embedding.
            clean_content = re.sub(r"\[\[([^\]]+)\]\]", "", content).strip()
            print(f"Generating new embedding for: '{clean_content[:50]}...'")
            response = ollama.embeddings(model=embed_model, prompt=clean_content)
            embedding = response["embedding"]
            new_embeddings.append(embedding)
            embeddings_to_cache.append({"content": content, "embedding": embedding})

    # If new embeddings were generated, update the cache file.
    if embeddings_to_cache:
        existing_cache = list(cached_embeddings.values())
        all_cached_data = embeddings_to_cache + [{"content": k, "embedding": v} for k, v in cached_embeddings.items()]
        # To avoid duplicates, create a unique set before writing
        unique_cache = {item['content']: item for item in all_cached_data}
        with open(cache_path, "w") as f:
            json.dump(list(unique_cache.values()), f)

    return torch.tensor(new_embeddings) if new_embeddings else torch.empty(0)


def get_relevant_context(query, vault_embeddings, vault_content, top_k):
    """
    Finds the most relevant documents from the vault based on a user query.

    It generates an embedding for the query and computes the cosine similarity
    against all pre-computed document embeddings to find the best matches.

    Args:
        query (str): The user's input query.
        vault_embeddings (torch.Tensor): The tensor of all document embeddings.
        vault_content (list[str]): The list of all document texts.
        top_k (int): The number of top results to return.

    Returns:
        list[str]: A list of the most relevant document texts.
    """
    if vault_embeddings.nelement() == 0:
        return [] # Return early if the vault is empty.
    
    query_embedding = torch.tensor(
        ollama.embeddings(model=Config.EMBED_MODEL, prompt=query)["embedding"]
    ).unsqueeze(0)
    
    # Calculate cosine similarity between the query and all vault entries.
    cos_scores = torch.cosine_similarity(query_embedding, vault_embeddings)
    
    # Ensure we don't request more items than available.
    k = min(top_k, len(cos_scores))
    if k == 0:
        return []
        
    # Get the indices of the top k scores.
    top_indices = torch.topk(cos_scores, k=k)[1].tolist()
    
    return [vault_content[idx] for idx in top_indices]


def parse_response_for_media(response_text):
    """
    Parses the LLM's response to separate the text caption from media file paths.

    Assumes the first line is the caption and subsequent lines might contain file paths.

    Args:
        response_text (str): The full response from the language model.

    Returns:
        tuple[str, str | None]: A tuple containing the caption and a single media URL, if found.
    """
    lines = [l.strip() for l in response_text.splitlines() if l.strip()]
    if not lines:
        return "", None
    
    caption = lines[0]
    media_url = None
    
    # Check if the second line is a potential file path.
    if len(lines) > 1 and (lines[1].startswith(("http://", "https://")) or os.path.isabs(lines[1])):
        media_url = lines[1]
        
    # As a fallback, check for [[...]] links within the entire response.
    if not media_url:
        media_links = extract_media_links(response_text)
        if media_links:
            media_url = media_links[0]
            
    return caption, media_url


async def chat_with_context(query, system_message, vault_embeddings, vault_content, model, history, client):
    """
    Core RAG function to get a contextual response from the LLM.

    Args:
        query (str): The user's query.
        system_message (str): The system prompt to guide the LLM.
        vault_embeddings (torch.Tensor): All document embeddings.
        vault_content (list[str]): All document texts.
        model (str): The name of the LLM to use.
        history (list): The conversation history.
        client (OpenAI): The client for communicating with the LLM.

    Returns:
        tuple[str, list[str]]: The LLM's response and a list of media links from the context.
    """
    # 1. Retrieve relevant context from the vault.
    relevant_context = get_relevant_context(
        query, vault_embeddings, vault_content, Config.TOP_K
    )
    
    # 2. Construct the prompt with the retrieved context.
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print(f"{CYAN}Context from documents:\n{context_str}{RESET}\n")
        query_with_context = f"Using the following context, answer the user's query.\n\nContext:\n{context_str}\n\nUser Query: {query}"
    else:
        print(f"{CYAN}No relevant context found.{RESET}")
        query_with_context = f"Answer the user's query. You have no context to work from.\n\nUser Query: {query}"
        
    history.append({"role": "user", "content": query_with_context})
    messages = [{"role": "system", "content": system_message}] + history
    
    # 3. Call the LLM with the complete prompt.
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
    )
    assistant_response = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_response})
    
    # 4. Extract any media links from the context that was used.
    context_media_links = []
    if relevant_context:
        for ctx in relevant_context:
            context_media_links.extend(extract_media_links(ctx))
            
    return assistant_response, context_media_links


async def process_query(query):
    """
    Top-level handler for processing a single user query and returning a structured response.

    Args:
        query (str): The user's raw query from the frontend.

    Returns:
        dict: A dictionary containing the caption and media URLs for the frontend.
    """
    global conversation_history
    print(f"{YELLOW}User Query: {query}{RESET}")
    
    # Get the AI's response and any media links from the retrieved context.
    response, context_media_links = await chat_with_context(
        query,
        system_message,
        vault_embeddings,
        vault_content,
        Config.MODEL_LLM,
        conversation_history,
        client,
    )
    
    main_media_url, pip_media_url = None, None
    if response.strip() == "分かりません。": # "I don't know."
        result = {"caption": "分かりません。", "main_media_url": None, "pip_media_url": None}
    else:
        caption, response_media_url = parse_response_for_media(response)
        
        # Prioritize media links found in the original context.
        media_paths = context_media_links or ([response_media_url] if response_media_url else [])
        
        # Convert file paths to web-accessible URLs.
        if media_paths:
            main_media_url = convert_path_to_http_url(media_paths[0])
            if len(media_paths) > 1:
                pip_media_url = convert_path_to_http_url(media_paths[1])
                
        result = {
            "caption": caption,
            "main_media_url": main_media_url,
            "pip_media_url": pip_media_url,
        }
        
    print(f"{GREEN}Response sent to client: {result}{RESET}")
    return result


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler to serve web files and handle API requests.
    """
    def __init__(self, *args, **kwargs):
        # Serve files from the WEB_BASE_PATH directory.
        super().__init__(*args, directory=Config.WEB_BASE_PATH, **kwargs)

    def do_GET(self):
        """Handles GET requests for both web files and media files."""
        # Route for serving media files.
        if self.path.startswith("/media/"):
            # Decode the URL path to get the actual filename.
            media_filename = urllib.parse.unquote(self.path[len("/media/") :])
            full_path = os.path.join(Config.MEDIA_BASE_PATH, media_filename)

            # Security check: prevent directory traversal attacks.
            if not os.path.abspath(full_path).startswith(os.path.abspath(Config.MEDIA_BASE_PATH)):
                self.send_error(403, "Forbidden")
                return

            try:
                with open(full_path, "rb") as f:
                    self.send_response(200)
                    mimetype, _ = mimetypes.guess_type(full_path)
                    self.send_header("Content-type", mimetype or "application/octet-stream")
                    fs = os.fstat(f.fileno())
                    self.send_header("Content-Length", str(fs[6]))
                    self.end_headers()
                    self.copyfile(f, self.wfile)
            except FileNotFoundError:
                self.send_error(404, "File Not Found")
            return
        
        # Route for the main web page.
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        """Handles POST requests to the /api/query endpoint."""
        if self.path == "/api/query":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data)
            query = body.get("query")
            
            if not query:
                self.send_response(400, "Bad Request")
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Query is required"}).encode("utf-8"))
                return
            
            # Process the query and get the result.
            result = asyncio.run(process_query(query))
            
            # Send the JSON response back to the client.
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def initialize_app():
    """Initializes all global components of the application before starting the server."""
    global client, vault_content, vault_embeddings, system_message, conversation_history
    print(f"{GREEN}Initializing application...{RESET}")
    
    # Set up the client to communicate with the local Ollama instance.
    client = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key="ollama")
    
    # Load the text documents from the vault file.
    vault_content = load_vault_content(Config.VAULT_FILE)
    
    # Load or generate embeddings for the vault content.
    vault_embeddings = load_or_create_embeddings(
        vault_content, Config.CACHE_FILE, Config.EMBED_MODEL
    )
    
    # Define the system message that guides the LLM's behavior.
    system_message = """If there's a question about grandparents, you are talking to grandkids. If there's a question about grandkids, you are talking to grandparents. All documents are information[[VIDEO IMAGE OR AUDIO LINK (if applicable)]]. Each row represents a single document/fact.

When responding:
1.  **Prioritize the provided context first.** If the answer is in the context, give a very short one-sentence answer in Japanese on the first line.
2.  If there's a relevant media file from the context, provide just the file path on the second line.
3.  **If the answer is NOT in the context, use your general knowledge to answer the user's question.**
5.  Do not include [[]] brackets in your response.
6.  Mention the date if possible like when the photo was taken.
"""
    # Initialize an empty conversation history.
    conversation_history = []
    print(f"{GREEN}Initialization complete.{RESET}")


def find_free_port(start_port=8000):
    """
    Finds an available TCP port on the local machine to avoid port conflicts.

    Args:
        start_port (int): The port number to start searching from.

    Returns:
        int: The first available port number.
    """
    import socket
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue # Port is already in use.
    raise OSError("No free ports found")


if __name__ == "__main__":
    # 1. Initialize the application state.
    initialize_app()
    handler = MyHttpRequestHandler
    
    try:
        # 2. Find an available port and start the web server.
        port = find_free_port(Config.SERVER_PORT)
        httpd = socketserver.TCPServer(("", port), handler)
        httpd.allow_reuse_address = True
        
        print(f"{GREEN}Server started at http://localhost:{port}{RESET}")
        print(f"Serving web files from: {Config.WEB_BASE_PATH}")
        print(f"Serving media files from: {Config.MEDIA_BASE_PATH} under the /media/ path")
        print(f"Open http://localhost:{port} in your browser.")
        
        # 3. Keep the server running until manually stopped.
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Server shutting down...{RESET}")
    finally:
        if 'httpd' in locals():
            httpd.server_close()

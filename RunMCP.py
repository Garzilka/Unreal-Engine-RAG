import os
import json
import logging
import sys
import argparse
import asyncio
import faiss
import torch
from sentence_transformers import SentenceTransformer
from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("rag-mcp")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _read_file_sync(full_path: str, relative_path: str, source_dir: str) -> str:
    """Синхронное чтение файла с fallback по кодировкам. Запускается в потоке."""
    # Защита от Directory Traversal
    if not full_path.startswith(source_dir + os.sep) and full_path != source_dir:
        return "Error: Access denied. Path is outside of project directory."

    if not os.path.exists(full_path):
        return f"Error: File {relative_path} not found."

    # Fallback по кодировкам — как в билдере
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            with open(full_path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue

    # Если все кодировки упали — читаем игнорируя ошибки
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Run RAG MCP server")
    parser.add_argument("--server-name", required=True,
                        help="MCP server name, e.g. UE_RAG_MCP")
    parser.add_argument("--base-name", required=True,
                        help="Base file name for index and metadata, e.g. UE58 "
                             "(files: {base}_faiss.index, {base}_chunks.jsonl)")
    parser.add_argument("--input-dir", required=True,
                        help="Путь к корневой директории с исходным кодом проекта для чтения файлов")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind the server (default: 8765).")
    args = parser.parse_args()

    server_name = args.server_name
    base_name = args.base_name
    port = args.port
    source_dir = os.path.normpath(args.input_dir)

    index_file = f"{base_name}_faiss.index"
    metadata_file = f"{base_name}_chunks.jsonl"

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        raise SystemExit(
            f"[RAG] Индекс не найден ({index_file} / {metadata_file}).\n"
            f"Сначала запустите: python3 build_index.py --output-name {base_name}"
        )

    if not os.path.isdir(source_dir):
        raise SystemExit(f"[RAG] Указанная директория исходников не существует: {source_dir}")

    logger.info(f"Loading FAISS index: {index_file}")
    index = faiss.read_index(index_file)

    chunks = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info(f"Loaded: {index.ntotal} chunks, {len(chunks)} metadata")

    torch.set_num_threads(4)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    mcp = MCPServer(server_name)
    tool_name = f"search_{base_name.lower()}"

    def _search(query: str) -> str:
        logger.info(f"[{server_name}] Search: {query}")
        query_text = QUERY_PREFIX + query
        with torch.inference_mode():
            emb = embedder.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

        distances, indices = index.search(emb, TOP_K)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(chunks):
                continue
            c = chunks[idx]
            results.append({
                "file": c["file"],
                "score": float(dist),
                "snippet": c.get("text", ""),
            })
        return json.dumps(results, ensure_ascii=False)

    @mcp.tool(name="read_full_file")
    async def read_full_file(relative_path: str) -> str:
        """Получить полное содержимое файла кодовой базы по его относительному пути."""
        full_path = os.path.normpath(os.path.join(source_dir, relative_path))
        return await asyncio.to_thread(_read_file_sync, full_path, relative_path, source_dir)

    @mcp.tool(name=tool_name)
    async def rag_search(query: str) -> str:
        """Search the RAG knowledge base associated with this MCP server."""
        if not query.strip():
            return "Error: empty query"
        result = await asyncio.to_thread(_search, query)
        return result

    logger.info(f"Starting MCP server '{server_name}' (tool: {tool_name}) on port {port}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port,
            stateless_http=True, json_response=True)


if __name__ == "__main__":
    main()

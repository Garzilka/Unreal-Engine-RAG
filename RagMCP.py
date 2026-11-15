import os
import sys
import gc
import json
import logging
import argparse
import resource
import torch
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ue-rag-builder")

# ================= НАСТРОЙКИ =================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_CODE_PROMPT = "Represent the C++ source code for semantic search: "
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
MAX_FILE_SIZE = 5_000_000

ALLOWED_EXTENSIONS = {".h", ".hpp", ".cpp", ".cc", ".md", ".cs"}
IGNORE_DIRS = {
    "Intermediate", "Binaries", "Saved", ".git", "Images", "img",
    "Content", "ThirdParty", "obj", "Debug", "Release",
    "Build", "DerivedDataCache"
}
# =============================================

# ---- Определение устройства ----
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
BATCH_SIZE = 256 if USE_GPU else 64
logger.info(f"Device: {DEVICE}, batch_size: {BATCH_SIZE}")


def mem_info():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return f"{usage / 1024:.0f} MB"


def vram_info():
    if not USE_GPU:
        return "N/A (CPU mode)"
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return f"{allocated:.0f} MB allocated, {reserved:.0f} MB reserved"


def extract_includes(text):
    includes = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#include") and not stripped.endswith(".generated.h\""):
            includes.add(stripped)
    return sorted(list(includes))


def is_noise_cpp(text, rel_path):
    """Пропускаем авто-сгенерированные и пустые .cpp"""
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    if "IMPL_MODULE" in stripped and len(stripped) < 500:
        return True
    meaningful_lines = [
        l for l in stripped.splitlines()
        if l.strip()
        and not l.strip().startswith("#")
        and l.strip() != "{"
        and l.strip() != "}"
        and l.strip() != "//"
    ]
    if len(meaningful_lines) < 10:
        return True
    return False


def split_ue_code_into_chunks(text, rel_path):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    file_context = f"// File: {rel_path}\n"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if "GENERATED_BODY()" in stripped or "GENERATED_UCLASS_BODY()" in stripped:
            i += 1
            continue

        group = [line]

        if stripped.startswith(("UPROPERTY(", "UFUNCTION(", "UCLASS(", "USTRUCT(")):
            # --- Собираем макрос по балансу скобок ---
            depth = stripped.count("(") - stripped.count(")")
            j = i

            # Пока скобки не сбалансированы — берём следующие строки
            while j + 1 < len(lines) and depth > 0:
                j += 1
                group.append(lines[j])
                depth += lines[j].count("(") - lines[j].count(")")

            # --- Ищем объявление после закрытия скобок ---
            close_line = lines[j].rstrip()
            paren_idx = close_line.find(")")

            # Если на строке закрытия есть текст после ')' — объявление уже здесь
            if 0 <= paren_idx < len(close_line) - 1:
                pass  # объявление захвачено в составе group
            else:
                # ')' в конце строки или depth ушёл в минус — ищем объявление ниже
                while j + 1 < len(lines):
                    j += 1
                    nxt = lines[j].strip()
                    group.append(lines[j])
                    if nxt.endswith(";") or nxt.endswith("{"):
                        break
                    # Если наткнулись на новый макрос или пустую строку после пустой — стоп
                    if not nxt and len(group) > 1 and lines[j - 1].strip() == "":
                        group.pop()
                        j -= 1
                        break

            i = j

        group_text = "\n".join(group)
        group_size = len(group_text)

        if current_size + group_size > CHUNK_SIZE and current_chunk:
            chunks.append(file_context + "\n".join(current_chunk))
            # Overlap в символах с выравниванием по границе строки
            overlap_str = "\n".join(current_chunk)[-CHUNK_OVERLAP:]
            overlap_lines = overlap_str.splitlines()
            # Отбрасываем первую неполную строку
            if len(overlap_lines) > 1 and "\n".join(current_chunk)[-CHUNK_OVERLAP - 1:] != "\n" + overlap_lines[0]:
                overlap_lines = overlap_lines[1:] if len(overlap_lines) > 1 else overlap_lines
            current_chunk = overlap_lines + group
            current_size = sum(len(l) for l in current_chunk)
        else:
            current_chunk.extend(group)
            current_size += group_size

        i += 1

    if current_chunk:
        chunk_text = file_context + "\n".join(current_chunk)
        # Фильтруем пустые чанки (только file_context, без тела)
        if chunk_text.strip() != file_context.strip():
            chunks.append(chunk_text)

    return chunks


def read_file_with_fallback(path):
    """Безопасное чтение файлов с поддержкой разных кодировок"""
    for encoding in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def scan_files(root_dir):
    paths = []
    big_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() not in ALLOWED_EXTENSIONS:
                continue
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
                if size > MAX_FILE_SIZE:
                    big_files.append((full, size))
                    continue
                paths.append(full)
            except OSError:
                pass
    if big_files:
        for path, size in sorted(big_files, key=lambda x: -x[1])[:10]:
            logger.warning(f"  {size:>10,} bytes  {path}")
    return paths


def phase1_chunk_files(paths, chunks_temp_path):
    logger.info(f"Phase 1: chunking {len(paths)} files...")
    count = 0
    common_dir = os.path.dirname(os.path.commonpath(paths)) if len(paths) > 1 else ""

    with open(chunks_temp_path, "w", encoding="utf-8") as fp:
        for i, path in enumerate(paths):
            text = read_file_with_fallback(path)
            rel_path = os.path.relpath(path, common_dir) if common_dir else os.path.basename(path)

            # Пропускаем мусорные .cpp
            if rel_path.endswith(".cpp") and is_noise_cpp(text, rel_path):
                continue

            includes = extract_includes(text)
            if includes:
                include_text = f"// File: {rel_path}\n// Module Includes:\n" + "\n".join(includes)
                fp.write(json.dumps({"file": rel_path, "text": include_text}, ensure_ascii=False) + "\n")
                count += 1

            file_chunks = split_ue_code_into_chunks(text, rel_path)
            for c in file_chunks:
                fp.write(json.dumps({"file": rel_path, "text": c}, ensure_ascii=False) + "\n")
                count += 1

            if (i + 1) % 2000 == 0:
                fp.flush()
                gc.collect()
                logger.info(f"Progress: {i+1}/{len(paths)} files, {count} chunks, RAM: {mem_info()}")

    logger.info(f"Phase 1 done: {count} chunks, peak RAM: {mem_info()}")
    return count


def encode_batch(embedder, texts):
    """Кодирует батч текстов с BGE-промптом, возвращает numpy-массив."""
    texts_with_prompt = [BGE_CODE_PROMPT + t for t in texts]
    with torch.inference_mode():
        embs = embedder.encode(
            texts_with_prompt,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return embs


def phase2_build_index(total_chunks, chunks_temp_path, metadata_path, index_path):
    logger.info("Phase 2: embeddings + FAISS...")
    logger.info(f"GPU: {USE_GPU}, device: {DEVICE}")

    if not USE_GPU:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    dim = embedder.get_embedding_dimension()

    if USE_GPU:
        embedder.half()
        logger.info(f"Model loaded in fp16 on GPU. VRAM: {vram_info()}")
    else:
        logger.info(f"Model loaded on CPU. RAM: {mem_info()}")

    index = faiss.IndexFlatIP(dim)
    processed = 0

    with open(chunks_temp_path, "r", encoding="utf-8") as fin, \
         open(metadata_path, "w", encoding="utf-8") as fout:

        batch_texts = []
        batch_meta = []

        for line in fin:
            chunk = json.loads(line)
            batch_texts.append(chunk["text"])
            batch_meta.append(chunk)

            if len(batch_texts) >= BATCH_SIZE:
                embs = encode_batch(embedder, batch_texts)
                index.add(embs.astype("float32"))
                del embs

                for m in batch_meta:
                    fout.write(json.dumps(m, ensure_ascii=False) + "\n")

                processed += len(batch_texts)
                batch_texts.clear()
                batch_meta.clear()

                if processed % 5000 < BATCH_SIZE:
                    if USE_GPU:
                        logger.info(f"Embedded: {processed}/{total_chunks}, VRAM: {vram_info()}")
                        torch.cuda.empty_cache()
                    else:
                        logger.info(f"Embedded: {processed}/{total_chunks}, RAM: {mem_info()}")
                    gc.collect()

        if batch_texts:
            embs = encode_batch(embedder, batch_texts)
            index.add(embs.astype("float32"))
            del embs
            for m in batch_meta:
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            processed += len(batch_texts)

    logger.info(f"Saving final FAISS index to {index_path}...")
    faiss.write_index(index, index_path)

    if os.path.exists(chunks_temp_path):
        os.remove(chunks_temp_path)

    if USE_GPU:
        logger.info(f"Phase 2 done: {processed} chunks indexed. Peak VRAM: {vram_info()}")
    else:
        logger.info(f"Phase 2 done: {processed} chunks indexed. Final RAM: {mem_info()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS RAG index for Unreal Engine source code")
    parser.add_argument("--input-dir", required=True, help="Path to directory with UE source files")
    parser.add_argument("--output-name", required=True, help="Base name for output files")
    args = parser.parse_args()

    input_dir = args.input_dir
    base_name = args.output_name

    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    index_file = f"{base_name}_faiss.index"
    metadata_file = f"{base_name}_chunks.jsonl"
    chunks_temp = f"{base_name}_raw.jsonl"

    logger.info(f"=== STARTING RAG INDEX BUILD ===")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output index: {index_file}")
    logger.info(f"Output metadata: {metadata_file}")
    logger.info(f"Device: {DEVICE}")

    paths = scan_files(input_dir)
    total = phase1_chunk_files(paths, chunks_temp)
    if total > 0:
        phase2_build_index(total, chunks_temp, metadata_file, index_file)
        logger.info("=== SUCCESS: Index ready for MCP Server ===")
    else:
        logger.error("No chunks generated. Check your input directory path.")

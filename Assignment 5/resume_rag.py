"""Index PDF, TXT, and DOCX resumes into ChromaDB using Cohere embeddings.

Existing collection and metadata names are preserved for compatibility with job_matcher.py.
"""

import argparse
import difflib
import json
import os
import re
import uuid
from pathlib import Path

import chromadb
import cohere
from dotenv import load_dotenv

from fs_tools import list_files, read_file


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")
RESUME_DIR = os.getenv("RESUME_DIR", "resumes")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "resume_collection")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
CHUNK_SIZE = int(os.getenv("RESUME_CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("RESUME_CHUNK_OVERLAP", "250"))
EMBED_BATCH_SIZE = int(os.getenv("COHERE_EMBED_BATCH_SIZE", "50"))
NORMALIZE_WITH_COHERE = os.getenv("COHERE_NORMALIZE_TEXT", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
NORMALIZE_SEGMENT_CHARS = int(os.getenv("COHERE_NORMALIZE_SEGMENT_CHARS", "3500"))

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is missing. Add it to your .env file.")

cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# --------------------------- text extraction ---------------------------

def extract_text_from_result(result):
    """Read the text payload returned by fs_tools.read_file()."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    if result.get("success") is False:
        return ""
    for key in ("content", "text", "extracted_text", "text_content"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    candidates = [value for value in result.values() if isinstance(value, str)]
    return max(candidates, key=len, default="")


def _normalise_path(path):
    """Canonical path used for metadata matching on Windows and other OSes."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def get_resume_paths(directory=RESUME_DIR):
    """Discover supported files using the project's existing fs_tools.list_files()."""
    try:
        items = list_files(str(directory))
    except Exception as exc:
        print(f"[ERROR] Could not list resumes in {directory}: {exc}")
        return []

    if not isinstance(items, list):
        print(f"[ERROR] list_files({directory!r}) did not return a list.")
        return []

    paths = []
    for item in items:
        if isinstance(item, dict):
            if item.get("success") is False or item.get("error"):
                print(f"[WARNING] File listing problem: {item.get('error', item)}")
                continue
            raw_path = item.get("path") or item.get("filepath") or item.get("file_path")
            if not raw_path and item.get("name"):
                raw_path = Path(directory) / item["name"]
        elif isinstance(item, str):
            raw_path = item
        else:
            continue

        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(directory) / path if path.parent == Path(".") else path
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.is_file():
            paths.append(str(path.resolve()))

    # Avoid duplicate paths returned by a custom filesystem helper.
    return sorted({_normalise_path(p): str(Path(p).resolve()) for p in paths}.values())


def load_resume(path):
    """Extract text with fs_tools; unreadable or empty files are skipped."""
    result = read_file(str(path))
    if isinstance(result, dict) and result.get("success") is False:
        print(f"[WARNING] Could not read {path}: {result.get('error', 'unknown error')}")
        return ""
    text = extract_text_from_result(result)
    if not text.strip():
        print(f"[WARNING] No readable text found in {path}; leaving any existing index untouched.")
        return ""
    return text


# --------------------------- conservative cleanup ---------------------------

def basic_text_cleanup(text):
    """Perform deterministic, low-risk cleanup before asking the LLM."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    # Join words split by a hyphen at a line break, e.g. "develop-\nment".
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    # Normalise line endings and horizontal whitespace without flattening sections.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_for_normalization(text, max_chars=NORMALIZE_SEGMENT_CHARS):
    """Split into sequential, non-overlapping pieces; no text is duplicated."""
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer paragraph, then line, then word boundaries.
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 2
            else:
                boundary = text.rfind("\n", start, end)
                if boundary > start + max_chars // 2:
                    end = boundary + 1
                else:
                    boundary = text.rfind(" ", start, end)
                    if boundary > start + max_chars // 2:
                        end = boundary + 1
        if end <= start:
            end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        start = end
    return pieces


def _response_text(response):
    """Extract text from Cohere ClientV2 chat response."""
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    return block["text"].strip()
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if block_type in (None, "text") and isinstance(block_text, str):
                    return block_text.strip()
    return ""


def _alphanumeric_stream(text):
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _critical_tokens(text):
    """Tokens that spacing cleanup must not change."""
    emails = re.findall(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text)
    urls = re.findall(r"(?i)\b(?:https?://|www\.)\S+", text)
    numbers = re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:\+)?(?!\w)", text)
    return sorted(token.casefold() for token in emails + urls + numbers)


def _valid_normalization(original, corrected):
    """Accept spacing/punctuation cleanup only; reject factual rewrites."""
    if not corrected or not corrected.strip():
        return False
    # The letter/digit stream must be identical: only spacing/punctuation can change.
    if _alphanumeric_stream(original) != _alphanumeric_stream(corrected):
        return False
    if _critical_tokens(original) != _critical_tokens(corrected):
        return False
    ratio = difflib.SequenceMatcher(None, original.casefold(), corrected.casefold()).ratio()
    return ratio >= 0.45


def normalize_segment_with_cohere(text):
    """Conservatively repair extraction spacing; on any failure keep original text."""
    if not text.strip() or not NORMALIZE_WITH_COHERE:
        return text
    prompt = (
        "Correct ONLY text-extraction spacing and obvious line-break artifacts in the resume text below. "
        "Examples include 'Ja va' -> 'Java', 'J a v a' -> 'Java', 'Spr ing Boot' -> 'Spring Boot', "
        "and '5 y ears' -> '5 years'. Do not summarize, paraphrase, reorder, add, or remove content. "
        "Preserve every factual detail, name, employer, qualification, date, number, skill, email, URL, "
        "heading, and bullet. If uncertain, leave the text unchanged. Return only the corrected text, "
        "with no explanation or code fence.\n\nRESUME TEXT:\n"
        + text
    )
    try:
        response = cohere_client.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative text-extraction cleanup tool. Never infer missing information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=min(5000, max(800, len(text) * 2)),
        )
        corrected = _response_text(response)
        if _valid_normalization(text, corrected):
            return corrected
        print("[WARNING] Cohere normalization failed validation; keeping original segment.")
    except Exception as exc:
        print(f"[WARNING] Cohere normalization unavailable; keeping original segment ({exc}).")
    return text


def _looks_like_broken_spacing(text):
    """Trigger chat cleanup only when spacing artifacts are plausible, saving API calls."""
    # Detect isolated letters such as "J a v a".
    if re.search(r"\\b(?:[A-Za-z]\\s+){2,}[A-Za-z]\\b", text):
        return True

    targets = {
        re.sub(r"[^a-z]", "", value.casefold())
        for value in SKILL_KEYWORDS
    }
    targets.update({
        "software", "years", "developer", "developers", "engineer", "engineering",
        "experience", "professional", "education", "university", "computer", "science",
        "bachelor", "master", "certification", "certifications", "responsibilities",
        "achievement", "achievements", "project", "projects", "database", "development",
        "technical", "application", "applications", "programming", "communication",
        "analytical", "architecture", "microservices", "qualification", "qualifications",
    })
    words = list(re.finditer(r"[A-Za-z]+", text))
    for count in (2, 3):
        for start in range(0, len(words) - count + 1):
            group = words[start:start + count]
            # Only combine neighboring words separated by whitespace, not across punctuation.
            between = text[group[0].end():group[-1].start()]
            if not between or not between.isspace():
                continue
            joined = "".join(item.group(0).casefold() for item in group)
            if joined in targets:
                phrase = " ".join(item.group(0).casefold() for item in group)
                if re.sub(r"[^a-z]", "", phrase) == joined:
                    # A clean multiword skill like "Spring Boot" is not a spacing error.
                    target_phrases = {
                        re.sub(r"[^a-z]", "", value.casefold())
                        for value in SKILL_KEYWORDS
                    }
                    if phrase in {value.casefold() for value in SKILL_KEYWORDS}:
                        continue
                    # At least one tiny fragment is a useful signal of extraction damage.
                    if any(len(item.group(0)) <= 3 for item in group):
                        return True
    return False


def normalize_resume_text(text):
    """Basic cleanup followed by bounded, validated Cohere spacing correction."""
    cleaned = basic_text_cleanup(text)
    if not NORMALIZE_WITH_COHERE:
        return cleaned
    pieces = _split_for_normalization(cleaned)
    corrected = [
        normalize_segment_with_cohere(piece) if _looks_like_broken_spacing(piece) else piece
        for piece in pieces
    ]
    return basic_text_cleanup("".join(corrected))


# --------------------------- metadata and chunking ---------------------------

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", ".net", "sql",
    "mysql", "postgresql", "mongodb", "redis", "html", "css", "react", "angular",
    "node.js", "express", "spring boot", "spring", "django", "flask", "fastapi",
    "rest api", "graphql", "microservices", "docker", "kubernetes", "aws", "azure",
    "gcp", "git", "github", "linux", "jenkins", "ci/cd", "machine learning",
    "deep learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "data analysis", "power bi", "tableau", "excel", "agile", "scrum", "communication",
]


def _has_phrase(text, phrase):
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(r"(?<![A-Za-z0-9+#.])" + escaped + r"(?![A-Za-z0-9+#.])", text, re.I))


def extract_metadata(text, path):
    """Extract conservative metadata from corrected resume text."""
    lines = [line.strip(" \t|•-") for line in text.splitlines() if line.strip()]
    filename_name = Path(path).stem.replace("_", " ").replace("-", " ").strip()
    candidate_name = filename_name
    for line in lines[:8]:
        if (
            1 <= len(line.split()) <= 5
            and len(line) <= 70
            and not re.search(r"[@:/\\\d]", line)
            and not re.search(r"\b(resume|curriculum vitae|experience|education|profile|summary)\b", line, re.I)
        ):
            candidate_name = line
            break

    skills = [skill for skill in SKILL_KEYWORDS if _has_phrase(text, skill)]
    years_patterns = [
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience\b",
        r"\bexperience\s*[:\-]\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+in\s+(?:software|IT|the industry|development)\b",
    ]
    years = []
    for pattern in years_patterns:
        for match in re.finditer(pattern, text, re.I):
            try:
                value = float(match.group(1))
                if 0 <= value <= 60:
                    years.append(value)
            except (ValueError, TypeError):
                pass
    experience_years = max(years) if years else None

    education_terms = [
        "B.Tech", "B.E.", "Bachelor", "B.Sc", "M.Tech", "M.E.", "Master",
        "M.Sc", "Ph.D", "Diploma", "MCA", "BCA",
    ]
    education = [term for term in education_terms if re.search(re.escape(term), text, re.I)]
    return {
        "candidate_name": candidate_name,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "resume_path": _normalise_path(path),
    }


def split_long_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks while preferring word boundaries."""
    if not text:
        return []
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


SECTION_PATTERN = re.compile(
    r"(?im)^\s*(professional\s+summary|summary|objective|skills|technical\s+skills|"
    r"work\s+experience|professional\s+experience|experience|employment\s+history|"
    r"projects|education|certifications|certificates|achievements|publications)\s*:?\s*$"
)


def chunk_resume(text):
    """Split by recognizable headings, then by length; maintain original section labels."""
    if not text.strip():
        return []
    matches = list(SECTION_PATTERN.finditer(text))
    sections = []
    if not matches:
        sections = [("general", text)]
    else:
        if text[:matches[0].start()].strip():
            sections.append(("general", text[:matches[0].start()].strip()))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((match.group(1).lower(), text[match.start():end].strip()))

    result = []
    for section, section_text in sections:
        for piece in split_long_text(section_text):
            result.append({"section": section, "text": piece})
    return result


# --------------------------- embeddings and safe updates ---------------------------

def _embedding_vectors(response):
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    vectors = getattr(embeddings, "float", None)
    if vectors is None and isinstance(embeddings, dict):
        vectors = embeddings.get("float")
    if vectors is None:
        raise ValueError("Cohere response did not contain float embeddings.")
    return [list(vector) for vector in vectors]


def generate_embeddings(texts, batch_size=EMBED_BATCH_SIZE):
    """Generate document embeddings in bounded batches."""
    if not texts:
        return []
    all_embeddings = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = cohere_client.embed(
            model=EMBED_MODEL,
            texts=batch,
            input_type="search_document",
            embedding_types=["float"],
        )
        all_embeddings.extend(_embedding_vectors(response))
    if len(all_embeddings) != len(texts):
        raise ValueError("Embedding count did not match document count.")
    return all_embeddings


def _existing_ids_for_resume(resume_path):
    """Find old chunk IDs for this resume without touching other resumes."""
    result = collection.get(include=["metadatas"])
    ids_to_delete = []
    target = _normalise_path(resume_path)
    for chunk_id, metadata in zip(result.get("ids", []), result.get("metadatas", [])):
        metadata = metadata or {}
        stored_path = metadata.get("resume_path") or metadata.get("filepath") or metadata.get("file_path")
        if stored_path and _normalise_path(stored_path) == target:
            ids_to_delete.append(chunk_id)
    return ids_to_delete


def index_one_resume(path):
    """Index one resume atomically with respect to API failures; retain old chunks on failure."""
    original_text = load_resume(path)
    if not original_text.strip():
        return 0

    cleaned_text = normalize_resume_text(original_text)
    metadata = extract_metadata(cleaned_text, path)
    chunks = chunk_resume(cleaned_text)
    if not chunks:
        print(f"[WARNING] No chunks produced for {path}; existing index untouched.")
        return 0

    documents = [item["text"] for item in chunks]
    # Generate all new embeddings before deleting any old chunks.
    embeddings = generate_embeddings(documents)
    path_key = _normalise_path(path)
    ids = [
        f"{uuid.uuid5(uuid.NAMESPACE_URL, path_key)}_{index}"
        for index in range(len(documents))
    ]
    metadatas = []
    for item in chunks:
        metadatas.append({
            "candidate_name": metadata["candidate_name"],
            "resume_path": path_key,
            "section": item["section"],
            "skills": ", ".join(metadata["skills"]),
            "experience_years": (
                float(metadata["experience_years"])
                if metadata["experience_years"] is not None else -1.0
            ),
            "education": ", ".join(metadata["education"]),
        })

    old_ids = _existing_ids_for_resume(path_key)
    if old_ids:
        collection.delete(ids=old_ids)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[OK] {metadata['candidate_name']}: indexed {len(documents)} chunks.")
    return len(documents)


def index_resumes(directory=RESUME_DIR):
    """Update each readable resume in place; never clear the full Chroma collection."""
    paths = get_resume_paths(directory)
    if not paths:
        print(f"No supported resume files found in: {directory}")
        return

    print(f"Found {len(paths)} supported resume file(s) in {directory}.")
    total_chunks = 0
    indexed_files = 0
    for path in paths:
        try:
            count = index_one_resume(path)
            if count:
                total_chunks += count
                indexed_files += 1
        except Exception as exc:
            # Do not delete prior chunks if reading, chat, or embedding fails.
            print(f"[ERROR] Failed to index {path}: {exc}")
            continue

    print(
        f"\nFinished. Successfully updated {indexed_files} resume(s), "
        f"wrote {total_chunks} chunks. Collection now contains {collection.count()} chunks."
    )
    print("No full database reset was performed. Removed or renamed files are not automatically purged.")


def main():
    parser = argparse.ArgumentParser(description="Normalize and index resumes into ChromaDB.")
    parser.add_argument("--resume-dir", default=RESUME_DIR, help="Folder containing PDF/TXT/DOCX resumes.")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip Cohere chat cleanup and only use deterministic text cleanup.",
    )
    args = parser.parse_args()

    global NORMALIZE_WITH_COHERE
    if args.skip_normalization:
        NORMALIZE_WITH_COHERE = False
    index_resumes(args.resume_dir)


if __name__ == "__main__":
    main()

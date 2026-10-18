from pathlib import Path
from datetime import datetime

from PyPDF2 import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _get_metadata(path: Path) -> dict:
    """Create common file metadata."""
    stat = path.stat()

    return {
        "name": path.name,
        "path": str(path.absolute()),
        "extension": path.suffix.lower(),
        "size": stat.st_size,
        "modified_date": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat()
    }


def _validate_file(filepath: str) -> Path:
    """Validate that the path exists, is a file, and has a supported extension."""

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    return path


def _extract_text(path: Path) -> str:
    """Extract text based on file extension."""

    extension = path.suffix.lower()

    if extension == ".txt":
        return path.read_text(encoding="utf-8")

    if extension == ".pdf":
        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    if extension == ".docx":
        document = Document(str(path))

        paragraphs = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {extension}")


def read_file(filepath: str) -> dict:
    """
    Read a PDF, TXT, or DOCX file.

    Returns content and file metadata.
    """

    try:
        path = _validate_file(filepath)
        content = _extract_text(path)

        return {
            "success": True,
            "content": content,
            "metadata": _get_metadata(path)
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {filepath}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def list_files(directory: str, extension: str = None) -> list:
    """
    List files in a directory.

    Optionally filter by extension.
    """

    try:
        path = Path(directory)

        if not path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not path.is_dir():
            raise ValueError(
                f"Path is not a directory: {directory}"
            )

        if extension:
            extension = extension.lower()

            if not extension.startswith("."):
                extension = "." + extension

        files = []

        for file_path in path.iterdir():

            if not file_path.is_file():
                continue

            if extension and file_path.suffix.lower() != extension:
                continue

            files.append(_get_metadata(file_path))

        return files

    except PermissionError:
        return [{
            "success": False,
            "error": f"Permission denied: {directory}"
        }]

    except Exception as e:
        return [{
            "success": False,
            "error": str(e)
        }]


def write_file(filepath: str, content: str) -> dict:
    """
    Write content to a file.

    Creates parent directories automatically.
    """

    try:
        path = Path(filepath)

        # Create directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "message": "File written successfully",
            "metadata": _get_metadata(path)
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {filepath}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def search_in_file(filepath: str, keyword: str) -> dict:
    """
    Search for a keyword in a file.

    Search is case-insensitive and returns
    matching text with surrounding context.
    """

    try:
        if not keyword:
            raise ValueError("Keyword cannot be empty")

        # Reuse read_file instead of duplicating extraction logic
        result = read_file(filepath)

        if not result["success"]:
            return result

        content = result["content"]

        content_lower = content.lower()
        keyword_lower = keyword.lower()

        matches = []
        start = 0
        context_size = 100

        while True:
            index = content_lower.find(
                keyword_lower,
                start
            )

            if index == -1:
                break

            context_start = max(
                0,
                index - context_size
            )

            context_end = min(
                len(content),
                index + len(keyword) + context_size
            )

            matches.append({
                "position": index,
                "context": content[
                    context_start:context_end
                ]
            })

            start = index + len(keyword)

        return {
            "success": True,
            "keyword": keyword,
            "match_count": len(matches),
            "matches": matches,
            "metadata": result["metadata"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


"""Markdown document loading primitives.

Example
-------
```python
from pathlib import Path

from rag_from_scratch_codex.loaders.markdown import MarkdownLoader

loader = MarkdownLoader()
documents = loader.load_directory(Path("./docs"))

for document in documents:
    print(document.metadata["relative_path"])
```
"""

from dataclasses import dataclass, field
from pathlib import Path

from rag_from_scratch_codex.config.settings import AppConfig


@dataclass
class Document:
    """A loaded document in the project's internal format."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class MarkdownLoader:
    """Load Markdown files recursively from disk."""

    def load_directory(self, directory: Path) -> list[Document]:
        """Load all Markdown files under ``directory``.

        Files are discovered recursively and returned in a stable, sorted order
        so the loader behaves predictably across runs.
        """
        root = directory.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Markdown directory does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Markdown path must be a directory: {root}")

        documents: list[Document] = []
        for path in sorted(root.rglob("*.md")):
            documents.append(self._load_file(path=path, root=root))
        return documents

    def load_from_config(self, config: AppConfig) -> list[Document]:
        """Load Markdown files using the configured ``docs_path`` setting."""
        return self.load_directory(Path(config.docs_path))

    def _load_file(self, path: Path, root: Path) -> Document:
        """Read one Markdown file and attach basic file metadata."""
        return Document(
            text=path.read_text(encoding="utf-8"),
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "relative_path": str(path.relative_to(root)),
            },
        )

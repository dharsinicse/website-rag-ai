
import os
import shutil
import tempfile
from pathlib import Path

from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


def save_staged_indexes(
    vector_store: FAISSVectorStore,
    keyword_store: BM25KeywordStore,
    vector_path: str,
    keyword_path: str,
) -> None:
    """Stage both indexes and replace the live files with recovery backups."""
    vector_target = Path(vector_path)
    keyword_target = Path(keyword_path)
    vector_target.parent.mkdir(parents=True, exist_ok=True)
    keyword_target.parent.mkdir(parents=True, exist_ok=True)

    vector_metadata_target = vector_target.with_suffix(".metadata.npy")

    targets = [
        vector_target,
        vector_metadata_target,
        keyword_target,
    ]

    with tempfile.TemporaryDirectory(
        dir=str(vector_target.parent)
    ) as temp_dir:
        temp = Path(temp_dir)

        staged_vector = temp / vector_target.name
        staged_keyword = temp / keyword_target.name

        vector_store.save(str(staged_vector))
        keyword_store.save(str(staged_keyword))

        staged_files = [
            staged_vector,
            staged_vector.with_suffix(".metadata.npy"),
            staged_keyword,
        ]

        # Keep backups so a failed replacement can be rolled back.
        backup_dir = temp / "backups"
        backup_dir.mkdir()

        backups: dict[Path, Path | None] = {}
        for target in targets:
            if target.exists():
                backup = backup_dir / f"{len(backups)}_{target.name}"
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None

        replaced: list[Path] = []

        try:
            for staged, target in zip(staged_files, targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
                replaced.append(target)
        except Exception:
            for target in reversed(replaced):
                backup = backups[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup, target)
            raise

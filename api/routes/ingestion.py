from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.ingestion_app import get_ingestion_service
from api.schemas.ingestion import IngestionResponse
from config.settings import get_settings
from api.rag_app import get_rag_pipeline

from retrieval.knowledge_base_persistence import save_staged_indexes

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".html", ".htm"
}


@router.post("/ingest", response_model=IngestionResponse)
async def ingest(file: UploadFile = File(...)):
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    max_size = settings.max_upload_size_mb * 1024 * 1024

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid4().hex}{extension}"
    size = 0

    try:
        with saved_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the configured size limit.",
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        service = get_ingestion_service()
        document_id = saved_path.stem

        result = service.ingest_file(
            str(saved_path),
            metadata={
                "document_id": document_id,
                "original_filename": filename,
            },
        )

        if result["status"] == "duplicate":
            raise HTTPException(
                status_code=409,
                detail="This document has already been indexed.",
            )

        if result["status"] != "indexed":
            raise HTTPException(
                status_code=422,
                detail="No indexable text was found in the uploaded file.",
            )

        pipeline = get_rag_pipeline()
        hybrid = pipeline.retriever.hybrid_retriever

        # Persist the same stores used by the chat pipeline.
        vector_path = settings.vector_index_path
        keyword_path = settings.keyword_index_path

        save_staged_indexes(
            vector_store=hybrid.vector_store,
            keyword_store=hybrid.keyword_store,
            vector_path=vector_path,
            keyword_path=keyword_path,
        )

        return IngestionResponse(
            document_id=document_id,
            filename=filename,
            status="indexed",
            chunks=result["chunks"],
            message="Document uploaded and indexed successfully.",
        )

    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc
    finally:
        await file.close()

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete an indexed document by its generated ID."""
    settings = get_settings()
    pipeline = get_rag_pipeline()
    hybrid = pipeline.retriever.hybrid_retriever

    matching_sources = {
        item.get("source")
        for item in hybrid.keyword_store.metadata
        if item.get("metadata", {}).get("document_id") == document_id
    }

    if not matching_sources:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if len(matching_sources) != 1:
        raise HTTPException(
            status_code=409,
            detail="Document ID maps to multiple sources.",
        )

    source = next(iter(matching_sources))

    if not source:
        raise HTTPException(
            status_code=409,
            detail="Document source metadata is invalid.",
        )

    service = get_ingestion_service()

    try:
        result = service.delete_document(
            source=source,
            vector_path=settings.vector_index_path,
            keyword_path=settings.keyword_index_path,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
        ) from exc

    if result["status"] != "deleted":
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # Remove the uploaded file only after index persistence succeeds.
    upload_dir = Path(settings.upload_dir).resolve()
    source_path = Path(source).resolve()

    if source_path.parent == upload_dir and source_path.is_file():
        source_path.unlink(missing_ok=True)

    return {
        "document_id": document_id,
        "status": "deleted",
        "deleted_chunks": result["deleted_chunks"],
        "message": "Document deleted successfully.",
    }
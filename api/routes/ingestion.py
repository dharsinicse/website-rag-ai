
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.ingestion_app import get_ingestion_service
from api.schemas.ingestion import IngestionResponse
from config.settings import get_settings
from api.rag_app import get_rag_pipeline


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
        result = service.ingest_file(str(saved_path))

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

        hybrid.vector_store.save(vector_path)
        hybrid.keyword_store.save(keyword_path)

        return IngestionResponse(
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
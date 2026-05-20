from fastapi.testclient import TestClient

from rag_service.main import app


def test_health_endpoint_reports_service_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "index" in payload


def test_retrieve_requires_loaded_index_when_not_loaded(monkeypatch):
    from rag_service import main

    monkeypatch.setattr(main.rag_service, "_engine", None)

    client = TestClient(app)
    response = client.post("/retrieve", json={"query": "What is RAG?", "top_k": 3})

    assert response.status_code == 409
    assert "RAG index is not loaded" in response.json()["detail"]


def test_document_endpoints_delegate_to_service(monkeypatch):
    from rag_service import main

    record = {
        "document_id": "doc-1",
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size_bytes": 13,
        "source": "notes",
        "status": "indexed",
        "chunk_count": 1,
        "created_at": "2026-05-21T00:00:00+00:00",
    }

    monkeypatch.setattr(main.rag_service, "list_documents", lambda: {"documents": [record]})
    monkeypatch.setattr(main.rag_service, "get_document", lambda document_id: {**record, "document_id": document_id})
    monkeypatch.setattr(main.rag_service, "delete_document", lambda document_id: {**record, "document_id": document_id})

    def upload_document(filename, content, content_type=None, source=None):
        return {
            **record,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": len(content),
            "source": source,
            "message": "Document uploaded and indexed.",
        }

    monkeypatch.setattr(main.rag_service, "upload_document", upload_document)

    client = TestClient(app)

    assert client.get("/documents").json()["documents"][0]["document_id"] == "doc-1"
    assert client.get("/documents/doc-2").json()["document_id"] == "doc-2"

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello document", "text/plain")},
        data={"source": "notes"},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["filename"] == "notes.txt"
    assert upload_response.json()["size_bytes"] == 14

    delete_response = client.delete("/documents/doc-3")
    assert delete_response.status_code == 200
    assert delete_response.json()["document_id"] == "doc-3"

from datetime import datetime, timedelta

from src.api.dependencies.session import SessionManager
from src.pipeline.vision.types import UIDocument


def _document() -> UIDocument:
    return UIDocument(
        blocks=[],
        full_page_text="",
        images={},
        metadata={},
        dimensions=(0, 0),
    )


def test_session_manager_evicts_oldest_session_when_bounded():
    manager = SessionManager(max_sessions=2)

    first = manager.create_session(_document(), "one")
    second = manager.create_session(_document(), "two")
    third = manager.create_session(_document(), "three")

    assert manager.get_session(first) is None
    assert manager.get_session(second) is not None
    assert manager.get_session(third) is not None
    assert manager.get_stats()["active_sessions"] == 2


def test_session_manager_expires_old_sessions():
    manager = SessionManager(session_timeout_minutes=1, max_sessions=5)
    doc_id = manager.create_session(_document(), "image")

    manager._sessions[doc_id].created_at = datetime.utcnow() - timedelta(minutes=2)

    assert manager.get_session(doc_id) is None
    assert manager.get_stats()["active_sessions"] == 0

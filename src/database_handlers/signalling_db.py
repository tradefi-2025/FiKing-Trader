import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

from pymongo import ASCENDING, MongoClient
from pymongo.errors import ConnectionFailure

from src.utils.env import load_env

load_env()

logger = logging.getLogger(__name__)


class SignallingDBService:
    """MongoDB access layer for signalling agent records."""

    COLLECTION = "agents"

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        *,
        strict_errors: bool = False,
    ) -> None:
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.database_name = database or os.getenv("MONGO_DATABASE", "fiking_trader")
        self.strict_errors = strict_errors

        self.client: MongoClient = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.COLLECTION]

        self._init()

    def _init(self) -> None:
        try:
            self.client.admin.command("ping")
            self.collection.create_index([("model_id", ASCENDING)], unique=True, name="model_id_unique")
            self.collection.create_index([("status", ASCENDING)], name="status_idx")
            logger.info("Connected to MongoDB: %s", self.database_name)
        except Exception as exc:
            logger.error("Failed to connect to MongoDB: %s", exc)
            if isinstance(exc, ConnectionFailure) or self.strict_errors:
                raise

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    def insert_agent(self, doc: Dict) -> bool:
        """Insert a new agent document. Expects at least model_id and status."""
        try:
            doc.setdefault("created_at", datetime.utcnow())
            doc.setdefault("updated_at", datetime.utcnow())
            self.collection.insert_one(doc)
            return True
        except Exception as exc:
            logger.error("insert_agent failed: %s", exc)
            if self.strict_errors:
                raise
            return False

    def get_agent(self, model_id: str) -> Optional[Dict]:
        """Return a single agent document by model_id, or None."""
        try:
            doc = self.collection.find_one({"model_id": model_id}, {"_id": 0})
            return doc
        except Exception as exc:
            logger.error("get_agent failed: %s", exc)
            if self.strict_errors:
                raise
            return None

    def update_agent(self, model_id: str, update: Dict) -> bool:
        """Partial update an agent by model_id. Returns True if a document was modified."""
        try:
            update["updated_at"] = datetime.utcnow()
            result = self.collection.update_one({"model_id": model_id}, {"$set": update})
            return result.modified_count > 0
        except Exception as exc:
            logger.error("update_agent failed: %s", exc)
            if self.strict_errors:
                raise
            return False

    def delete_agent(self, model_id: str) -> bool:
        """Delete an agent by model_id. Returns True if deleted."""
        try:
            result = self.collection.delete_one({"model_id": model_id})
            return result.deleted_count > 0
        except Exception as exc:
            logger.error("delete_agent failed: %s", exc)
            if self.strict_errors:
                raise
            return False

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    def get_agents_by_status(self, status: str) -> List[Dict]:
        """Return all agents matching the given status."""
        try:
            return list(self.collection.find({"status": status}, {"_id": 0}))
        except Exception as exc:
            logger.error("get_agents_by_status failed: %s", exc)
            if self.strict_errors:
                raise
            return []

    def get_pending_agents(self) -> List[Dict]:
        return self.get_agents_by_status("PENDING")

    def mark_inprogress(self, model_id: str) -> bool:
        return self.update_agent(model_id, {"status": "INPROGRESS"})

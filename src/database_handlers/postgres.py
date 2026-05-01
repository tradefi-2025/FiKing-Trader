from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json, RealDictCursor

from src.utils.env import load_env

load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGNAL_STATUS_VALUES = {"NEW", "READ", "ARCHIVED"}
AGENT_STATUS_VALUES = {
"PENDING",
"IN_PROGRESS",
"COMPLETED",
"FAILED",
"CANCELLED",
"ACTIVE",
"INACTIVE",
"CREATED",
}
class DatabaseClient:
    """Generic communication layer for a remote PostgreSQL database using env vars."""

    

    def __init__(self) -> None:
        self.host = self._get_env("POSTGRES_HOST")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.name = os.getenv("POSTGRES_DB", "postgres")
        self.user = self._get_env("POSTGRES_USER")
        self.password = self._get_env("POSTGRES_PASSWORD")
        self.sslmode = os.getenv("POSTGRES_SSLMODE", "prefer")

    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.name,
            user=self.user,
            password=self.password,
            sslmode=self.sslmode,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_agent_status(self, agent_id: int, status: str) -> Optional[Dict[str, Any]]:
        normalized_status = (status or "").strip().upper()
        if normalized_status not in self.AGENT_STATUS_VALUES:
            raise ValueError(f"Invalid agent status: {status}")

        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE public.agent
                    SET training_status = %s
                    WHERE agent_id = %s
                    RETURNING agent_id, name, training_status, version, user_id
                    """,
                    (normalized_status, agent_id),
                )
                row = cur.fetchone()

        return dict(row) if row else None
    
    def mark_agent_pending(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "PENDING")

    def mark_agent_inprogress(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "IN_PROGRESS")

    def mark_agent_completed(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "COMPLETED")

    def mark_agent_created(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "CREATED")
    
    def mark_agent_failed(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "FAILED")
    
    def mark_agent_failed(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "FAILED")

    def mark_agent_cancelled(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "CANCELLED")

    def mark_agent_active(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "ACTIVE")
    
    def mark_agent_inactive(self, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.update_agent_status(agent_id, "INACTIVE")

        
    def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_id, name, training_status, version, user_id
                FROM public.agent
                WHERE agent_id = %s
                """,
                (agent_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_agent_parameters(self, agent_id: int) -> Dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    a.agent_id AS agent_id,
                    a.name AS agent_name,
                    a.training_status,
                    a.version,
                    a.user_id,
                    f.id AS feature_id,
                    f.name AS feature_name,
                    f.description AS feature_description,
                    pd.parameter_definition_id AS parameter_definition_id,
                    pd.name AS parameter_name,
                    pd.type AS parameter_type,
                    pd.default_value,
                    pd.required,
                    pd.enum_values,
                    pd.file_name,
                    pv.value AS parameter_value
                FROM public.agent a
                LEFT JOIN public.agent_feature af
                    ON af.agent_id = a.agent_id
                LEFT JOIN public.feature f
                    ON f.id = af.feature_id
                LEFT JOIN public.parameter_value pv
                    ON pv.agent_feature_id = af.agent_feature_id
                LEFT JOIN public.parameter_definition pd
                    ON pd.parameter_definition_id = pv.parameter_definition_id
                WHERE a.agent_id = %s
                ORDER BY af.feature_id, pd.parameter_definition_id
                """,
                (agent_id,),
            )
            rows = cur.fetchall()

        if not rows:
            raise ValueError(f"Agent {agent_id} not found")

        first = rows[0]
        result: Dict[str, Any] = {
            "agent_id": str(first["agent_id"]),
            "name": first.get("agent_name"),
            "training_status": first.get("training_status"),
            "version": first.get("version"),
            "user_id": first.get("user_id"),
            "features": [],
            "parameters": {},
        }

        feature_map: Dict[Any, Dict[str, Any]] = {}

        for row in rows:
            feature_id = row.get("feature_id")
            feature_name = row.get("feature_name")
            parameter_name = row.get("parameter_name")

            if feature_id is not None and feature_id not in feature_map:
                feature_entry = {
                    "id": feature_id,
                    "name": feature_name,
                    "description": row.get("feature_description"),
                    "parameters": {},
                }
                feature_map[feature_id] = feature_entry
                result["features"].append(feature_entry)

            if not parameter_name:
                continue

            typed_value = self._coerce_parameter_value(
                value=row.get("parameter_value"),
                default_value=row.get("default_value"),
                parameter_type=row.get("parameter_type"),
                enum_values=row.get("enum_values"),
            )

            parameter_entry = {
                "value": typed_value,
                "type": row.get("parameter_type"),
                "required": bool(row.get("required")) if row.get("required") is not None else False,
                "default_value": row.get("default_value"),
                "enum_values": self._parse_enum_values(row.get("enum_values")),
                "file_name": row.get("file_name"),
                "feature": feature_name,
            }

            result["parameters"][parameter_name] = parameter_entry

            if feature_id is not None:
                feature_map[feature_id]["parameters"][parameter_name] = parameter_entry

        return result

    def build_service_payload(self, agent_id: int, service_name: Optional[str] = None) -> Dict[str, Any]:
        agent_data = self.get_agent_parameters(agent_id)
        payload: Dict[str, Any] = {
            "service": service_name or self._infer_service_name(agent_data),
            "agent_id": agent_data["agent_id"],
            "status": self._normalize_status(agent_data.get("training_status")),
        }

        for parameter_name, metadata in agent_data["parameters"].items():
            payload[parameter_name] = metadata["value"]

        return payload

    def build_service_payload_json(self, agent_id: int, service_name: Optional[str] = None) -> str:
        return json.dumps(self.build_service_payload(agent_id, service_name), ensure_ascii=False, indent=2)

    def build_launch_payload(self, agent_id: int, service_name: str) -> Dict[str, str]:
        return {
            "service": service_name,
            "agent_id": str(agent_id),
        }

    def build_signal_payload(self, agent_id: int, service_name: str) -> Dict[str, str]:
        return {
            "service": service_name,
            "agent_id": str(agent_id),
        }

    def get_pending_agents(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_id, name, training_status, version, user_id
                FROM public.agent
                WHERE UPPER(REPLACE(training_status, '_', '')) = 'PENDING'
                ORDER BY agent_id ASC
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def watch_pending_requests(self, poll_interval: float = 2.0, batch_size: int = 50) -> Iterable[List[Dict[str, Any]]]:
        while True:
            pending = self.get_pending_agents(limit=batch_size)
            if pending:
                yield pending
            time.sleep(poll_interval)

    def build_flat_metadata(self, agent_id: int, service_name: Optional[str] = None) -> Dict[str, Any]:
        agent_data = self.get_agent_parameters(agent_id)

        inferred_service = service_name or self._infer_service_name(agent_data)
        request: Dict[str, Any] = {
            "model_id": str(agent_data["agent_id"]),
            "agent_id": str(agent_data["agent_id"]),
            "service": inferred_service,
            "agent_name": agent_data.get("name"),
        }

        for parameter_name, metadata in agent_data.get("parameters", {}).items():
            request[parameter_name] = metadata.get("value")

        if "status" not in request:
            request["status"] = self._normalize_status(agent_data.get("training_status"))

        return request

    @staticmethod
    def _infer_service_name(agent_data: Dict[str, Any]) -> str:
        features = agent_data.get("features", [])
        if features:
            first_feature_name = features[0].get("name")
            if first_feature_name:
                return str(first_feature_name).lower()
        return "unknown"

    @staticmethod
    def _normalize_status(status: Optional[str]) -> Optional[str]:
        if status is None:
            return None
        normalized = str(status).replace("_", "").upper()
        if normalized == "COMPLETED":
            return "CREATED"
        return normalized

    def _coerce_parameter_value(
        self,
        value: Any,
        default_value: Any,
        parameter_type: Optional[str],
        enum_values: Any,
    ) -> Any:
        raw = value if value not in (None, "") else default_value
        if raw is None:
            return None

        kind = (parameter_type or "STRING").upper()
        text = str(raw).strip()

        if kind == "INTEGER":
            return int(float(text))
        if kind == "DOUBLE":
            return float(text)
        if kind == "BOOLEAN":
            return text.lower() in {"1", "true", "yes", "y", "on"}
        if kind == "DATE":
            return text
        if kind == "ENUM":
            valid = self._parse_enum_values(enum_values)
            if valid and text not in valid:
                raise ValueError(f"Invalid enum value '{text}', expected one of {valid}")
            return text
        if kind == "FILE":
            return text

        parsed = self._try_parse_json(text)
        if parsed is not None:
            return parsed
        if "," in text:
            parts = [item.strip() for item in text.split(",") if item.strip()]
            if len(parts) > 1:
                return parts
        return text

    @staticmethod
    def _parse_enum_values(enum_values: Any) -> List[str]:
        if enum_values in (None, ""):
            return []
        if isinstance(enum_values, list):
            return [str(item) for item in enum_values]
        text = str(enum_values).strip()
        if text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in text.split(",") if item.strip()]

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        if not text:
            return None
        if not ((text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}"))):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def create_agent_with_parameters(
        self,
        *,
        user_id: int,
        agent_name: str,
        training_status: str,
        version: Optional[str],
        feature_name: str,
        feature_description: Optional[str],
        parameters: Dict[str, Dict[str, Any]],
    ) -> int:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.agent (name, training_status, version, user_id)
                VALUES (%s, %s, %s, %s)
                RETURNING agent_id
                """,
                (agent_name, training_status, version, user_id),
            )
            agent_id = cur.fetchone()["agent_id"]

            cur.execute(
                """
                INSERT INTO public.feature (name, description)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
                RETURNING id
                """,
                (feature_name, feature_description),
            )
            feature_id = cur.fetchone()["id"]

            cur.execute(
                """
                INSERT INTO public.agent_feature (agent_id, feature_id)
                VALUES (%s, %s)
                RETURNING agent_feature_id
                """,
                (agent_id, feature_id),
            )
            agent_feature_id = cur.fetchone()["agent_feature_id"]

            for parameter_name, metadata in parameters.items():
                parameter_type = metadata.get("type", "STRING")
                default_value = self._serialize_value(metadata.get("default_value"))
                required = bool(metadata.get("required", False))
                enum_values = self._serialize_value(metadata.get("enum_values"))
                file_name = metadata.get("file_name")
                description = metadata.get("description")
                value = self._serialize_value(metadata.get("value"))
                min_value = self._serialize_value(metadata.get("min_value"))
                max_value = self._serialize_value(metadata.get("max_value"))

                cur.execute(
                    """
                    INSERT INTO public.parameter_definition (
                        name, default_value, description, min_value, max_value,
                        type, enum_values, file_name, required, feature_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING parameter_definition_id
                    """,
                    (
                        parameter_name,
                        default_value,
                        description,
                        min_value,
                        max_value,
                        parameter_type,
                        enum_values,
                        file_name,
                        required,
                        feature_id,
                    ),
                )
                parameter_definition_id = cur.fetchone()["parameter_definition_id"]

                cur.execute(
                    """
                    INSERT INTO public.parameter_value (value, agent_feature_id, parameter_definition_id)
                    VALUES (%s, %s, %s)
                    """,
                    (value, agent_feature_id, parameter_definition_id),
                )

            return int(agent_id)

    @staticmethod
    def _serialize_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _normalize_signal_row(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row:
            return row

        probabilities = row.get("probabilities") or {}
        warnings = row.get("warnings") or []

        return {
            "signalId": row["signal_id"],
            "agentId": row["agent_id"],
            "agentName": row.get("agent_name"),
            "signalDate": row["signal_date"].isoformat() if row.get("signal_date") else None,
            "estimatedAction": row.get("estimated_action"),
            "signal": row.get("signal"),
            "probability": row.get("probability"),
            "probabilities": probabilities,
            "volume": row.get("volume"),
            "notional": row.get("notional"),
            "stopLossPrice": row.get("stop_loss_price"),
            "riskAmount": row.get("risk_amount"),
            "sizingMethod": row.get("sizing_method"),
            "warnings": warnings,
            "status": row.get("status"),
        }

    def create_signal(
        self,
        agent_id: int,
        signal_date,
        estimated_action: str,
        signal: str,
        probability: float,
        probabilities: Dict[str, Any],
        volume: Optional[float] = None,
        notional: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        risk_amount: Optional[float] = None,
        sizing_method: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        status: str = "NEW",
    ) -> Dict[str, Any]:
        status = (status or "NEW").upper()
        if status not in self.SIGNAL_STATUS_VALUES:
            raise ValueError(f"Invalid signal status: {status}")

        warnings = warnings or []
        probabilities = probabilities or {}

        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO public.signal (
                        agent_id,
                        signal_date,
                        estimated_action,
                        signal,
                        probability,
                        probabilities,
                        volume,
                        notional,
                        stop_loss_price,
                        risk_amount,
                        sizing_method,
                        warnings,
                        status
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING
                        signal_id,
                        agent_id,
                        signal_date,
                        estimated_action,
                        signal,
                        probability,
                        probabilities,
                        volume,
                        notional,
                        stop_loss_price,
                        risk_amount,
                        sizing_method,
                        warnings,
                        status
                    """,
                    (
                        agent_id,
                        signal_date,
                        estimated_action,
                        signal,
                        probability,
                        Json(probabilities),
                        volume,
                        notional,
                        stop_loss_price,
                        risk_amount,
                        sizing_method,
                        warnings,
                        status,
                    ),
                )
                row = cur.fetchone()

                cur.execute(
                    """
                    SELECT a.name AS agent_name
                    FROM public.agent a
                    WHERE a.agent_id = %s
                    """,
                    (agent_id,),
                )
                agent_row = cur.fetchone()
                row["agent_name"] = agent_row["agent_name"] if agent_row else None

        return self._normalize_signal_row(row)

    def get_user_signals(self, user_id: int) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        s.signal_id,
                        s.agent_id,
                        a.name AS agent_name,
                        s.signal_date,
                        s.estimated_action,
                        s.signal,
                        s.probability,
                        s.probabilities,
                        s.volume,
                        s.notional,
                        s.stop_loss_price,
                        s.risk_amount,
                        s.sizing_method,
                        s.warnings,
                        s.status
                    FROM public.signal s
                    JOIN public.agent a
                      ON a.agent_id = s.agent_id
                    WHERE a.user_id = %s
                    ORDER BY s.signal_date DESC, s.signal_id DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()

        return [self._normalize_signal_row(row) for row in rows]

    def update_signal_status(
        self,
        user_id: int,
        signal_id: int,
        status: str,
    ) -> Optional[Dict[str, Any]]:
        status = (status or "").upper()
        if status not in self.SIGNAL_STATUS_VALUES:
            raise ValueError(f"Invalid signal status: {status}")

        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE public.signal s
                    SET status = %s
                    FROM public.agent a
                    WHERE s.agent_id = a.agent_id
                      AND s.signal_id = %s
                      AND a.user_id = %s
                    RETURNING
                        s.signal_id,
                        s.agent_id,
                        a.name AS agent_name,
                        s.signal_date,
                        s.estimated_action,
                        s.signal,
                        s.probability,
                        s.probabilities,
                        s.volume,
                        s.notional,
                        s.stop_loss_price,
                        s.risk_amount,
                        s.sizing_method,
                        s.warnings,
                        s.status
                    """,
                    (status, signal_id, user_id),
                )
                row = cur.fetchone()

        return self._normalize_signal_row(row) if row else None

    def delete_signal(self, user_id: int, signal_id: int) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM public.signal s
                    USING public.agent a
                    WHERE s.agent_id = a.agent_id
                      AND s.signal_id = %s
                      AND a.user_id = %s
                    """,
                    (signal_id, user_id),
                )
                deleted = cur.rowcount > 0

        return deleted

    def get_signal_by_id(self, user_id: int, signal_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        s.signal_id,
                        s.agent_id,
                        a.name AS agent_name,
                        s.signal_date,
                        s.estimated_action,
                        s.signal,
                        s.probability,
                        s.probabilities,
                        s.volume,
                        s.notional,
                        s.stop_loss_price,
                        s.risk_amount,
                        s.sizing_method,
                        s.warnings,
                        s.status
                    FROM public.signal s
                    JOIN public.agent a
                      ON a.agent_id = s.agent_id
                    WHERE s.signal_id = %s
                      AND a.user_id = %s
                    """,
                    (signal_id, user_id),
                )
                row = cur.fetchone()

        return self._normalize_signal_row(row) if row else None

    def agent_belongs_to_user(self, agent_id: int, user_id: int) -> bool:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM public.agent
                    WHERE agent_id = %s
                      AND user_id = %s
                    """,
                    (agent_id, user_id),
                )
                return cur.fetchone() is not None

    def build_flat_metadata_for_user(self, agent_id: int, user_id: int):
        if not self.agent_belongs_to_user(agent_id, user_id):
            return None
        return self.build_flat_metadata(agent_id)

    def list_existing_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if schema:
                    cur.execute(
                        """
                        SELECT table_schema, table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                          AND table_schema = %s
                        ORDER BY table_schema, table_name
                        """,
                        (schema,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT table_schema, table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                          AND table_schema NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY table_schema, table_name
                        """
                    )

                return [dict(row) for row in cur.fetchall()]

    def debug_query(
        self,
        query: str,
        params: Optional[Iterable[Any]] = None,
        *,
        fetch: bool = True,
        commit: bool = False,
    ) -> Dict[str, Any]:
        query_clean = (query or "").strip()
        if not query_clean:
            raise ValueError("Query must not be empty.")

        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query_clean, tuple(params) if params is not None else None)

                columns = [desc.name for desc in cur.description] if cur.description else []
                rows = [dict(row) for row in cur.fetchall()] if fetch and cur.description else []

                result = {
                    "ok": True,
                    "query": query_clean,
                    "params": list(params) if params is not None else [],
                    "columns": columns,
                    "rows": rows,
                    "rowcount": cur.rowcount,
                }

                if commit:
                    conn.commit()
                else:
                    conn.rollback()

                return result

    def interactive_sql_shell(self) -> None:
        print("Interactive PostgreSQL debug shell")
        print("Type SQL statements and press Enter.")
        print("Type 'quit' or 'exit' to leave.")

        with self.connection() as conn:
            conn.autocommit = False

            while True:
                try:
                    query = input("\nsql> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting shell.")
                    break

                if not query:
                    continue

                lowered = query.lower()
                if lowered in {"quit", "exit"}:
                    print("Bye.")
                    break

                if lowered == "rollback":
                    conn.rollback()
                    print("Rolled back current transaction.")
                    continue

                if lowered == "commit":
                    conn.commit()
                    print("Committed current transaction.")
                    continue

                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(query)

                        if cur.description:
                            rows = cur.fetchall()
                            columns = [desc.name for desc in cur.description]
                            print(f"Columns: {columns}")
                            print(json.dumps([dict(r) for r in rows], indent=2, default=str))
                            print(f"{len(rows)} row(s)")
                        else:
                            print(f"OK - {cur.rowcount} row(s) affected")

                except Exception as e:
                    conn.rollback()
                    print(f"ERROR: {e}")
    
def test_create_and_fetch_three_agents() -> List[Dict[str, Any]]:
    db = DatabaseClient()

    agents_to_create = [
        {
            "user_id": 1,
            "agent_name": "signal-aapl",
            "training_status": "PENDING",
            "version": "v1",
            "feature_name": "signaling",
            "feature_description": "Signaling service",
            "parameters": {
                "equity": {"type": "STRING", "required": True, "value": "AAPL"},
                "time_frequency": {"type": "ENUM", "required": True, "value": "1H", "enum_values": ["1MIN", "5MIN", "1H", "1D"]},
                "observation_horizon": {"type": "INTEGER", "required": True, "value": 120},
                "prediction_horizon": {"type": "INTEGER", "required": True, "value": 8},
                "news_resources": {"type": "STRING", "value": ["reuters", "bloomberg"]},
                "confidence_level": {"type": "DOUBLE", "value": 0.8},
            },
        },
        {
            "user_id": 1,
            "agent_name": "signal-tsla",
            "training_status": "IN_PROGRESS",
            "version": "v1",
            "feature_name": "signaling",
            "feature_description": "Signaling service",
            "parameters": {
                "equity": {"type": "STRING", "required": True, "value": "TSLA"},
                "time_frequency": {"type": "ENUM", "required": True, "value": "1D", "enum_values": ["1H", "1D"]},
                "observation_horizon": {"type": "INTEGER", "required": True, "value": 30},
                "prediction_horizon": {"type": "INTEGER", "required": True, "value": 5},
                "signal_frequency": {"type": "INTEGER", "required": True, "value": 1},
            },
        },
        {
            "user_id": 1,
            "agent_name": "risk-btc",
            "training_status": "ACTIVE",
            "version": "v2",
            "feature_name": "risk_management",
            "feature_description": "Risk service",
            "parameters": {
                "asset": {"type": "STRING", "required": True, "value": "BTCUSD"},
                "max_drawdown": {"type": "DOUBLE", "required": True, "value": 0.12},
                "use_stop_loss": {"type": "BOOLEAN", "required": True, "value": True},
                "lookback_window": {"type": "INTEGER", "required": True, "value": 90},
            },
        },
    ]

    created_agent_ids = []
    for agent_data in agents_to_create:
        created_agent_ids.append(db.create_agent_with_parameters(**agent_data))

    return [db.build_service_payload(agent_id) for agent_id in created_agent_ids]



if __name__ == "__main__":
    db = DatabaseClient()
    db.interactive_sql_shell()
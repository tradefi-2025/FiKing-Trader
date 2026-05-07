"""
Flask Server for FiKing-Trader API
Handles requests for agent management, predictions, and data operations
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from functools import wraps

import jwt
import pika
import redis
from flask import Flask, g, jsonify, request

from src.database_handlers.postgres import DatabaseClient
from src.utils.env import load_env
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS


# Load environment variables first
load_env()

def _split_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

    
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Auth config
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
FLASK_ENV = os.getenv("FLASK_ENV", "production")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AUTH_MODE = os.getenv("AUTH_MODE", "cookie").strip().lower()  # bearer | cookie | both

CORS_ALLOWED_ORIGINS = _split_csv_env("CORS_ALLOWED_ORIGINS")
CORS_ALLOWED_METHODS = _split_csv_env(
    "CORS_ALLOWED_METHODS",
    "GET,POST,PUT,PATCH,DELETE,OPTIONS",
)
CORS_ALLOWED_HEADERS = _split_csv_env(
    "CORS_ALLOWED_HEADERS",
    "Content-Type,Authorization",
)
CORS_SUPPORTS_CREDENTIALS = _env_bool("CORS_SUPPORTS_CREDENTIALS", False)

JWT_COOKIE_NAME = os.getenv("JWT_COOKIE_NAME", "AccessToken")
JWT_COOKIE_SECURE = _env_bool("JWT_COOKIE_SECURE", FLASK_ENV == "production")
JWT_COOKIE_SAMESITE = os.getenv("JWT_COOKIE_SAMESITE", "None")
JWT_COOKIE_DOMAIN = os.getenv("JWT_COOKIE_DOMAIN")
JWT_COOKIE_MAX_AGE = int(os.getenv("JWT_COOKIE_MAX_AGE", "3600"))
JWT_EXPIRATION_SECONDS = int(os.getenv("JWT_EXPIRATION_SECONDS", "3600"))

CORS(
    app,
    origins=CORS_ALLOWED_ORIGINS if CORS_ALLOWED_ORIGINS else [],
    methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
    supports_credentials=CORS_SUPPORTS_CREDENTIALS,
    vary_header=True,
)


def validate_startup_config():
    errors = []

    if not JWT_SECRET:
        errors.append("JWT_SECRET is required")

    if FLASK_ENV == "production":
        if not CORS_ALLOWED_ORIGINS:
            errors.append("CORS_ALLOWED_ORIGINS must be set in production")
        if CORS_SUPPORTS_CREDENTIALS and "*" in CORS_ALLOWED_ORIGINS:
            errors.append("CORS_ALLOWED_ORIGINS cannot contain '*' when credentials are enabled")
        if AUTH_MODE in {"cookie", "both"}:
            if JWT_COOKIE_SAMESITE.lower() == "none" and not JWT_COOKIE_SECURE:
                errors.append("JWT_COOKIE_SECURE must be true when JWT_COOKIE_SAMESITE=None")

    if AUTH_MODE not in {"bearer", "cookie", "both"}:
        errors.append("AUTH_MODE must be one of: bearer, cookie, both")

    if errors:
        raise RuntimeError("Invalid startup configuration: " + "; ".join(errors))

validate_startup_config()




# Services
_db_service = DatabaseClient()

# Redis configuration
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    redis_client.ping()
    logger.info("Connected to Redis")
except redis.ConnectionError as e:
    logger.warning(f"Redis not available: {e}. Agent status checks will fail.")
    redis_client = None


def op_response(success: bool, error_message: str | None = None, status_code: int = 200):
    return jsonify({
        "success": success,
        "errorMessage": error_message,
    }), status_code


def _extract_token():
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip(), "bearer"

    cookie_token = request.cookies.get(JWT_COOKIE_NAME)
    print(request.cookies)
    if cookie_token:
        return cookie_token, "cookie"

    return None, None


def require_internal_jwt(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not JWT_SECRET:
            return op_response(False, "JWT secret is not configured on the server.", 500)

        token, token_source = _extract_token()


        if AUTH_MODE == "bearer" and token_source != "bearer":
            return op_response(False, "Missing Bearer token.", 401)

        if AUTH_MODE == "cookie" and token_source != "cookie":
            return op_response(False, f"Missing {JWT_COOKIE_NAME} cookie.", 401)

        if AUTH_MODE == "both" and not token:
            return op_response(False, "Missing authentication token.", 401)

        if not token:
            return op_response(False, "Missing authentication token.", 401)

        decode_kwargs = {
            "key": JWT_SECRET,
            "algorithms": [JWT_ALGORITHM],
            "options": {"require": ["exp", "jti"]},
        }

        if JWT_ISSUER:
            decode_kwargs["issuer"] = JWT_ISSUER
        if JWT_AUDIENCE:
            decode_kwargs["audience"] = JWT_AUDIENCE

        try:
            payload = jwt.decode(token, **decode_kwargs)
            print(payload)
        except jwt.ExpiredSignatureError:
            return op_response(False, "Authentication token has expired.", 401)
        except jwt.InvalidIssuerError:
            return op_response(False, "Invalid token issuer.", 401)
        except jwt.InvalidAudienceError:
            return op_response(False, "Invalid token audience.", 401)
        except jwt.InvalidTokenError as e:
            return op_response(False, f"Invalid authentication token: {str(e)}", 401)

        token_user_id = payload.get("jti")
        if token_user_id is None:
            return op_response(False, "Token does not contain a user identifier.", 401)

        try:
            g.user_id = int(token_user_id)
        except (TypeError, ValueError):
            return op_response(False, "Token contains an invalid user identifier.", 401)

        g.user_email = payload.get("sub")
        g.jwt_payload = payload
        g.auth_mode = token_source
        return fn(*args, **kwargs)

    return wrapper


def get_user_agent_or_404(agent_id: int):
    try:
        agent = _db_service.build_flat_metadata_for_user(agent_id, g.user_id)
    except Exception as e:
        logger.error(f"Database error while fetching agent {agent_id} for user {g.user_id}: {e}")
        return None, op_response(False, "Database error while checking agent access.", 500)

    if not agent:
        return None, op_response(False, f"Agent {agent_id} not found for authenticated user.", 404)

    return agent, None


def verify(body):
    """
    Verify the request body for model creation.
    """
    return True


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "FiKing-Trader API is healthy"
    }), 200


@app.route("/status", methods=["GET"])
def get_status():
    pass


def _poll_pending_agents(interval: int = 10):
    """
    Background thread: polls for PENDING agents, transitions them to INPROGRESS,
    builds a flat training request from DB data, and enqueues it onto the
    service training queue.
    """
    logger.info("Signalling agent poller started.")

    while True:
        connection = None
        try:
            pending_agents = _db_service.get_pending_agents()

            if pending_agents:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost"))
                )
                channel = connection.channel()

                for agent in pending_agents:
                    agent_id = agent["agent_id"]
                    training_request = _db_service.build_flat_metadata(agent_id=agent_id)

                    service = training_request.get("service", "signaling")
                    model_id = training_request.get("model_id", str(agent_id))
                    queue = f"{service}_training_queue"

                    updated = _db_service.mark_agent_inprogress(agent_id)
                    if not updated:
                        logger.warning(f"Could not mark agent {agent_id} as INPROGRESS, skipping.")
                        continue

                    channel.queue_declare(queue=queue, durable=True)
                    channel.basic_publish(
                        exchange="",
                        routing_key=queue,
                        body=json.dumps(training_request),
                        properties=pika.BasicProperties(delivery_mode=2),
                    )

                    logger.info(f"Enqueued agent {model_id} (agent_id={agent_id}) onto {queue}.")

        except Exception as e:
            logger.error(f"Poller error: {e}")

        finally:
            if connection is not None and connection.is_open:
                connection.close()

        time.sleep(interval)


def start_background_poller():
    """Start the agent poller. Called by gunicorn post_worker_init (worker 1 only)."""
    t = threading.Thread(
        target=_poll_pending_agents,
        kwargs={"interval": int(os.getenv("AGENT_POLL_INTERVAL", 10))},
        daemon=True,
        name="signalling-agent-poller",
    )
    t.start()
    logger.info("Background agent poller started.")

@app.route("/ai/agent/launch", methods=["POST", "OPTIONS"])
@require_internal_jwt
def start_model():
    """
    Start/launch a trained agent for continuous prediction.
    Response schema:
        {
          "success": boolean,
          "errorMessage": string | null
        }
    """
    body = request.get_json() or {}
    agent_id = body.get("agentId")

    if not agent_id:
        return op_response(False, "Missing required field: agentId (or modelId for backward compatibility).", 400)

    try:
        agent_id = int(agent_id)
    except (TypeError, ValueError):
        return op_response(False, f"Invalid agent identifier: {agent_id}", 400)

    agent, error = get_user_agent_or_404(agent_id)
    if error:
        return error

    raw_status = agent.get("training_status") or agent.get("status")
    normalized_status = str(raw_status).replace("_", "").upper() if raw_status else ""

    if normalized_status in {"PENDING", "INPROGRESS", "FAILED"}:
        return op_response(False, f"Agent {agent_id} is not ready to launch. Current status: {raw_status}.", 409)

    if normalized_status == "ACTIVE":
        return op_response(False, f"Agent {agent_id} is already active.", 409)

    if normalized_status not in {"CREATED", "INACTIVE"}:
        return op_response(False, f"Agent {agent_id} has unsupported launch status: {raw_status}.", 409)

    if redis_client is None:
        return op_response(False, "Redis not available. Cannot check active agent cache.", 503)

    service = agent.get("service", "signaling")

    if redis_client.sismember(f"active_agents:{service}", str(agent_id)):
        return op_response(False, f"Agent {agent_id} is already active.", 409)

    queue = f"{service}_launch_queue"
    connection = None

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost"))
        )
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)

        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(agent),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        return op_response(True, None, 200)

    except Exception as e:
        logger.error(f"Launch queue error for agent {agent_id}, user {g.user_id}: {e}")
        return op_response(False, "Failed to queue launch request.", 500)

    finally:
        if connection is not None and connection.is_open:
            connection.close()


@app.route("/ai/agent/deactivate", methods=["POST", "OPTIONS"])
@require_internal_jwt
def stop_model():
    """
    Stop a currently active agent.
    Response schema:
        {
          "success": boolean,
          "errorMessage": string | null
        }
    """
    body = request.get_json() or {}
    agent_id = body.get("agentId")

    if not agent_id:
        return op_response(False, "Missing required field: agentId (or modelId for backward compatibility).", 400)

    try:
        agent_id = int(agent_id)
    except (TypeError, ValueError):
        return op_response(False, f"Invalid agent identifier: {agent_id}", 400)

    agent, error = get_user_agent_or_404(agent_id)
    if error:
        return error

    raw_status = agent.get("training_status") or agent.get("status")
    normalized_status = str(raw_status).replace("_", "").upper() if raw_status else ""

    if normalized_status != "ACTIVE":
        return op_response(False, f"Agent {agent_id} is not active and cannot be stopped. Current status: {raw_status}.", 409)

    service = agent.get("service", "signaling")
    queue = f"{service}_stop_queue"
    connection = None

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost"))
        )
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)

        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(agent),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        return op_response(True, None, 200)

    except Exception as e:
        logger.error(f"Stop queue error for agent {agent_id}, user {g.user_id}: {e}")
        return op_response(False, "Failed to queue stop request.", 500)

    finally:
        if connection is not None and connection.is_open:
            connection.close()


@app.route("/ai/agent/health", methods=["GET"])
def check_agent_health():
    return op_response(True, "Agent health check endpoint is a placeholder and always returns healthy.", 200)


@app.route("/ai/agent/inference", methods=["POST", "OPTIONS"])
@require_internal_jwt
def get_signals():
    """
    Get a signal from an agent, persist it in DB,
    and return OperationResponse.
    """
    body = request.get_json() or {}
    print(body)
    
    agent_id = body.get("agentId")

    if not agent_id:
        return op_response(False, "Missing required field: agent_id (or model_id for backward compatibility).", 400)

    try:
        agent_id = int(agent_id)
    except (TypeError, ValueError):
        return op_response(False, f"Invalid agent identifier: {agent_id}", 400)

    agent, error = get_user_agent_or_404(agent_id)
    if error:
        return error

    raw_status = agent.get("training_status") or agent.get("status")
    normalized_status = str(raw_status).replace("_", "").upper() if raw_status else ""

    if normalized_status in {"PENDING", "INPROGRESS", "FAILED"}:
        return op_response(False, f"Agent {agent_id} is not available for signal retrieval. Current status: {raw_status}.", 409)

    if normalized_status not in {"ACTIVE", "INACTIVE", "CREATED"}:
        return op_response(False, f"Agent {agent_id} has unsupported signal status: {raw_status}.", 409)

    model_id = agent.get("model_id", str(agent_id))
    queue = f"inference_queue_{model_id}"
    connection = None

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost"))
        )
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)

        reply_to = channel.queue_declare(queue="", exclusive=True).method.queue
        correlation_id = str(model_id)
        signal_response = None

        def on_response(ch, method, props, body_bytes):
            nonlocal signal_response
            if props.correlation_id == correlation_id:
                signal_response = json.loads(body_bytes)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()

        channel.basic_consume(queue=reply_to, on_message_callback=on_response)

        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(agent),
            properties=pika.BasicProperties(
                reply_to=reply_to,
                correlation_id=correlation_id,
                delivery_mode=2,
            )
        )

        connection.call_later(30, lambda: channel.stop_consuming())
        channel.start_consuming()

        if signal_response is None:
            return op_response(False, "Timeout waiting for agent response", 504)

        generated_signal = signal_response

        try:
            _db_service.create_signal(
                agent_id=agent_id,
                signal_date=datetime.now(timezone.utc),
                estimated_action=generated_signal.get("estimated_action"),
                signal=generated_signal.get("signal"),
                probability=generated_signal.get("probability"),
                probabilities=generated_signal.get("probabilities", {}),
                volume=generated_signal.get("volume"),
                notional=generated_signal.get("notional"),
                stop_loss_price=generated_signal.get("stop_loss_price"),
                risk_amount=generated_signal.get("risk_amount"),
                sizing_method=generated_signal.get("sizing_method"),
                warnings=generated_signal.get("warnings", []),
                status="NEW",
            )
        except Exception as e:
            logger.error(
                f"Failed to persist generated signal for agent {agent_id}, user {g.user_id}: {e}; signal={generated_signal}"
            )
            return op_response(False, "Signal generated but failed to persist.", 500)

        return op_response(True, None, 200)

    except Exception as e:
        logger.error(f"Error fetching signals for agent {agent_id}, user {g.user_id}: {str(e)}")
        return op_response(False, f"Error fetching signals: {str(e)}", 500)

    finally:
        if connection and connection.is_open:
            connection.close()

@app.route("/debug/patch-test", methods=["PATCH", "OPTIONS"])
@require_internal_jwt
def patch_test():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    return jsonify({
        "success": True,
        "errorMessage": None,
        "method": "PATCH",
        "payload": payload,
        "userId": g.user_id,
    }), 200

@app.route("/whoami", methods=["GET", "OPTIONS"])
@require_internal_jwt
def whoami():
    if request.method == "OPTIONS":
        return ("", 204)

    return jsonify({
        "success": True,
        "errorMessage": None,
        "data": {
            "userId": g.user_id,
            "email": g.user_email,
            "authMode": g.get("auth_mode", None),
        }
    }), 200
# ==================== Data Management ====================

@app.route("/data/timeseries", methods=["POST"])
def upload_timeseries_data():
    pass


@app.route("/data/timeseries/<equity>", methods=["GET"])
def get_timeseries_data(equity):
    pass


@app.route("/data/equities", methods=["GET"])
def list_equities():
    pass


@app.route("/data/live/<equity>", methods=["GET"])
def get_live_price(equity):
    pass


# ==================== Training ====================

@app.route("/model/<service>/<model_id>/train", methods=["POST"])
def train_model(service, model_id):
    pass


@app.route("/model/<model_id>/training/status", methods=["GET"])
def get_training_status(model_id):
    pass


# ==================== Configuration ====================

@app.route("/config/entities", methods=["GET"])
def get_entities():
    pass


@app.route("/config/frequencies", methods=["GET"])
def get_time_frequencies():
    pass


# ==================== Error Handlers ====================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "errorMessage": getattr(error, "description", "Bad request"),
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "errorMessage": getattr(error, "description", "Resource not found"),
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "success": False,
        "errorMessage": "Internal server error",
    }), 500


def run_server(host="0.0.0.0", port=5000, debug=False):
    logger.info(f"Starting Flask server on {host}:{port}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    run_server(host=host, port=port, debug=debug)
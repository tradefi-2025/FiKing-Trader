"""
Flask Server for FiKing-Trader API
Handles requests for agent management, predictions, and data operations
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from src.utils.env import load_env
import pika
import redis

# Load environment variables
load_env()

def verify(body):
    """
    Verify the request body for model creation
    
    Args:
        body: Request JSON body
        service: Service type (e.g., 'signaling', 'forecasting')
    """
    return True
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# redis configuration
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    logger.info("Connected to Redis")
except redis.ConnectionError as e:
    logger.warning(f"Redis not available: {e}. Agent status checks will fail.")
    redis_client = None
# ==================== Health Check ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        'status': 'OK',
        'message': 'FiKing-Trader API is healthy'
    }), 200


@app.route('/status', methods=['GET'])
def get_status():
        """Get system status and statistics"""
        pass



# ==================== Signalling Agent Poller (replaces /model/create) ====================

from src.database_handlers.signalling_db import SignallingDBService
import threading
import time

_db_service = SignallingDBService()

def _poll_pending_agents(interval: int = 10):
    """
    Background thread: polls for PENDING agents, transitions them to INPROGRESS,
    and enqueues them onto the service training queue.
    Runs every `interval` seconds.
    """
    logger.info("Signalling agent poller started.")
    while True:
        try:
            pending = _db_service.get_pending_agents()

            if pending:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost'))
                )

                for agent in pending:
                    model_id = agent["model_id"]
                    service  = agent.get("service", "signaling")
                    queue    = f"{service}_training_queue"

                    # Transition status first — prevents double-pickup on next poll
                    updated = _db_service.mark_inprogress(model_id)
                    if not updated:
                        logger.warning(f"Could not mark {model_id} as INPROGRESS, skipping.")
                        continue

                    channel = connection.channel()
                    channel.queue_declare(queue=queue, durable=True)
                    channel.basic_publish(
                        exchange='',
                        routing_key=queue,
                        body=json.dumps(agent),
                        properties=pika.BasicProperties(delivery_mode=2),
                    )
                    logger.info(f"Enqueued agent {model_id} onto {queue}.")

                connection.close()

        except Exception as e:
            logger.error(f"Poller error: {e}")

        time.sleep(interval)


# Start the poller as a daemon thread when the app module loads
_poller_thread = threading.Thread(
    target=_poll_pending_agents,
    kwargs={"interval": int(os.getenv("AGENT_POLL_INTERVAL", 10))},
    daemon=True,
    name="signalling-agent-poller",
)
_poller_thread.start()
# ==================== Model/Agent Management ====================


    

@app.route('/lauch', methods=['POST'])
def start_model( ):

    """
    Start/launch a trained model for continuous prediction
    
    Path Parameters:
        - model_id: Unique model identifier
    
    Response:
        - status: SUCCESS or FAILURE
        - message: Result description
    """
    body = request.get_json() or {}
    model_id = body.get('model_id')
    service = body.get('service')
    # Check Redis availability
    if redis_client is None:
        return jsonify({
            'status': 'ERROR',
            'message': 'Redis not available. Cannot check agent status.'
        }), 503
    
    # Check if model is already active

    if redis_client.sismember(f'active_agents:{service}', model_id):
        return jsonify({
            'status': 'ERROR',
            'message': f'Model {model_id} is already active.'
        }), 400
    
    queue = f'{service}_launch_queue'
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost')))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    
    channel.basic_publish(
        exchange='',
        routing_key=queue,
        body=json.dumps(body),
        properties=pika.BasicProperties()
    )

    return jsonify({
        'status': 'SUCCESS',
        'message': f'Model {model_id} launch request accepted and queued for processing.'
    }), 200



@app.route('/stop', methods=['POST'])
def stop_model():
    """
    Stop a running model
    
    Path Parameters:
        - model_id: Unique model identifier
    
    Response:
        - status: SUCCESS or FAILURE
        - message: Result description
    """
    pass



# ==================== Inference/Prediction ====================




@app.route('/signal', methods=['POST'])
def get_signals():
    """
    Get recent trading signals from a model
    
    Path Parameters:
        - model_id: Unique model identifier
    
    Request Body:
        - limit: (optional) Maximum number of signals
        - start_date: (optional) Filter from date
        - end_date: (optional) Filter to date
    
    Response:
        - signals: List of signal objects
        - total: Total number of signals
    """
    # Check Redis availability
    if redis_client is None:
        return jsonify({
            'status': 'ERROR',
            'message': 'Redis not available. Cannot check agent status.'
        }), 503
    
    body = request.get_json() or {}
    service=body.get('service')
    model_id = body.get('model_id')
    if redis_client.sismember(f'active_agents:{service}', model_id):
        connection = None
        try:
            # Fetch signals from agent via RabbitMQ RPC
            queue = f'inference_queue_{model_id}'
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost')))
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)
            reply_to = channel.queue_declare(queue='', exclusive=True).method.queue
            correlation_id = str(model_id)
            
            channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=json.dumps(body),
                properties=pika.BasicProperties(
                    reply_to=reply_to,
                    correlation_id=correlation_id,
                )
            )
            
            signal_response = None
            
            def on_response(ch, method, props, body):
                nonlocal signal_response
                if props.correlation_id == correlation_id:
                    signal_response = json.loads(body)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    ch.stop_consuming()
            
            channel.basic_consume(
                queue=reply_to,
                on_message_callback=on_response
            )
            
            # Add timeout to prevent infinite blocking
            connection.call_later(30, lambda: channel.stop_consuming())  # 30 second timeout
            channel.start_consuming()
            
            if signal_response is None:
                return jsonify({
                    'status': 'ERROR',
                    'message': 'Timeout waiting for agent response'
                }), 504
            
            return jsonify({
                'signals': signal_response.get('signals', []),
                'total': len(signal_response.get('signals', []))
            }), 200
            
        except Exception as e:
            logger.error(f"Error fetching signals: {str(e)}")
            return jsonify({
                'status': 'ERROR',
                'message': f'Error fetching signals: {str(e)}'
            }), 500
        finally:
            if connection and connection.is_open:
                connection.close()
    else:
        return jsonify({
            'status': 'ERROR',
            'message': f'Model {model_id} is not active or does not exist.'
        }), 404
    


# ==================== Data Management ====================

@app.route('/data/timeseries', methods=['POST'])
def upload_timeseries_data():
    """
    Upload time series data for an equity
    
    Request Body:
        - equity: Equity symbol
        - frequency: Time frequency
        - data: Array of OHLCV data points
    
    Response:
        - status: SUCCESS or FAILURE
        - records_inserted: Number of records inserted
        - message: Result description
    """
    pass


@app.route('/data/timeseries/<equity>', methods=['GET'])
def get_timeseries_data(equity):
    """
    Retrieve time series data for an equity
    
    Path Parameters:
        - equity: Equity symbol
    
    Query Parameters:
        - frequency: Time frequency (required)
        - start_date: (optional) Filter from date
        - end_date: (optional) Filter to date
        - limit: (optional) Maximum number of records
    
    Response:
        - equity: Equity symbol
        - frequency: Time frequency
        - data: Array of OHLCV data points
        - total: Total number of records
    """
    pass


@app.route('/data/equities', methods=['GET'])
def list_equities():
    """
    List all equities with available data
    
    Query Parameters:
        - frequency: (optional) Filter by frequency
    
    Response:
        - equities: List of equity symbols
        - total: Total number of equities
    """
    pass


@app.route('/data/live/<equity>', methods=['GET'])
def get_live_price(equity):
    """
    Get live price data for an equity
    
    Path Parameters:
        - equity: Equity symbol
    
    Response:
        - equity: Equity symbol
        - price: Current price data (OHLCV)
        - timestamp: Data timestamp
    """
    pass


# ==================== Training ====================

@app.route('/model/<service>/<model_id>/train', methods=['POST'])
def train_model(service, model_id):
    """
    Trigger model training
    
    Path Parameters:
        - model_id: Unique model identifier
    
    Request Body:
        - training_params: (optional) Training parameters override
    
    Response:
        - status: SUCCESS or FAILURE
        - message: Result description
        - job_id: Training job identifier (if async)
    """
    pass


@app.route('/model/<model_id>/training/status', methods=['GET'])
def get_training_status(model_id):
    """
    Get training status for a model
    
    Path Parameters:
        - model_id: Unique model identifier
    
    Response:
        - status: Training status (pending, running, completed, failed)
        - progress: Training progress percentage
        - current_epoch: Current training epoch
        - total_epochs: Total number of epochs
        - metrics: Current training metrics
    """
    pass


# ==================== Configuration ====================

@app.route('/config/entities', methods=['GET'])
def get_entities():
    """
    Get list of supported entities from config
    
    Response:
        - entities: List of entity configurations
    """
    pass


@app.route('/config/frequencies', methods=['GET'])
def get_time_frequencies():
    """
    Get list of supported time frequencies
    
    Response:
        - frequencies: List of time frequency configurations
    """
    pass


# ==================== Error Handlers ====================

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'status': 'ERROR',
        'message': 'Bad request',
        'error': str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'status': 'ERROR',
        'message': 'Resource not found',
        'error': str(error)
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'ERROR',
        'message': 'Internal server error',
        'error': str(error)
    }), 500


# ==================== Main ====================

def run_server(host='0.0.0.0', port=5000, debug=False):
    """
    Run the Flask server
    
    Args:
        host: Host address
        port: Port number
        debug: Debug mode flag
    """
    logger.info(f"Starting Flask server on {host}:{port}")
    _poller_thread = threading.Thread(
        target=_poll_pending_agents,
        kwargs={"interval": int(os.getenv("AGENT_POLL_INTERVAL", 10))},
        daemon=True,
        name="signalling-agent-poller",
    )
    _poller_thread.start()
    app.run(host=host, port=port, debug=debug)
    


if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    run_server(host=host, port=port, debug=debug)

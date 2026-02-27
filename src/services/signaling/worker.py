import os
import json
import logging
import threading
import pika
from dotenv import load_dotenv

from .config import SignalingConfig
from .dl import SignalingDataLoader
from .model import SignalingModelV1
from .verification import SignalingVerification
from .agent import Agent
from ...database_handlers.mongoDB import MongoDBService

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalingWorker:
    """Worker that consumes training requests from RabbitMQ queue"""

    def __init__(self):
        self.config = SignalingConfig()
        
        # RabbitMQ configuration
        self.rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
        self.rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
        self.rabbitmq_user = os.getenv('RABBITMQ_USER', 'guest')
        self.rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'guest')
        
        # Queue name for training requests (specific to signaling service)
        self.training_queue_name = 'signaling_training_queue'
        self.launch_queue_name = 'signaling_launch_queue'
        
        # Thread tracking (thread-safe)
        self.active_threads = []
        self.threads_lock = threading.Lock()  # Lock for thread-safe access
        self.max_concurrent_trainings = int(os.getenv('MAX_CONCURRENT_TRAININGS', 5))
        
        # Launch thread tracking (thread-safe)
        self.active_launch_threads = []
        self.launch_threads_lock = threading.Lock()
        self.max_concurrent_launches = int(os.getenv('MAX_CONCURRENT_LAUNCHES', 3))
        
        # Agent tracking for launched agents
        self.active_agents = {}  # {agent_id: Agent instance}
        self.agents_lock = threading.Lock()
        
        # Channel reference for graceful shutdown
        self.channel = None
        self.channel_lock = threading.Lock()
    
    def _get_rabbitmq_connection(self):
        """Create RabbitMQ connection"""
        credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        return pika.BlockingConnection(parameters)
    
    def _process_launch_request(self, request_data, reply_to=None, correlation_id=None):
        """
        Process a launch request by initializing an agent, fetching the model weights from MongoDB, and launching the agent
        
        Args:
            request_data: Dictionary containing launch request parameters including agent_id
            reply_to: Optional reply queue name for sending response
            correlation_id: Optional correlation ID for response matching
        
        Returns:
            bool: True if launch successful, False otherwise
        """
        try:
            agent_id = request_data.get('agent_id')
            logger.info(f"Processing launch request for agent: {agent_id}")
            
            # Check if agent is already running
            with self.agents_lock:
                if agent_id in self.active_agents:
                    logger.warning(f"Agent {agent_id} is already running")
                    # Send error response if reply_to is specified
                    if reply_to:
                        self._send_launch_response(
                            reply_to=reply_to,
                            correlation_id=correlation_id,
                            agent_id=agent_id,
                            success=False,
                            message="Agent is already running"
                        )
                    return False
            
            # Load model weights from MongoDB
            logger.info(f"Loading model weights from MongoDB for agent: {agent_id}")
            mongo_service = MongoDBService()
            model_data = mongo_service.load_agent_weights(agent_id)
            mongo_service.close()
            
            if not model_data:
                logger.error(f"❌ Model weights not found for agent: {agent_id}")
                # Send error response if reply_to is specified
                if reply_to:
                    self._send_launch_response(
                        reply_to=reply_to,
                        correlation_id=correlation_id,
                        agent_id=agent_id,
                        success=False,
                        message="Model weights not found"
                    )
                return False
            
            # Extract metadata for dataloader
            metadata_dataloader = {
                'equity': request_data.get('entity_name') or model_data.get('equity'),
                'observation_horizon': request_data.get('observation_horizon'),
                'prediction_horizon': request_data.get('prediction_horizon'),
                'time_frequency': request_data.get('time_frequency'),
                'news_observation_horizon': request_data.get('news_observation_horizon'),
                'news_retrieval_prompt': request_data.get('news_retrieval_prompt'),
                'news_resources': request_data.get('news_resources'),
            }
            
            # Create dataloader object
            logger.info("Creating dataloader...")
            dataloader = SignalingDataLoader(metadata=metadata_dataloader)
            
            # Create model object and load weights
            logger.info("Creating model and loading weights...")
            model = SignalingModelV1(self.config)
            model.load_state_dict(model_data['weights'])  # Load the trained weights
            model.eval()  # Set to evaluation mode
            
            # Create verification object
            logger.info("Creating verification...")
            verification = SignalingVerification(model)
            
            # Create agent object with all components
            logger.info("Creating agent...")
            agent = Agent(
                meta_data=request_data,
                model=model,
                dataloader=dataloader,
                verification=verification
            )
            
            # Store agent reference before launching (thread-safe)
            with self.agents_lock:
                self.active_agents[agent_id] = agent
            
            # Launch the agent (starts main loop and inference consumer threads)
            logger.info(f"Launching agent: {agent_id}...")
            agent.launch()
            
            logger.info(f"✅ Successfully launched agent: {agent_id}")
            logger.info(f"📊 Active agents: {len(self.active_agents)}")
            
            # Send response to reply_to queue if specified
            if reply_to:
                self._send_launch_response(
                    reply_to=reply_to,
                    correlation_id=correlation_id,
                    agent_id=agent_id,
                    success=True,
                    message="Agent launched successfully"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing launch request: {str(e)}")
            
            # Send error response to reply_to queue if specified
            if reply_to:
                self._send_launch_response(
                    reply_to=reply_to,
                    correlation_id=correlation_id,
                    agent_id=agent_id,
                    success=False,
                    message=str(e)
                )
            
            # Clean up agent reference on failure
            with self.agents_lock:
                if agent_id in self.active_agents:
                    del self.active_agents[agent_id]
            return False
    
    def _send_launch_response(self, reply_to, correlation_id, agent_id, success, message):
        """
        Send a response to the reply_to queue
        
        Args:
            reply_to: Queue name to send response to
            correlation_id: Correlation ID for matching request/response
            agent_id: Agent identifier
            success: Whether the launch was successful
            message: Status message
        """
        try:
            # Create a new connection for thread-safe publishing
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            
            # Prepare response message
            response = {
                'agent_id': agent_id,
                'success': success,
                'message': message,
                'timestamp': str(threading.current_thread().name)
            }
            
            # Publish response
            channel.basic_publish(
                exchange='',
                routing_key=reply_to,
                body=json.dumps(response),
                properties=pika.BasicProperties(
                    correlation_id=correlation_id,
                    content_type='application/json'
                )
            )
            
            logger.info(f"📤 Sent launch response to {reply_to} for agent: {agent_id}")
            
            # Close connection
            connection.close()
            
        except Exception as e:
            logger.error(f"❌ Error sending launch response: {str(e)}")
    
    def _process_training_request(self, request_data):
        """
        Process a training request by creating agent and training model
        
        Args:
            request_data: Dictionary containing training request parameters
        """
        try:
            logger.info(f"Processing training request: {request_data.get('model_id')}")
            
            # Extract metadata from request
            
            metadata_dataloader = {
                'equity': request_data.get('entity_name'),
                'observation_horizon': request_data.get('observation_horizon'),
                'prediction_horizon': request_data.get('prediction_horizon'),
                'time_frequency': request_data.get('time_frequency'),
                'news_observation_horizon': request_data.get('news_observation_horizon'),
                'news_retrieval_prompt': request_data.get('news_retrieval_prompt'),
                'news_resources': request_data.get('news_resources'),

            }
            
           

            # Create dataloader object
            logger.info("Creating dataloader...")
            dataloader = SignalingDataLoader(metadata=metadata_dataloader)
            
            # Create model object
            logger.info("Creating model...")
            model = SignalingModelV1(self.config)
            
            # Create verification object
            logger.info("Creating verification...")
            verification = SignalingVerification(model)
            
            # Create agent object with all components
            logger.info("Creating agent...")
            agent = Agent(
                meta_data=request_data,
                model=model,
                dataloader=dataloader,
                verification=verification
            )
            
            # Call build_and_store to train and save the model
            logger.info("Starting build_and_store...")
            success = agent.build_and_store()
            
            if success:
                logger.info(f"✅ Successfully trained and stored model: {request_data['agent_id']}")
            else:
                logger.error(f"❌ Failed to train model: {request_data['agent_id']}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing training request: {str(e)}")
            return False
    
    def _process_stop_request(self, request_data):
        """
        Process a stop request for a specific agent
        
        Args:
            request_data: Dictionary containing agent_id to stop
        
        Returns:
            bool: True if stop successful, False otherwise
        """
        try:
            agent_id = request_data.get('agent_id')
            logger.info(f"Processing stop request for agent: {agent_id}")
            
            # Get agent from active agents (thread-safe)
            with self.agents_lock:
                agent = self.active_agents.get(agent_id)
                if not agent:
                    logger.warning(f"⚠️ Agent {agent_id} is not running")
                    return False
            
            # Stop the agent
            logger.info(f"Stopping agent: {agent_id}...")
            agent.stop()
            
            # Remove from active agents (thread-safe)
            with self.agents_lock:
                if agent_id in self.active_agents:
                    del self.active_agents[agent_id]
            
            logger.info(f"✅ Successfully stopped agent: {agent_id}")
            logger.info(f"📊 Active agents: {len(self.active_agents)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing stop request: {str(e)}")
            return False
    
    def get_active_agents(self):
        """
        Get list of currently active agent IDs
        
        Returns:
            List[str]: List of active agent IDs
        """
        with self.agents_lock:
            return list(self.active_agents.keys())
    
    def _training_thread_wrapper(self, request_data, model_id):
        """
        Wrapper to run training in a separate thread
        
        Args:
            request_data: Dictionary containing training request parameters
            model_id: Model identifier for logging
        """
        try:
            logger.info(f"🧵 Thread started for training: {model_id}")
            success = self._process_training_request(request_data)
            
            if success:
                logger.info(f"✅ Thread completed successfully: {model_id}")
            else:
                logger.error(f"❌ Thread failed: {model_id}")
                
        except Exception as e:
            logger.error(f"❌ Thread error for {model_id}: {str(e)}")
        finally:
            # Clean up thread from active list (thread-safe)
            current_thread = threading.current_thread()
            with self.threads_lock:
                if current_thread in self.active_threads:
                    self.active_threads.remove(current_thread)
                active_count = len(self.active_threads)
            logger.info(f"🧹 Thread cleaned up: {model_id}. Active threads: {active_count}")
    
    def _launch_thread_wrapper(self, request_data, agent_id, reply_to=None, correlation_id=None):
        """
        Wrapper to run launch in a separate thread
        
        Args:
            request_data: Dictionary containing launch request parameters
            agent_id: Agent identifier for logging
            reply_to: Optional reply queue name for sending response
            correlation_id: Optional correlation ID for response matching
        """
        try:
            logger.info(f"🧵 Thread started for launch: {agent_id}")
            success = self._process_launch_request(request_data, reply_to, correlation_id)
            
            if success:
                logger.info(f"✅ Thread completed successfully: {agent_id}")
            else:
                logger.error(f"❌ Thread failed: {agent_id}")
                
        except Exception as e:
            logger.error(f"❌ Thread error for {agent_id}: {str(e)}")
        finally:
            # Clean up thread from active list (thread-safe)
            current_thread = threading.current_thread()
            with self.launch_threads_lock:
                if current_thread in self.active_launch_threads:
                    self.active_launch_threads.remove(current_thread)
                active_count = len(self.active_launch_threads)
            logger.info(f"🧹 Launch thread cleaned up: {agent_id}. Active launch threads: {active_count}")
    
    
    def _cleanup_finished_threads(self):
        """Remove finished threads from active list (thread-safe)"""
        with self.threads_lock:
            self.active_threads = [t for t in self.active_threads if t.is_alive()]
    
    def _cleanup_finished_launch_threads(self):
        """Remove finished launch threads from active list (thread-safe)"""
        with self.launch_threads_lock:
            self.active_launch_threads = [t for t in self.active_launch_threads if t.is_alive()]
    
    def run(self):
        """
        Start consuming training requests from RabbitMQ queue
        Main worker loop that waits for training jobs
        """
        try:
            # Establish connection
            logger.info(f"Connecting to RabbitMQ at {self.rabbitmq_host}:{self.rabbitmq_port}")
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            
            # Store channel reference for graceful shutdown
            with self.channel_lock:
                self.channel = channel
            
            # Declare queue (durable to survive broker restarts)
            channel.queue_declare(queue=self.training_queue_name, durable=True)
            logger.info(f"✅ Connected to queue: {self.training_queue_name}")
            
            # Declare launch queue (durable to survive broker restarts)
            channel.queue_declare(queue=self.launch_queue_name, durable=True)
            logger.info(f"✅ Connected to queue: {self.launch_queue_name}")
            
            def callback(ch, method, properties, body):
                """Callback for processing messages from queue"""
                try:
                    # Parse the training request
                    request_data = json.loads(body)
                    model_id = request_data.get('model_id', 'unknown')
                    logger.info(f"📥 Received training request: {model_id}")
                    
                    # Clean up finished threads before checking limit
                    self._cleanup_finished_threads()
                    
                    # Check if we've reached max concurrent trainings (thread-safe)
                    with self.threads_lock:
                        active_count = len(self.active_threads)
                    
                    if active_count >= self.max_concurrent_trainings:
                        logger.warning(f"⚠️ Max concurrent trainings ({self.max_concurrent_trainings}) reached. "
                                     f"Rejecting request: {model_id}")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                        return
                    
                    # Create and start thread for processing
                    training_thread = threading.Thread(
                        target=self._training_thread_wrapper,
                        args=(request_data, model_id),
                        name=f"Training-{model_id}",
                        daemon=False  # Non-daemon for graceful shutdown
                    )
                    
                    # Add to active threads (thread-safe)
                    with self.threads_lock:
                        self.active_threads.append(training_thread)
                        active_count = len(self.active_threads)
                    
                    # Start thread
                    training_thread.start()
                    
                    # Acknowledge message after thread starts successfully
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    logger.info(f"✅ Message acknowledged: {model_id}")
                    
                    logger.info(f"🚀 Training thread spawned for: {model_id}. "
                              f"Active threads: {active_count}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in message: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                except Exception as e:
                    logger.error(f"❌ Error in callback: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            def launch_callback(ch, method, properties, body):
                """Callback for processing launch requests from queue"""
                try:
                    # Parse the launch request
                    request_data = json.loads(body)
                    agent_id = request_data.get('agent_id', 'unknown')
                    logger.info(f"📥 Received launch request: {agent_id}")
                    
                    # Extract reply_to and correlation_id from properties
                    reply_to = properties.reply_to
                    correlation_id = properties.correlation_id
                    
                    # Clean up finished launch threads before checking limit
                    self._cleanup_finished_launch_threads()
                    
                    # Check if we've reached max concurrent launches (thread-safe)
                    with self.launch_threads_lock:
                        active_count = len(self.active_launch_threads)
                    
                    if active_count >= self.max_concurrent_launches:
                        logger.warning(f"⚠️ Max concurrent launches ({self.max_concurrent_launches}) reached. "
                                     f"Rejecting request: {agent_id}")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                        return
                    
                    # Create and start thread for processing
                    launch_thread = threading.Thread(
                        target=self._launch_thread_wrapper,
                        args=(request_data, agent_id, reply_to, correlation_id),
                        name=f"Launch-{agent_id}",
                        daemon=False  # Non-daemon for graceful shutdown
                    )
                    
                    # Add to active launch threads (thread-safe)
                    with self.launch_threads_lock:
                        self.active_launch_threads.append(launch_thread)
                        active_count = len(self.active_launch_threads)
                    
                    # Start thread
                    launch_thread.start()
                    
                    # Acknowledge message after thread starts successfully
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    logger.info(f"✅ Launch message acknowledged: {agent_id}")
                    
                    logger.info(f"🚀 Launch thread spawned for: {agent_id}. "
                              f"Active launch threads: {active_count}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in launch message: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                except Exception as e:
                    logger.error(f"❌ Error in launch callback: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # Set QoS - allow multiple messages to be prefetched for concurrent processing
            channel.basic_qos(prefetch_count=self.max_concurrent_trainings)
            
            # Start consuming messages from training queue
            channel.basic_consume(
                queue=self.training_queue_name,
                on_message_callback=callback
            )
            
            # Start consuming messages from launch queue
            channel.basic_consume(
                queue=self.launch_queue_name,
                on_message_callback=launch_callback
            )
            
            logger.info("🔄 Worker started. Waiting for training and launch requests...")
            logger.info(f"📊 Max concurrent trainings: {self.max_concurrent_trainings}")
            logger.info(f"� Max concurrent launches: {self.max_concurrent_launches}")
            logger.info(f"�📋 Training queue: {self.training_queue_name}")
            logger.info(f"🚀 Launch queue: {self.launch_queue_name}")
            logger.info("Press CTRL+C to stop")
            
            # Block and wait for messages
            channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("🛑 Worker stopped by user")
        except Exception as e:
            logger.error(f"❌ Worker error: {str(e)}")
        finally:
            # Clear channel reference
            with self.channel_lock:
                self.channel = None
            
            if 'connection' in locals() and connection.is_open:
                connection.close()
                logger.info("🔌 RabbitMQ connection closed")
    
    def stop(self):
        """Stop the worker and wait for active threads to complete"""
        logger.info("Stopping worker...")
        
        # Stop consuming from RabbitMQ
        with self.channel_lock:
            if self.channel and self.channel.is_open:
                try:
                    self.channel.stop_consuming()
                    logger.info("🛑 Stopped consuming from queue")
                except Exception as e:
                    logger.warning(f"Error stopping consumer: {str(e)}")
        
        # Stop all launched agents (thread-safe)
        with self.agents_lock:
            agents_to_stop = list(self.active_agents.values())
            agent_ids = list(self.active_agents.keys())
        
        if agents_to_stop:
            logger.info(f"🛑 Stopping {len(agents_to_stop)} active agents...")
            for agent_id, agent in zip(agent_ids, agents_to_stop):
                try:
                    agent.stop()
                    logger.info(f"✅ Stopped agent: {agent_id}")
                except Exception as e:
                    logger.error(f"❌ Error stopping agent {agent_id}: {str(e)}")
            
            # Clear agents dictionary
            with self.agents_lock:
                self.active_agents.clear()
        
        # Wait for active training threads to complete (thread-safe)
        with self.threads_lock:
            threads_to_wait = list(self.active_threads)  # Copy list
        
        if threads_to_wait:
            logger.info(f"⏳ Waiting for {len(threads_to_wait)} active training threads to complete...")
            for thread in threads_to_wait:
                if thread.is_alive():
                    thread.join(timeout=60)  # Wait up to 60 seconds per thread
            logger.info("✅ All training threads completed")
        
        # Wait for active launch threads to complete (thread-safe)
        with self.launch_threads_lock:
            launch_threads_to_wait = list(self.active_launch_threads)  # Copy list
        
        if launch_threads_to_wait:
            logger.info(f"⏳ Waiting for {len(launch_threads_to_wait)} active launch threads to complete...")
            for thread in launch_threads_to_wait:
                if thread.is_alive():
                    thread.join(timeout=60)  # Wait up to 60 seconds per thread
            logger.info("✅ All launch threads completed")


# Convenience function to run worker
def start_worker():
    """Start the signaling worker"""
    worker = SignalingWorker()
    worker.run()


if __name__ == "__main__":
    start_worker()

import time
import json
import threading
import logging
import os
import pika
from dotenv import load_dotenv
from ..mongoDB import MongoDBHandler
from ..postgres import PostgresHandler
from .config import SignalingConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, meta_data, model, dataloader, verification):
        self.meta_data = meta_data
        self.model = model
        self.dl = dataloader
        self.verification = verification
        self.config = SignalingConfig()
        
        # RabbitMQ setup
        self.rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
        self.rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
        self.rabbitmq_user = os.getenv('RABBITMQ_USER', 'guest')
        self.rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'guest')
        
        # Queue names
        agent_id = self.meta_data.get('agent_id', 'default')
        self.inference_queue_name = f'inference_queue_{agent_id}'
        
        # Thread control
        self.is_running = False
        self.inference_thread = None
        self.main_loop_thread = None
    
    def _get_rabbitmq_connection(self):
        """Create RabbitMQ connection"""
        credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
            credentials=credentials
        )
        return pika.BlockingConnection(parameters)

    def buid_and_store(self):
        dataloader = self.dl.fetch_training_dataloader_data()
        self.model.fit(dataloader)
        # Store the model (e.g., save to disk or database)
        MongoDBHandler.save_model(self.model, self.meta_data)
        PostgresHandler.update_model_metadata({'status': 'trained'}, self.meta_data['model_id'])

    
    def launch(self):
        """Launch the agent's main loop for continuous operation in a separate thread or process.
        This function also launches the inference loop that waits for user requests and processes them."""
        if self.is_running:
            logger.warning("Agent is already running")
            return
        
        self.is_running = True
        
        # Start main loop thread
        self.main_loop_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.main_loop_thread.start()
        logger.info("Main loop thread started")
        
        # Start inference consumer thread
        self.inference_thread = threading.Thread(target=self._consume_inference_queue, daemon=True)
        self.inference_thread.start()
        logger.info("Inference consumer thread started")

    def inference(self, request_data):
        """Publish inference request to RabbitMQ queue"""
        try:
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            
            # Declare queue
            channel.queue_declare(queue=self.inference_queue_name, durable=True)
            
            # Publish message
            message = json.dumps(request_data)
            channel.basic_publish(
                exchange='',
                routing_key=self.inference_queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                )
            )
            
            logger.info(f"Published inference request to queue: {self.inference_queue_name}")
            connection.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish inference request: {str(e)}")
            return False
    
    def _consume_inference_queue(self):
        """Wait on the RabbitMQ queue for inference requests made by the user"""
        try:
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            
            # Declare queue
            channel.queue_declare(queue=self.inference_queue_name, durable=True)
            
            def callback(ch, method, properties, body):
                try:
                    # Parse request
                    request_data = json.loads(body)
                    logger.info(f"Received inference request: {request_data}")
                    
                    # Get input data and confidence level
                    input_data = request_data.get('input_data')
                    confidence_level = request_data.get('confidence_level', 
                                                       self.config.default_confidence_level)
                    
                    # Make prediction
                    result = self.model.inference(input_data, confidence_level)
                    
                    # Verify prediction
                    verification_result = self.verification.verify(result)
                    
                    # Log result
                    logger.info(f"Inference result: {result}")
                    logger.info(f"Verification: {verification_result}")
                    
                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"Error processing inference request: {str(e)}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # Set QoS
            channel.basic_qos(prefetch_count=1)
            
            # Start consuming
            channel.basic_consume(
                queue=self.inference_queue_name,
                on_message_callback=callback
            )
            
            logger.info(f"Started consuming from queue: {self.inference_queue_name}")
            
            while self.is_running:
                try:
                    channel.connection.process_data_events(time_limit=1)
                except Exception as e:
                    logger.error(f"Error in consume loop: {str(e)}")
                    break
            
            connection.close()
            logger.info("Inference consumer stopped")
            
        except Exception as e:
            logger.error(f"Failed to start inference consumer: {str(e)}")
    def main_loop(self):
        """
        Main loop to continuously fetch data, make predictions, and verify them.
        This loop runs indefinitely, fetching new data at the specified frequency,
        making predictions with the model, and verifying those predictions.
        """
        freq = self.meta_data.get('frequency', self.config.default_frequency)
        
        while self.is_running:
            try:
                # Fetch new data for inference
                input_data = self.dl.fetch_inference_data()
                
                # Make predictions with the model
                confidence_level = self.meta_data.get('confidence_level', self.config.default_confidence_level)
                result = self.model.inference(input_data, confidence_level)
                
                # Verify predictions
                verification_result = self.verification.verify(result)
                
                # if results are good then send a signal to the user or execute a trade
                if result['estimated action'] == 1 and verification_result['is_valid']:
                    # Send buy signal or execute buy trade
                    self.config.send_signal('long')
                elif result['estimated action'] == -1 and verification_result['is_valid']:
                    # Send sell signal or execute sell trade
                    self.config.send_signal('short')
                
                # Log or store results as needed
                logger.info(f"Prediction: {result}, Verification: {verification_result}")
                
                # Wait for the next cycle based on the specified frequency
                time.sleep(freq)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(freq)
        
        logger.info("Main loop stopped")
    
    def stop(self):
        """Stop the agent"""
        logger.info("Stopping agent...")
        self.is_running = False
        
        if self.main_loop_thread:
            self.main_loop_thread.join(timeout=5)
        if self.inference_thread:
            self.inference_thread.join(timeout=5)
        
        logger.info("Agent stopped")


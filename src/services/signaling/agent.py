import time
import json
import threading
import logging
import os
import pika
from src.utils.env import load_env
from ...database_handlers.mongoDB import MongoDBService
from ...database_handlers.postgres import PostgreSQLService
from .config import SignalingConfig

load_env()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, meta_data=None, model=None, dataloader=None, verification=None,stop_event=None):
        self.meta_data = meta_data
        self.model = model
        self.dl = dataloader
        self.verification = verification
        self.config = SignalingConfig()
        self.model_id = self.meta_data.get('model_id', 'default_model_id')
        
        # RabbitMQ setup
        self.rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
        self.rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
        self.rabbitmq_user = os.getenv('RABBITMQ_USER', 'guest')
        self.rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'guest')
        
        # Queue names
        self.inference_queue_name = f'inference_queue_{self.model_id}'
        self.stop_queue = f'stop_queue_{self.model_id}'
        
        # Thread control (thread-safe)
        self.global_stop_event = stop_event # Event to signal threads to stop
        self.stop_event=threading.Event()  # Local event to control this agent's threads
        self.stop_event.set()  # Initially stopped
        self.inference_thread = None
        self.main_loop_thread = None
        self.mongo_service = MongoDBService()
        self.postgres_service = PostgreSQLService()
        self.client = None  # Placeholder for any external API client (e.g., trading API)
    
    def _get_rabbitmq_connection(self):
        """Create RabbitMQ connection"""
        credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
            credentials=credentials
        )
        return pika.BlockingConnection(parameters)

    def build_and_store(self):
        try:
            logger.info("Building dataloader and model...")
            dataloader, test_dataset = self.dl.fetch_dataloader()
            logger.info(f"Fetched dataloader with {len(dataloader)} batches and test dataset with {len(test_dataset)} samples")
            self.model.fit(dataloader)
            self.model.test(test_dataset)
            # Store the model (e.g., save to disk or database)
            self.mongo_service.push_agent_weights(model_id=self.meta_data['model_id'], weights=self.model.state_dict(), agent_name=self.meta_data['agent_name'])
            # self.postgres_service.update_model_metadata({'status': 'trained'}, self.meta_data['model_id'])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to train model: {self.meta_data.get('model_id', 'unknown')}. Error: {str(e)}")
            return False

    
    def launch(self):
        """Launch the agent's main loop for continuous operation in a separate thread or process.
        This function also launches the inference loop that waits for user requests and processes them."""
        if not self.stop_event.is_set():
            logger.warning("Agent is already running")
            return
        
        self.stop_event.clear()  # Agent is now running
        
        # Start main loop thread
        self.main_loop_thread = threading.Thread(target=self.main_loop, daemon=False)
        self.main_loop_thread.start()
        logger.info("Main loop thread started")
        
        # Start inference consumer thread
        self.inference_thread = threading.Thread(target=self._consume_inference_queue, daemon=False)
        self.inference_thread.start()
        logger.info("Inference consumer thread started")


        return self.main_loop_thread, self.inference_thread, self.stop_event

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
                    
                    # Prepare response
                    response = {
                        'inference_result': result,
                        'verification': verification_result,
                        'model_id': self.meta_data.get('model_id'),
                        'timestamp': time.time()
                    }
                    
                    # Log result
                    logger.info(f"Inference result: {result}")
                    logger.info(f"Verification: {verification_result}")
                    
                    # Send response to reply queue if specified
                    if properties.reply_to:
                        ch.basic_publish(
                            exchange='',
                            routing_key=properties.reply_to,
                            body=json.dumps(response),
                            properties=pika.BasicProperties(
                                correlation_id=properties.correlation_id,
                                delivery_mode=2
                            )
                        )
                        logger.info(f"Response sent to reply queue: {properties.reply_to}")
                    
                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"Error processing inference request: {str(e)}")
                    # Send error response if reply_to is specified
                    if properties.reply_to:
                        error_response = {
                            'error': str(e),
                            'model_id': self.meta_data.get('model_id'),
                            'timestamp': time.time()
                        }
                        try:
                            ch.basic_publish(
                                exchange='',
                                routing_key=properties.reply_to,
                                body=json.dumps(error_response),
                                properties=pika.BasicProperties(
                                    correlation_id=properties.correlation_id,
                                    delivery_mode=2
                                )
                            )
                        except Exception as pub_error:
                            logger.error(f"Failed to send error response: {str(pub_error)}")
                    
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # Set QoS
            channel.basic_qos(prefetch_count=1)
            
            

            
            logger.info(f"Started consuming from queue: {self.inference_queue_name}")
            
            channel.basic_consume(
                queue=self.inference_queue_name,
                on_message_callback=callback,
            )

            logger.info(f"Started consuming from queue: {self.inference_queue_name}")

            # Main loop: process messages and check stop_event
            while not (self.stop_event.is_set() or self.global_stop_event.is_set()):
                try:
                    # process pending events, but return every 1 second
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
        freq = self.meta_data.get('signal_frequency', self.config.default_frequency)
        
        while not (self.stop_event.is_set() or self.global_stop_event.is_set()):
            try:
                # Fetch new data for inference
                input_data = self.dl.fetch_inference_data()
                if input_data is None:
                    logger.warning("No input data fetched for inference")
                    time.sleep(freq)
                    continue
                # Make predictions with the model
                confidence_level = self.meta_data.get('confidence_level', self.config.default_confidence_level)
                result = self.model.inference(input_data, confidence_level)
                
                # Verify predictions
                verification_result = self.verification.verify(result)
                
                # if results are good then send a signal to the user or execute a trade
                if result['estimated action'] == 1 and verification_result['is_valid']:
                    # Send buy signal or execute buy trade
                    self.config.send_signal(message='long',destination=self.client)
                elif result['estimated action'] == -1 and verification_result['is_valid']:
                    # Send sell signal or execute sell trade
                    self.config.send_signal(message='short',destination=self.client)
                
                # Log or store results as needed
                logger.info(f"Prediction: {result}, Verification: {verification_result}")
                
                # Wait for the next cycle based on the specified frequency
                time.sleep(freq)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(freq)
        
        logger.info("Main loop stopped")
    



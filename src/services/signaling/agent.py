import time
import json
import threading
import logging
import os
from unittest import result
import pika
from src.utils.env import load_env
from ...database_handlers.mongoDB import MongoDBService
from ...database_handlers.postgres import PostgreSQLService
from .config import SignalingConfig
import pandas as pd

load_env()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position:
    def __init__(self, action, price, shares, timestamp, status='open', close_condition=0.02):
        self.action = action  # 'long' or 'short'
        self.price = price
        self.shares = shares
        self.timestamp = timestamp
        self.status = status
        self.close_condition = close_condition

    def check_close_condition(self, ts):
        if self.status == 'closed':
            return False

        if self.action == 'long':
            max_price = ts.max()
            return (max_price - self.price) / self.price >= self.close_condition
        elif self.action == 'short':
            min_price = ts.min()
            return (self.price - min_price) / self.price >= self.close_condition
        return False
    
    def close(self):
        self.status = 'closed'


class PortfolioManager:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_shares = 0
        self.open_positions = []
    def execute_trade(self, action, price, shares, close_condition=0.02):
        if action == 'long':
            cost = price * shares
            if self.current_capital >= cost:
                self.current_capital -= cost
                self.current_shares += shares
                logger.info(f"Executed long trade: Bought {shares} shares at {price}, Cost: {cost}")
            else:
                logger.warning("Insufficient capital for long trade")

            self.open_positions.append(Position(action, price, shares, time.time(),close_condition=close_condition))

        elif action == 'short':
            revenue = price * shares
            self.current_capital += revenue
            self.current_shares -= shares
            logger.info(f"Executed short trade: Sold {shares} shares at {price}, Revenue: {revenue}")
            self.open_positions.append(Position(action, price, shares, time.time(),close_condition=close_condition))
        else:
            logger.warning("Invalid trade action")


    def get_portfolio_value(self, current_price):
        return self.current_capital + self.current_shares * current_price

    def update_open_position(self, ts):

        for position in self.open_positions:
            if position.check_close_condition(ts[:, 0]):
                position.close()
                if position.action == 'long':
                    revenue = ts.max() * position.shares
                    self.current_capital += revenue
                    self.current_shares -= position.shares
                    logger.info(f"Closed long position: Sold {position.shares} shares at {ts.max()}, Revenue: {revenue}")
                elif position.action == 'short':
                    cost = ts.min() * position.shares
                    self.current_capital -= cost
                    self.current_shares += position.shares
                    logger.info(f"Closed short position: Bought {position.shares} shares at {ts.min()}, Cost: {cost}")

    

class Agent:
    def __init__(self, meta_data=None, model=None, dataloader=None, verification=None, stop_event=None):
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
        self.global_stop_event = stop_event
        self.stop_event = threading.Event()
        self.stop_event.set()  # Initially stopped
        self.inference_thread = None
        self.main_loop_thread = None
        self.mongo_service = MongoDBService()
        self.postgres_service = PostgreSQLService()
        self.client = None

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
            dataloader, test_dataset, label_percentages = self.dl.fetch_dataloader()
            logger.info(f"Fetched dataloader with {len(dataloader)} batches and test dataset with {len(test_dataset)} samples")
            self.model.fit(dataloader, label_percentages=label_percentages)
            self.mongo_service.push_agent_weights(
                model_id=self.meta_data['model_id'],
                weights=self.model.state_dict(),
                agent_name=self.meta_data['agent_name']
            )
            return True
        except Exception as e:
            logger.error(
                f"❌ Failed to train model: {self.meta_data.get('model_id', 'unknown')}. Error: {str(e)}",
                exc_info=True,  # ← traceback added
            )
            return False
    def _get_risk_params(self,meta_data, override_entry_price=None, override_account_value=None):
        """Extract S3 risk params from meta_data, with optional live overrides."""
        return {
            "risk_method":   meta_data.get("risk_method", "fixed_fractional"),
            "account_value": override_account_value or meta_data.get("account_value"),
            "entry_price":   override_entry_price   or meta_data.get("entry_price"),
            "risk_kwargs":   meta_data.get("risk_kwargs", {}),
        }
    def backtest(self, meta_data):
        manager    = PortfolioManager(meta_data.get('initial_capital', 100000))
        start_time = meta_data.get('start_time')
        end_time   = meta_data.get('end_time')

        data_wrapper = self.dl.fetch_alliged_data(start_time, end_time)

        freq = self.meta_data.get('signal_frequency', self.config.default_frequency)
        step = pd.Timedelta(seconds=freq)                # ← seconds, not microseconds

        current_time = pd.Timestamp(start_time) + step
        last_price   = None

        while current_time <= pd.Timestamp(end_time):
            try:
                start_window = current_time - step
                ts_slice, embd, mask = data_wrapper.get(start_window, current_time)  # ← unpack tuple

                if ts_slice is None or ts_slice.empty:
                    logger.warning(f"No input data for backtesting at {current_time}")
                    current_time += step
                    continue

                # ── Build input for model ────────────────────────────────────
                input_data = (
                    ts_slice,   # pass whatever your model.inference() expects
                    embd,
                    mask,
                )

                confidence_level = self.meta_data.get(
                    'confidence_level', self.config.default_confidence_level
                )

                result = self.model.inference(
                    input_data, confidence_level,
                    **self._get_risk_params(
                        meta_data,
                        override_entry_price   = last_price,               # live price
                        override_account_value = manager.current_capital,  # live capital
                    )
                )

                position_size = result.get('position_size') or 0.0
                if result['estimated_action'] == 'BUY' and position_size > 0:
                    manager.execute_trade('long',  price=last_price, shares=position_size)
                elif result['estimated_action'] == 'SELL' and position_size > 0:
                    manager.execute_trade('short', price=last_price, shares=position_size)

                pv = manager.get_portfolio_value(last_price)
                logger.info(
                    f"Backtest at {current_time}: action={result['estimated action']}"
                    f"  prob={result['probability']:.3f}"
                    f"  price={last_price:.4f}"
                    f"  portfolio={pv:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"Error in backtest loop at {current_time}: {str(e)}",
                    exc_info=True,
                )

            current_time += step

        final_pv = manager.get_portfolio_value(last_price) if last_price is not None else manager.current_capital
        logger.info(f"Backtesting completed. Final Portfolio Value: {final_pv:.2f}")
        return final_pv
    

    def launch(self):
        """Launch the agent's main loop and inference consumer in separate threads."""
        if not self.stop_event.is_set():
            logger.warning("Agent is already running")
            return

        self.stop_event.clear()

        self.main_loop_thread = threading.Thread(target=self.main_loop, daemon=False)
        self.main_loop_thread.start()
        logger.info("Main loop thread started")

        self.inference_thread = threading.Thread(target=self._consume_inference_queue, daemon=False)
        self.inference_thread.start()
        logger.info("Inference consumer thread started")

        return self.main_loop_thread, self.inference_thread, self.stop_event

    def _consume_inference_queue(self):
        """Wait on the RabbitMQ queue for inference requests made by the user"""
        try:
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            channel.queue_declare(queue=self.inference_queue_name, durable=True)

            def callback(ch, method, properties, body):
                try:
                    request_data     = json.loads(body)
                    input_data       = request_data.get('input_data')
                    confidence_level = request_data.get('confidence_level', self.config.default_confidence_level)

                    result              = self.model.inference(input_data, confidence_level, **self._get_risk_params())
                    verification_result = self.verification.verify(result)

                    response = {
                        'inference_result': result,
                        'verification': verification_result,
                        'model_id': self.meta_data.get('model_id'),
                        'timestamp': time.time()
                    }

                    logger.info(f"Inference result: {result}")
                    logger.info(f"Verification: {verification_result}")

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

                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except Exception as e:
                    logger.error(
                        f"Error processing inference request: {str(e)}",
                        exc_info=True,  # ← traceback added
                    )
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
                            logger.error(
                                f"Failed to send error response: {str(pub_error)}",
                                exc_info=True,  # ← traceback added
                            )

                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_qos(prefetch_count=1)
            logger.info(f"Started consuming from queue: {self.inference_queue_name}")

            channel.basic_consume(
                queue=self.inference_queue_name,
                on_message_callback=callback,
            )

            while not (self.stop_event.is_set() or self.global_stop_event.is_set()):
                try:
                    channel.connection.process_data_events(time_limit=1)
                except Exception as e:
                    logger.error(
                        f"Error in consume loop: {str(e)}",
                        exc_info=True,  # ← traceback added
                    )
                    break

            connection.close()
            logger.info("Inference consumer stopped")

        except Exception as e:
            logger.error(
                f"Failed to start inference consumer: {str(e)}",
                exc_info=True,  # ← traceback added
            )

    def main_loop(self):
        freq             = self.meta_data.get('signal_frequency', self.config.default_frequency)
        confidence_level = self.meta_data.get('confidence_level', self.config.default_confidence_level)

        while not (self.stop_event.is_set() or self.global_stop_event.is_set()):
            try:
                input_data = self.dl.fetch_inference_data()
                if input_data is None:
                    logger.warning("No input data fetched for inference")
                    time.sleep(freq)
                    continue

                result              = self.model.inference(input_data, confidence_level, **self._get_risk_params(self.meta_data))
                verification_result = self.verification.verify(result)

                if result['estimated_action'] == 'BUY' and verification_result['is_valid']:
                    self.config.send_signal(message='long', destination=self.client)
                elif result['estimated_action'] == 'SELL' and verification_result['is_valid']:
                    self.config.send_signal(message='short', destination=self.client)

                logger.info(f"Prediction: {result}, Verification: {verification_result}")
                time.sleep(freq)

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                time.sleep(freq)

        logger.info("Main loop stopped")
"""
Refinitiv Live Data Service
Provides real-time price movements and market data from Refinitiv APIs

Environment Variables Required:
- REFINITIV_API_KEY: Your Refinitiv API key (required)
- REFINITIV_USERNAME: Your username (optional, for password flow)
- REFINITIV_PASSWORD: Your password (optional, for password flow)  
- REFINITIV_BASE_URL: API base URL (optional, defaults to https://api.refinitiv.com)

Create a .env file in the project root with these variables.
See .env.example for template.
"""

import asyncio
import json
import logging
import os
import time
import websocket
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import requests
import threading
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Data class for price information"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: str
    timestamp: datetime
    change: float = 0.0
    change_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": f"{self.bid:.2f}",
            "ask": f"{self.ask:.2f}",
            "last": f"{self.last:.2f}",
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "change": f"{self.change:.2f}",
            "change_percent": f"{self.change_percent:.2f}%"
        }

class RefinitivLiveService:
    """Service for connecting to Refinitiv APIs and fetching live price data"""
    
    def __init__(self, api_key: str = None, username: str = None, password: str = None):
        # Load from environment variables if not provided
        self.api_key = api_key or os.getenv('REFINITIV_API_KEY')
        self.username = username or os.getenv('REFINITIV_USERNAME')
        self.password = password or os.getenv('REFINITIV_PASSWORD')
        
        # Validate required credentials
        if not self.api_key:
            raise ValueError("API key is required. Set REFINITIV_API_KEY in .env file or pass as parameter")
            
        self.session_token = None
        self.websocket = None
        self.is_connected = False
        self.subscribed_symbols = set()
        self.price_callbacks = []
        self.base_url = os.getenv('REFINITIV_BASE_URL', "https://api.refinitiv.com")
        
    def authenticate(self) -> bool:
        """Authenticate with Refinitiv RDP API"""
        try:
            auth_url = f"{self.base_url}/auth/oauth2/v1/token"
            
            # Using Machine-to-Machine credentials
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "scope": "trapi"
            }
            
            # If username/password provided, use Resource Owner Password Credentials
            if self.username and self.password:
                auth_data = {
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                    "client_id": self.api_key,
                    "scope": "trapi"
                }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            response = requests.post(auth_url, data=auth_data, headers=headers)
            
            if response.status_code == 200:
                token_data = response.json()
                self.session_token = token_data.get("access_token")
                logger.info("Successfully authenticated with Refinitiv")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def get_snapshot_prices(self, symbols: List[str]) -> List[PriceData]:
        """Get snapshot prices for given symbols"""
        if not self.session_token:
            if not self.authenticate():
                return []
        
        try:
            # Using Quotes endpoint for snapshot data
            endpoint = f"{self.base_url}/data/pricing/snapshots/v1/"
            
            headers = {
                "Authorization": f"Bearer {self.session_token}",
                "Content-Type": "application/json"
            }
            
            # Format symbols for the request
            universe = [{"RIC": symbol} for symbol in symbols]
            
            request_data = {
                "universe": universe,
                "fields": ["BID", "ASK", "TRDPRC_1", "ACVOL_1", "NETCHNG_1", "PCTCHNG"]
            }
            
            response = requests.post(endpoint, json=request_data, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_snapshot_data(data)
            else:
                logger.error(f"Failed to get snapshot prices: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting snapshot prices: {str(e)}")
            return []
    
    def _parse_snapshot_data(self, data: Dict) -> List[PriceData]:
        """Parse snapshot response data into PriceData objects"""
        price_data_list = []
        
        try:
            for instrument in data.get("data", []):
                ric = instrument.get("RIC", "")
                fields = instrument.get("fields", {})
                
                # Extract price fields with safe conversion
                bid = self._safe_float(fields.get("BID"))
                ask = self._safe_float(fields.get("ASK"))
                last = self._safe_float(fields.get("TRDPRC_1"))
                volume = str(fields.get("ACVOL_1", "0"))
                change = self._safe_float(fields.get("NETCHNG_1"))
                change_percent = self._safe_float(fields.get("PCTCHNG"))
                
                if bid and ask and last:
                    price_data = PriceData(
                        symbol=ric,
                        bid=bid,
                        ask=ask,
                        last=last,
                        volume=volume,
                        timestamp=datetime.now(),
                        change=change or 0.0,
                        change_percent=change_percent or 0.0
                    )
                    price_data_list.append(price_data)
                    
        except Exception as e:
            logger.error(f"Error parsing snapshot data: {str(e)}")
            
        return price_data_list
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is not None and value != "":
                return float(value)
        except (ValueError, TypeError):
            pass
        return None
    
    def start_streaming(self, symbols: List[str], callback: Callable[[PriceData], None] = None):
        """Start real-time streaming for given symbols"""
        if not self.session_token:
            if not self.authenticate():
                return False
        
        self.subscribed_symbols.update(symbols)
        
        if callback:
            self.price_callbacks.append(callback)
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self._connect_websocket)
        ws_thread.daemon = True
        ws_thread.start()
        
        return True
    
    def _connect_websocket(self):
        """Connect to Refinitiv WebSocket for real-time data"""
        try:
            # WebSocket URL for Refinitiv Real-Time Distribution System
            ws_url = "wss://api.refinitiv.com/streaming/pricing/v1/"
            
            headers = {
                "Authorization": f"Bearer {self.session_token}"
            }
            
            def on_message(ws, message):
                self._handle_websocket_message(message)
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
                self.is_connected = False
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                self.is_connected = True
                self._subscribe_to_symbols(ws)
            
            self.websocket = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            self.websocket.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            self.is_connected = False
    
    def _subscribe_to_symbols(self, ws):
        """Subscribe to real-time updates for symbols"""
        try:
            subscription_msg = {
                "Type": "Request",
                "Domain": "MarketPrice",
                "ID": 1,
                "Key": {
                    "Name": list(self.subscribed_symbols)
                }
            }
            
            ws.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to symbols: {list(self.subscribed_symbols)}")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {str(e)}")
    
    def _handle_websocket_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Parse real-time price updates
            if data.get("Type") == "Update" and data.get("Domain") == "MarketPrice":
                symbol = data.get("Key", {}).get("Name", "")
                fields = data.get("Fields", {})
                
                if symbol and fields:
                    price_data = self._parse_realtime_data(symbol, fields)
                    if price_data:
                        # Call all registered callbacks
                        for callback in self.price_callbacks:
                            try:
                                callback(price_data)
                            except Exception as e:
                                logger.error(f"Callback error: {str(e)}")
                                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def _parse_realtime_data(self, symbol: str, fields: Dict) -> Optional[PriceData]:
        """Parse real-time update into PriceData object"""
        try:
            bid = self._safe_float(fields.get("BID"))
            ask = self._safe_float(fields.get("ASK"))
            last = self._safe_float(fields.get("TRDPRC_1"))
            volume = str(fields.get("ACVOL_1", ""))
            change = self._safe_float(fields.get("NETCHNG_1"))
            change_percent = self._safe_float(fields.get("PCTCHNG"))
            
            if any([bid, ask, last]):  # At least one price should be available
                return PriceData(
                    symbol=symbol,
                    bid=bid or 0.0,
                    ask=ask or 0.0,
                    last=last or 0.0,
                    volume=volume,
                    timestamp=datetime.now(),
                    change=change or 0.0,
                    change_percent=change_percent or 0.0
                )
        except Exception as e:
            logger.error(f"Error parsing real-time data: {str(e)}")
            
        return None
    
    def add_price_callback(self, callback: Callable[[PriceData], None]):
        """Add a callback function for price updates"""
        self.price_callbacks.append(callback)
    
    def remove_price_callback(self, callback: Callable[[PriceData], None]):
        """Remove a callback function"""
        if callback in self.price_callbacks:
            self.price_callbacks.remove(callback)
    
    def stop_streaming(self):
        """Stop real-time streaming"""
        if self.websocket:
            self.websocket.close()
            self.is_connected = False
            logger.info("Stopped streaming")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            "is_connected": self.is_connected,
            "has_token": bool(self.session_token),
            "subscribed_symbols": list(self.subscribed_symbols),
            "active_callbacks": len(self.price_callbacks)
        }

# Example usage and testing functions
def price_update_handler(price_data: PriceData):
    """Example callback for handling price updates"""
    print(f"Price Update - {price_data.symbol}: "
          f"Last: {price_data.last:.2f}, "
          f"Bid: {price_data.bid:.2f}, "
          f"Ask: {price_data.ask:.2f}, "
          f"Change: {price_data.change:.2f} ({price_data.change_percent:.2f}%)")

def test_refinitiv_service():
    """Test function for the Refinitiv service"""
    try:
        # Initialize service using environment variables
        service = RefinitivLiveService()
        print(f"✅ Service initialized with API key: {service.api_key[:10]}...")
        
        # Test authentication
        if service.authenticate():
            print("✅ Authentication successful")
            
            # Test snapshot data
            symbols = ["AAPL.O", "MSFT.O", "GOOGL.O"]  # RIC codes
            snapshot_data = service.get_snapshot_prices(symbols)
            
            if snapshot_data:
                print("📊 Snapshot data:")
                for price in snapshot_data:
                    print(f"  {price.symbol}: {price.last:.2f}")
            
            # Test real-time streaming
            print("🔄 Starting real-time streaming...")
            service.add_price_callback(price_update_handler)
            
            if service.start_streaming(symbols):
                print("✅ Streaming started")
                
                # Let it run for a while
                try:
                    time.sleep(30)  # Stream for 30 seconds
                except KeyboardInterrupt:
                    pass
                finally:
                    service.stop_streaming()
                    print("🛑 Streaming stopped")
            
            # Print connection status
            status = service.get_connection_status()
            print(f"📋 Final status: {status}")
        
        else:
            print("❌ Authentication failed")
            
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        print("💡 Make sure to set REFINITIV_API_KEY in your .env file")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_refinitiv_service()

from src.server.flask_server import app, start_background_poller

# Start the background poller once (gunicorn worker post_fork via env guard)
import os
if os.environ.get('_POLLER_STARTED') != '1':
    os.environ['_POLLER_STARTED'] = '1'
    start_background_poller()

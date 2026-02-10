# FiKing Trader

**AI-Powered Trading Platform with Custom Agent Creation**

## Overview

FiKing Trader is an AI trading platform that allows users to:
- **Create** custom AI trading agents for specific equities (asynchronous training workflow)
- **Run predictions** using trained agents (synchronous inference workflow)
- Zero AI/programming knowledge required - finance-focused interface

## Architecture Documentation

📊 **[🌐 View Full Architecture (Live)](https://htmlpreview.github.io/?https://github.com/tradefi-2025/FiKing-Trader/blob/main/docs/architecture.html)** - Click to view interactive diagrams in your browser

Or download and open locally: [docs/architecture.html](docs/architecture.html)

**Interactive Diagrams Include:**
- System Architecture (5-layer design)
- Prediction Flow (sequence diagram)
- JSON Input-Output Contract
- Complete Workflow (Inference + Creation state machine)

## Repository Status

🚧 **This repository is being set up**

Currently contains:
- ✅ Architecture documentation and workflow diagrams
- 🔄 Code migration in progress from `demo_ai_logic_back`

## Technology Stack (Planned)

- **Frontend**: React (user dashboard, forms, results visualization)
- **Backend**: Flask API (REST endpoints)
- **Queue**: RabbitMQ (async job processing)
- **Workers**: Python workers (LSTM training, inference)
- **Storage**: PostgreSQL (agent metadata, predictions, user data)
- **ML**: PyTorch/TensorFlow (LSTM models, pre-trained base models)
- **Data Sources**: Refinitiv API, Bloomberg API (market data)

## Workflows

### 1. Inference (Synchronous)
User selects custom-trained agent → System fetches live market data → Computes features → Runs LSTM → Returns trade signal in seconds

### 2. Creation (Asynchronous)
User configures agent (equity, date range, features, risk params) → Training job queued → Worker fetches historical data → Trains LSTM → Evaluates → Saves weights or fails → Notifies user

## Getting Started

1. Open `docs/architecture.html` in your browser
2. Review the complete system design
3. Stay tuned for code migration updates

## Project Structure

```
FiKing-Trader/
├── README.md                 # This file
├── docs/                     # Documentation
│   └── architecture.html     # Interactive architecture diagrams
├── src/                      # Source code (coming soon)
├── tests/                    # Test suite (coming soon)
└── docker-compose.yml        # Container orchestration (coming soon)
```

## License

TBD

## Contributors

TBD

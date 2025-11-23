# CLAUDE.md

## 🔄 Project Awareness & Context
Always read [Planning.md](PLANNING.md) at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
Check TASK.md before starting a new task. If the task isn’t listed or file isnt created create it, add it with a brief description and today's date.
Use consistent naming conventions, file structure, and architecture patterns as described in PLANNING.md.
Use venv_linux (the virtual environment) whenever executing Python commands, including for unit tests.
## 🧱 Code Structure & Modularity
Never create a file longer than 500 lines of code. If a file approaches this limit, refactor by splitting it into modules or helper files.
Organize code into clearly separated modules, grouped by feature or responsibility. For agents this looks like:
agent.py - Main agent definition and execution logic
tools.py - Tool functions used by the agent
prompts.py - System prompts
Use clear, consistent imports (prefer relative imports within packages).
Use clear, consistent imports (prefer relative imports within packages).
Use python_dotenv and load_env() for environment variables.
## 🧪 Testing & Reliability
Always create Pytest unit tests for new features (functions, classes, routes, etc).
After updating any logic, check whether existing unit tests need to be updated. If so, do it.
Tests should live in a /tests folder mirroring the main app structure.
Include at least:
1 test for expected use
1 edge case
1 failure case
## ✅ Task Completion
Mark completed tasks in TASK.md immediately after finishing them.
Add new sub-tasks or TODOs discovered during development to TASK.md under a “Discovered During Work” section.
## 📎 Style & Conventions
Use Python as the primary language.
Follow PEP8, use type hints, and format with black.
Use pydantic for data validation.
Use FastAPI for APIs and SQLAlchemy or SQLModel for ORM if applicable.
Write docstrings for every function using the Google style:
def example():
    """
    Brief summary.

    Args:
        param1 (type): Description.

    Returns:
        type: Description.
    """
## 📚 Documentation & Explainability
Update README.md when new features are added, dependencies change, or setup steps are modified.
Comment non-obvious code and ensure everything is understandable to a mid-level developer.
When writing complex logic, add an inline # Reason: comment explaining the why, not just the what.
## 🧠 AI Behavior Rules
Never assume missing context. Ask questions if uncertain.
Never hallucinate libraries or functions – only use known, verified Python packages.
Always confirm file paths and module names exist before referencing them in code or tests.
Never delete or overwrite existing code unless explicitly instructed to or if part of a task from TASK.md.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Loyalty AI Agent system developed as an SPM (Software Project Management) semester project. The agent implements customer segmentation, churn prediction, and personalized reward optimization using RFM (Recency, Frequency, Monetary) analysis. The project demonstrates a Supervisor-Worker architecture pattern with RESTful API integration.

**Team Members:** Soban Ahmed, Abdul Moeiz, Muhammad Uzair
**Project Timeline:** Sep 22 - Dec 21, 2025 (84 days)
**Technology Stack:** Python 3.14, FastAPI, pandas, scikit-learn, numpy

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Generation
```bash
# Generate synthetic customer and transaction data (1000 customers, 10,000 transactions)
python src/data_generator.py
# Output: data/customers.json, data/transactions.json
```

### Running the Agent
```bash
# Run the loyalty agent demo (analyzes sample customers)
python src/loyalty_agent.py
# Output: Customer analysis with RFM scores, churn predictions, and reward recommendations
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_data_generator.py -v

# Run tests with detailed output
pytest tests/ -vv -s
```

### API Server (when implemented)
```bash
# Start the API server
uvicorn api.agent_api:app --reload --host 0.0.0.0 --port 5000

# Health check
curl http://localhost:5000/health

# Analyze customer
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST000000"}'
```

## Architecture

### Core Components

**1. Data Layer (`src/data_generator.py`)**
- Generates synthetic customer profiles with realistic distributions
- Creates transaction logs with temporal patterns
- Segments: Premium (10%), Regular (30%), Occasional (40%), New (20%)
- Loyalty Tiers: Gold, Silver, Bronze, Standard
- Outputs: JSON files in `data/` directory

**2. Agent Logic (`src/loyalty_agent.py`)**
- **RFM Analysis**: Calculates Recency, Frequency, Monetary scores (0-100 normalized)
  - Recency: Days since last purchase (exponential decay)
  - Frequency: Total purchases (normalized to 150 max)
  - Monetary: Lifetime value (normalized to 300,000 PKR max)
  - Combined RFM score: Weighted average (R: 30%, F: 35%, M: 35%)

- **Churn Prediction**: Multi-factor probability model (0-1 scale)
  - Recency risk (35% weight)
  - Frequency risk (25% weight)
  - Engagement risk (25% weight)
  - RFM risk (15% weight)

- **Customer Segmentation**: Advanced behavioral segments
  - Champion, At-Risk Champion
  - Loyal Customer, At-Risk Loyal
  - Potential Loyalist, Hibernating
  - New Customer, Lost Customer

- **Reward Optimization**: Contextual multi-armed bandit approach
  - 10 reward types with costs (PKR 50-1000)
  - Strategy selection based on segment and churn risk
  - ROI calculation using expected retention lift

**3. Memory Architecture** (to be implemented in `src/memory.py`)
- Short-term: In-memory cache (Redis or dict) for active sessions
- Long-term: SQLite database for historical interactions
- Audit trail: All agent decisions with timestamps

**4. API Layer** (to be implemented in `api/agent_api.py`)
- Framework: FastAPI
- Endpoints:
  - `POST /analyze`: Customer optimization recommendation
  - `GET /health`: Status, uptime, model accuracy
  - `POST /register`: Supervisor registry integration
  - `GET /metrics`: KPIs (retention rate, avg reward cost, churn reduction)

### Data Flow

1. **Data Generation**: `data_generator.py` → JSON files in `data/`
2. **Agent Initialization**: Load customers.json and transactions.json
3. **Analysis Request**: Customer ID → RFM calculation → Churn prediction → Segmentation → Reward recommendation
4. **API Response**: JSON with comprehensive analysis and actionable insights

### Key Algorithms

**RFM Scoring Formula:**
```python
recency_score = max(0, 100 - (days_since_purchase / 3.65))
frequency_score = min(100, (total_purchases / 150) * 100)
monetary_score = min(100, (lifetime_value / 300000) * 100)
rfm_score = (recency_score * 0.3 + frequency_score * 0.35 + monetary_score * 0.35)
```

**Churn Probability Formula:**
```python
churn_probability = (
    recency_risk * 0.35 +
    frequency_risk * 0.25 +
    engagement_risk * 0.25 +
    rfm_risk * 0.15
)
```

## Project Structure

```
/project-root
├── src/
│   ├── data_generator.py      # Synthetic data generation
│   ├── loyalty_agent.py        # Core AI agent logic
│   ├── memory.py               # (TODO) Short/long-term memory
│   ├── logger.py               # (TODO) Centralized logging
│   └── registry_client.py      # (TODO) Supervisor-Worker integration
├── api/
│   └── agent_api.py            # (TODO) FastAPI endpoints
├── tests/
│   ├── test_data_generator.py  # (TODO) Data generation tests
│   ├── test_agent_logic.py     # (TODO) Agent logic tests
│   ├── test_api.py             # (TODO) API integration tests
│   └── test_memory.py          # (TODO) Memory system tests
├── data/
│   ├── customers.json          # Generated customer profiles
│   └── transactions.json       # Generated transaction logs
├── docs/                       # (TODO) API contract, integration guide
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Implementation Status

**Phase 1: Foundation (Completed)**
- ✅ Repository structure
- ✅ Data generator (1000 customers, 10,000 transactions)
- ✅ Core agent logic (RFM, segmentation, churn prediction, reward optimization)

**Phase 2: API & Integration (In Progress)**
- ⏳ HTTP API with FastAPI
- ⏳ Memory architecture (Redis + SQLite)
- ⏳ Supervisor-Worker registry integration
- ⏳ Logging system

**Phase 3: Testing & Documentation (Planned)**
- ⏳ Unit and integration tests (>80% coverage target)
- ⏳ API contract documentation
- ⏳ System architecture diagram
- ⏳ Integration guide

## Development Guidelines

### Adding New Features

**When adding reward types:**
- Update `REWARD_CATALOG` in `loyalty_agent.py`
- Include: name, cost (PKR), and value description
- Update reward selection logic in `recommend_reward()` method

**When modifying RFM weights:**
- Adjust weights in `calculate_rfm_score()` (must sum to 1.0)
- Consider impact on segmentation thresholds
- Validate with sample customer analysis

**When adding new customer segments:**
- Update segment logic in `segment_customer()` method
- Add corresponding reward strategies in `recommend_reward()`
- Update data generator segment distributions if needed

### Testing Approach

- Unit tests: Test individual methods in isolation
- Integration tests: Test data generation → agent analysis → API response flow
- Validation tests: Check data distributions, RFM score ranges, churn probability bounds
- Mock external dependencies (registry, database) in tests

### Code Style

- Use type hints for function parameters and return values
- Docstrings: Google style with Args, Returns, and description
- Constants: UPPER_CASE for class-level constants
- Private methods: Prefix with underscore `_method_name()`
- JSON outputs: Use 2-space indentation for readability

## Critical Paths & Dependencies

**Critical Path Activities** (from network_diagram_analysis.md):
1.1 → 1.2 → 2.1/2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2

**Non-Critical Activities with Slack:**
- Data Generator (4.1): 7 days total slack
- API & Dashboard (4.3): 7 days total and free slack

**Merge Points:**
- Day 14: Requirements Analysis + Scope Definition → Gantt Chart
- Day 49: Data Generator + Agent Logic Design → Agent Core
- Day 63: Agent Core + API/Dashboard → Testing

## Common Pitfalls

1. **Data loading errors**: Ensure `data/customers.json` and `data/transactions.json` exist before running agent analysis. Run `data_generator.py` first if files are missing.

2. **RFM score calculations**: Customer transactions must have `status: "Completed"` to be included in RFM calculations. Failed transactions are filtered out.

3. **Churn prediction thresholds**: Thresholds are defined in `CHURN_THRESHOLDS` constant. Modify these based on business requirements, not arbitrary values.

4. **Reward ROI calculations**: Expected retention lift is capped at 30% (confidence * 0.3). This prevents unrealistic ROI projections for high-confidence recommendations.

5. **Date parsing**: All dates in JSON use ISO format (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`). Use `datetime.strptime()` with correct format strings.

## Project Management Context

This project follows established PM artifacts:
- **Budget (BAC)**: 1,726,725 PKR
- **Timeline**: 84 days (Sep 22 - Dec 21, 2025)
- **Earned Value**: CPI=0.882, SPI=0.833 (as of 3-month assessment)
- **Risk**: 87.5% of activities are critical path (minimal schedule buffer)

When implementing features, prioritize critical path items and maintain alignment with project timeline and cost constraints.

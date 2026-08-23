# TuxGuard AI

### Autonomous Fault Detection & Self-Healing for Linux Systems

**Monitor • Detect • Explain • Predict • Heal**

TuxGuard AI is an AI-powered Linux operations and system reliability platform that continuously monitors a Linux system, detects anomalies and faults, predicts potential failures, explains issues using real system evidence, and provides controlled recovery recommendations.

It combines real-time system monitoring, intelligent natural-language routing, local LLM reasoning, explainable AI, predictive analysis, and safe self-healing into a unified dashboard.

---

## 🚀 Key Features

### 🖥️ Real-Time System Monitoring
- CPU utilization
- Memory/RAM usage
- Disk utilization
- System uptime
- Running processes
- Linux services
- Docker containers

### 🔍 Intelligent Fault Detection
Detects potential issues such as:
- High CPU or memory utilization
- Disk usage anomalies
- Failed services
- Docker/container issues
- Hardware health problems
- Driver anomalies
- Kernel-related issues

### 📈 Failure Prediction
Analyzes system trends to identify potential failures involving:
- CPU
- Memory
- Disk
- Overall system stability

### 🤖 Natural-Language AI Operations Assistant
Users can ask questions in natural language, for example:

```text
What is my current CPU usage?
What percentage of RAM is currently being used?
Are there any active system faults?
Are there any driver anomalies?
Are there any predicted memory failures?
```

The assistant identifies the intent, gathers relevant system evidence, and generates a structured response.

### 🧠 Explainable AI
Responses are designed to provide:
- A plain-language explanation
- Supporting system evidence
- Recommended commands/actions
- Confidence score
- Reasoning/basis for the answer

### 🛡️ Safe Self-Healing
The recovery workflow follows:

```text
Detect → Diagnose → Explain → Recommend
       → Safety Check → User Approval
       → Execute → Verify
```

The system is designed to keep recovery actions controlled rather than blindly executing arbitrary commands.

### 🔐 Local & Privacy-Focused AI
TuxGuard AI uses a local Ollama deployment with `llama3.2:3b`, keeping system information on the Linux machine instead of requiring an external LLM API for inference.

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      User/Admin      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Web Dashboard      │
                         │   + AI Chat          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   FastAPI Backend    │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
       ┌────────────────────┐               ┌────────────────────┐
       │ Intent & Ops       │               │ Monitoring &       │
       │ Classification     │               │ Detection Engine   │
       └─────────┬──────────┘               └─────────┬──────────┘
                 │                                    │
                 ▼                                    ▼
       ┌────────────────────┐               ┌────────────────────┐
       │ Context Builder    │               │ CPU / RAM / Disk   │
       │ + Live Evidence    │               │ Hardware / Driver  │
       └─────────┬──────────┘               │ Docker / Services  │
                 │                          │ Failure Prediction │
                 │                          └─────────┬──────────┘
                 └──────────────────┬─────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Local AI Engine      │
                         │ Ollama + Llama 3.2   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Explainable Response │
                         │ Evidence + Confidence│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Safe Action Layer    │
                         │ Approval + Execution │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Recovery & Verify    │
                         └──────────────────────┘
```

---

## 🔄 AI Query Processing

For a question such as:

> **What is my current CPU usage?**

the application follows:

```text
User Question
     ↓
Intent Classification
     ↓
Performance Intent
     ↓
System Context Collection
     ↓
Real CPU/System Evidence
     ↓
Local LLM
     ↓
Structured JSON Response
     ↓
Explanation + Confidence + Recommendation
```

TuxGuard AI also distinguishes between overall resource questions and specific process-level questions to avoid incorrectly answering an overall CPU/RAM question with a single process metric.

---

## 🧩 Intelligent Routing

TuxGuard AI uses two complementary processing paths.

### Detector-Backed Pipeline

Used for specialized queries such as:
- Hardware health
- Driver health
- Failure prediction

```text
Query
 ↓
Intent Classifier
 ↓
Detector Context
 ↓
Real System Evidence
 ↓
Local LLM
 ↓
Explainable Response
```

### Tool-Grounded Operations Pipeline

Used for specific operational questions such as:
- Top CPU processes
- Top memory processes
- Disk usage
- Services
- Logs
- Network/listening ports

```text
Query
 ↓
Ops Intent Classifier
 ↓
Whitelisted Read-Only Tool
 ↓
Live System Output
 ↓
Local LLM
 ↓
Explainable Response
```

This routing prevents broad questions such as overall CPU/RAM usage or multi-domain health questions from being incorrectly handled by a narrow process-level tool.

---

## 🤖 Local AI Engine

| Component | Technology |
|---|---|
| LLM Runtime | Ollama |
| Model | Llama 3.2:3B |
| Backend | Python + FastAPI |
| Monitoring | psutil + Linux utilities |
| Database | SQLite |
| Frontend | HTML / CSS / JavaScript |
| Container Monitoring | Docker |
| Operating System | Linux / Ubuntu |

The LLM runs locally through Ollama.

---

## 🛡️ Safety Model

Safety is a core design principle.

TuxGuard AI:
- Uses monitoring data as system evidence.
- Favors read-only diagnostic commands.
- Classifies recommended actions by risk.
- Avoids uncontrolled destructive actions.
- Provides controlled recovery workflows.
- Supports user approval before recovery actions.
- Verifies recovery where applicable.
- Maintains logs for operational visibility.

---

## 📊 Monitoring & Detection

```text
System Health
│
├── CPU Monitoring
├── Memory Monitoring
├── Disk Monitoring
├── Process Monitoring
├── Service Monitoring
├── Docker Monitoring
├── Hardware Monitoring
├── Driver Monitoring
├── Kernel Health
├── Log Analysis
└── Failure Prediction
```

---

## 🔮 Predictive Analysis

The predictive layer evaluates system trends to identify potential issues before they become critical.

Example:

```text
Increasing Memory Usage
          ↓
Trend Analysis
          ↓
Potential Failure
          ↓
Prediction Alert
          ↓
AI Explanation
          ↓
Recommended Investigation
```

---

## 🔧 Self-Healing Workflow

When an issue is detected:

```text
System Monitoring
       ↓
Fault Detection
       ↓
Evidence Collection
       ↓
AI Diagnosis
       ↓
Confidence Assessment
       ↓
Recommended Fix
       ↓
Safety Validation
       ↓
User Approval
       ↓
Controlled Execution
       ↓
Post-Fix Verification
```

This combines automation with human control and helps reduce the risk of inappropriate recovery actions.

---

## ⚡ Performance & Reliability

The local inference configuration is tuned for a CPU-based demonstration environment.

Current safeguards include:
- Controlled generation token limit
- Compact prompt/context handling
- Local model execution
- Separate timeout handling for background diagnosis
- Structured JSON response parsing
- Graceful fallback for malformed model responses
- Controlled background diagnosis
- Intent-specific context gathering

These mechanisms help keep interactive requests responsive while background fault diagnosis is running.

---

## 📁 Project Structure

```text
TuxGuard-AI/
│
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── services/
│   │       ├── ai_assistant.py
│   │       ├── context_builder.py
│   │       ├── intent_classifier.py
│   │       ├── ops_intent_classifier.py
│   │       ├── ops_assistant.py
│   │       ├── tool_executor.py
│   │       ├── ollama_client.py
│   │       ├── llm_json.py
│   │       ├── prompts.py
│   │       ├── ops_prompts.py
│   │       ├── system_monitor.py
│   │       ├── hardware_monitor.py
│   │       ├── driver_monitor.py
│   │       ├── failure_predictor.py
│   │       ├── issue_alert_store.py
│   │       ├── fix_engine.py
│   │       ├── docker_monitor.py
│   │       └── conversation_store.py
│   │
│   └── logs/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── vendor/
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TuxGuard-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama on the Linux machine and pull the configured model:

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

### 5. Start Ollama

```bash
ollama serve
```

If Ollama is configured as a system service:

```bash
sudo systemctl start ollama
```

Verify:

```bash
systemctl status ollama
```

---

## ▶️ Run the Application

From the project root:

```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Open the project's frontend using the configured frontend route.

---

## 💬 Example Questions

Try these questions in the AI Chat:

```text
What is my current CPU usage?
```

```text
What percentage of RAM is currently being used?
```

```text
What is my current CPU, memory, and disk usage?
```

```text
Are there any active system faults?
```

```text
Are there any driver anomalies?
```

```text
Are there any predicted memory failures?
```

```text
What are the current CPU and GPU temperatures?
```

---

## 🌟 Key Advantages

| Capability | TuxGuard AI |
|---|---|
| Real-time monitoring | ✅ |
| Natural-language system queries | ✅ |
| Local LLM inference | ✅ |
| Explainable AI | ✅ |
| Fault detection | ✅ |
| Failure prediction | ✅ |
| Hardware monitoring | ✅ |
| Driver monitoring | ✅ |
| Docker monitoring | ✅ |
| Safe recovery recommendations | ✅ |
| Controlled self-healing | ✅ |
| User approval workflow | ✅ |
| Recovery verification | ✅ |
| External LLM API required | ❌ |

---

## 🎯 Impact

TuxGuard AI aims to move Linux administration from a primarily reactive monitoring workflow toward a proactive, explainable, and AI-assisted operations model.

It helps administrators:
- Detect issues earlier
- Understand system problems using real evidence
- Reduce manual troubleshooting
- Identify potential failures proactively
- Receive context-aware recommendations
- Perform controlled recovery actions
- Verify recovery results
- Keep sensitive system information local

---

## 🔮 Future Enhancements

Potential future enhancements include:
- Multi-server Linux fleet monitoring
- Advanced root-cause analysis
- Distributed incident correlation
- Improved predictive models
- Automated incident prioritization
- Role-based access control
- Enterprise notification integrations
- Long-term system health analytics
- Learning from verified remediation outcomes

---

## 📜 License

Add the appropriate project license before publishing the repository.

---

## 🐧 TuxGuard AI

**Monitor. Detect. Explain. Predict. Heal.**

> **Your Linux. Protected by AI.**

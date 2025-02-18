# 🚀 LLM Model Gateway

[![PyPI](https://img.shields.io/pypi/v/llm-model-gateway.svg)](https://pypi.org/project/llm-model-gateway/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/irthomasthomas/llm-model-gateway/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

A lightweight, OpenAI-compatible API gateway for [simonw's llm cli](https://github.com/simonw/llm). This gateway provides a unified interface for model interactions with robust logging and metrics.

## ✨ Features

- 🔄 **OpenAI API Compatibility**: Seamless integration with existing tools
- 🌊 **Streaming Responses**: Real-time, chunked responses
- 📊 **Comprehensive Metrics**: Track model performance and usage
- 🎯 **Model Agnostic**: Support for all [LLM](https://llm.datasette.io/en/stable/) models
- 📝 **Persistent Logging**: SQLite-based metrics tracking and prompt response logging.

## 🚀 Installation

```bash
pip install llm
llm install llm-model-gateway
```

## 🔧 Quick Start

### Starting the Server

```bash
# Serve all available models
llm serve

# Serve specific model
llm serve -m gpt-4

# Custom host and port
llm serve -h 0.0.0.0 -p 8080 --reload
```

#### 1. List Models

```bash
curl http://localhost:8000/v1/models
```
#### 2. Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Example response:
```json
{
  "id": "chatcmpl-8c96c0cf-f166-4cdf-8132-d6ddefaed27c",
  "object": "chat.completion",
  "created": 1735505968,
  "model": "gemini-pro",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hi there! How can I help you today?"
    },
    "finish_reason": "stop"
  }]
}
```

### API Usage

The gateway provides two main endpoints:

#### 1. Chat Completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key not checked
)

# Non-streaming request
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4"
)
print(response.choices[0].message.content)

# Streaming request
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4",
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## ⚙️ Configuration

### Environment Variables
- `LLM_USER_PATH`: Custom directory for logs and data
  - Default: System-specific app directory
- Logs Generated:
  - `llm_model_gateway.log`: Event logging
  - `logs.db`: SQLite metrics database

## 📊 Metrics

Every request is logged with:
- 🆔 Unique request ID
- 🕒 Timestamp
- 🤖 Model used
- ⏱️ Processing duration
- 🔢 Token count
- ✅ Success/failure status
- ❌ Error details (if any)

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/irthomasthomas/llm-model-gateway
cd llm-model-gateway

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## 🚨 Troubleshooting

Common issues and solutions:

- **Connection refused**: Check host/port settings
- **Model not found**: Verify model is registered with LLM
- **Streaming issues**: Confirm client streaming compatibility

## 🤝 Contributing

Contributions welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Pull requests
- Documentation improvements

## ⚖️ License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
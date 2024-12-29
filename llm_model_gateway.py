import click
import llm 
import sqlite_utils
import logging
import sys
import os
import pathlib
from datetime import datetime
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import uuid
import time

def user_dir() -> pathlib.Path:
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path

# Configure logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(str(user_dir() / "llm_model_gateway.log"))
        ]
    )
    return logging.getLogger("llm_model_gateway")

logger = setup_logging()

def logs_db_path():
    return user_dir() / "logs.db"

class ServerMetrics:
    def __init__(self):
        self.db = sqlite_utils.Database(logs_db_path())
        self._ensure_tables()
    
    def _ensure_tables(self):
        self.db["server_metrics"].create({
            "id": str,
            "timestamp": str,
            "model": str,
            "duration": float,
            "tokens": int,
            "success": bool,
            "error": str
        }, pk="id", if_not_exists=True)
    
    def log_request(self, model: str, duration: float, tokens: int, success: bool, error: str = None):
        self.db["server_metrics"].insert({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "duration": duration,
            "tokens": tokens,
            "success": success,
            "error": error
        })

# FastAPI app and routes
app = FastAPI(title="LLM Server", version="0.1.0")
metrics = ServerMetrics()

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    start_time = time.time()
    model_id = request.get("model")
    messages = request.get("messages", [])
    stream_flag = request.get("stream", False)
    prompt_content = messages[-1].get("content") if messages else None

    # Extract system prompt if present
    system_prompt = None
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
            break

    options = {k: v for k, v in request.items() if k not in ["model", "messages", "stream"]}

    # Get database connection for logging
    db = sqlite_utils.Database(logs_db_path())
    llm.migrations.migrate(db)

    try:
        model = llm.get_model(model_id)
        
        # Create prompt and response objects
        prompt_obj = llm.Prompt(
            prompt_content,
            model=model,
            system=system_prompt,
            options=model.Options(**options)
        )
        
        conversation = None
        if not stream_flag:  # Only create conversation if not streaming
            conversation = llm.Conversation(model=model)

        response_obj = llm.Response(
            prompt_obj,
            model=model,
            stream=stream_flag,
            conversation=conversation
        )
        response_obj._prompt_json = json.dumps(messages)  # Store original messages

        # Log the prompt details
        response_obj.log_to_db(db)

        if stream_flag:
            return StreamingResponse(
                stream_response(response_obj),
                media_type="text/event-stream"
            )

        # For non-streaming, get full response
        full_response_text = response_obj.text()
        duration = time.time() - start_time

        # Log metrics (maintaining backward compatibility)
        metrics.log_request(
            model=model_id,
            duration=duration,
            tokens=len(prompt_content.split()) if prompt_content else 0,
            success=True
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response_text
                },
                "finish_reason": "stop"
            }]
        }

    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            model=model_id,
            duration=duration,
            tokens=0,
            success=False,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(response_obj):
    for chunk in response_obj:
        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": model.model_id}
            for model in llm.get_models()
        ]
    }

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("model", required=False)
    @click.option("-h", "--host", default="127.0.0.1", help="Host to bind to")
    @click.option("-p", "--port", default=8000, help="Port to listen on")
    @click.option("--reload", is_flag=True, help="Enable auto-reload")
    def serve(model, host, port, reload):
        """Start an OpenAI-compatible API server for LLM models."""
        logger.info(f"Starting LLM server on {host}:{port}")
        if model:
            logger.info(f"Serving model: {model}")
            # Validate model exists
            llm.get_model(model)
            
        uvicorn.run(
            "llm_model_gateway:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
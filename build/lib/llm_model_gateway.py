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
def setup_logging(level=logging.DEBUG):
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
    
    logger.debug(f"Gateway request for model '{model_id}' with stream: {stream_flag}")
    logger.debug(f"Incoming messages: {json.dumps(messages, indent=2)}")

    # Filter options to prevent validation errors
    valid_options = [
        'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty',
        'stop', 'n', 'logit_bias', 'user', 'seed', 'tools', 'tool_choice', 'response_format'
    ]
    options = {k: v for k, v in request.items() 
               if k not in ["model", "messages", "stream"] and k in valid_options}

    # Get database connection for logging
    db = sqlite_utils.Database(logs_db_path())
    llm.migrations.migrate(db)

    try:
        model = llm.get_model(model_id)
        
        # Extract system prompt and build conversation context
        system_prompt = None
        context_parts = []
        final_user_message = None
        
        # Process messages to build context and find final user message
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif i == len(messages) - 1 and role == "user":
                # This is the final user message we need to respond to
                final_user_message = content
            elif role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        
        if not final_user_message:
            raise HTTPException(status_code=400, detail="No user message to respond to")
        
        # Create conversation
        conversation = llm.Conversation(model=model)
        
        # Build the full prompt with context
        if context_parts:
            context = "\n".join(context_parts)
            full_prompt = f"Previous conversation:\n{context}\n\nUser: {final_user_message}"
        else:
            full_prompt = final_user_message
        
        # Generate response
        response = conversation.prompt(full_prompt, system=system_prompt, stream=stream_flag, **options)

        if stream_flag:
            async def stream_response():
                for chunk in response:
                    if chunk:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            duration = time.time() - start_time
            response_text = response.text()
            
            # Log metrics
            metrics.log_request(
                model=model_id,
                duration=duration,
                tokens=len(response_text.split()) if response_text else 0,
                success=True
            )
            
            return {
                "id": getattr(response, 'id', str(uuid.uuid4())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": getattr(response, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response, 'response_tokens', 0),
                    "total_tokens": getattr(response, 'prompt_tokens', 0) + getattr(response, 'response_tokens', 0)
                }
            }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error processing request: {e}", exc_info=True)
        
        metrics.log_request(
            model=model_id,
            duration=duration,
            tokens=0,
            success=False,
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=str(e))

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
            log_level="debug"
        )

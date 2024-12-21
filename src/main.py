import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from src.agent_logic import run_agent_with_persistence, checkpointer  # Import with checkpointing
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env
load_dotenv()

# Define the FastAPI app
app = FastAPI()

def make_serializable(obj):
    """
    Convert objects to JSON serializable formats.
    """
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)  # Fallback for unsupported types


# REST Endpoint for synchronous invoke with thread persistence
@app.post("/invoke")
def invoke_graph(input_data: Dict[str, Any]):
    """
    REST API to invoke the graph synchronously with checkpoint persistence.
    """
    thread_id = input_data.get("thread_id", "default-thread")
    try:
        response = run_agent_with_persistence(input_data, thread_id)
        return JSONResponse(content=make_serializable(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# REST Endpoint for streaming results
@app.post("/stream")
async def stream_graph(input_data: Dict[str, Any]):
    """
    REST API to stream intermediate steps of the graph with persistence.
    """
    thread_id = input_data.get("thread_id", "default-thread")
    response_list = []
    try:
        stream = run_agent_with_persistence(input_data, thread_id)
        for event in stream:
            response_list.append(make_serializable(event))
        return JSONResponse(content={"stream_response": response_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to interact with the graph in real-time.
    Supports invoke and stream modes with checkpointing.
    """
    await websocket.accept()
    try:
        while True:
            # Receive a message from the frontend
            data = await websocket.receive_json()

            # Extract thread_id or assign default
            thread_id = data.get("thread_id", "default-thread")
            mode = data.get("mode", "invoke")
            input_data = {"messages": data["messages"]}

            if mode == "invoke":
                # Synchronous invoke mode with persistence
                response = run_agent_with_persistence(input_data, thread_id)
                await websocket.send_json({"response": make_serializable(response)})
            
            elif mode == "stream":
                # Streaming mode with thread checkpointing
                try:
                    stream = run_agent_with_persistence(input_data, thread_id)
                    for event in stream:
                        await websocket.send_json({"stream_response": make_serializable(event)})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        # Send error message and close the WebSocket
        await websocket.send_json({"error": str(e)})
        await websocket.close()

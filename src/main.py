import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from src.agent_logic import graph  # Import the graph directly
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

# REST Endpoint for invoke
@app.post("/invoke")
def invoke_graph(input_data: Dict[str, Any]):
    """
    REST API to invoke the graph synchronously.
    """
    try:
        thread_id = input_data.get("thread_id", "default")  # Use provided thread_id or default
        config = {"configurable": {"thread_id": thread_id}}
        response = graph.invoke(input=input_data, config=config)
        return JSONResponse(content=make_serializable(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# REST Endpoint for streaming results
@app.post("/stream")
async def stream_graph(input_data: Dict[str, Any]):
    """
    REST API to stream intermediate steps of the graph.
    """
    response_list = []
    try:
        thread_id = input_data.get("thread_id", "default")
        config = {"configurable": {"thread_id": thread_id}}
        for event in graph.stream(input=input_data, config=config, stream_mode="values"):
            response_list.append(make_serializable(event))
        return JSONResponse(content={"stream_response": response_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to interact with the graph in real-time.
    Supports invoke and stream modes.
    """
    await websocket.accept()
    try:
        while True:
            # Receive a message from the frontend
            data = await websocket.receive_json()

            # Expecting data to be {"mode": "invoke" or "stream", "messages": [{"role": "user", "content": "..."}]}
            mode = data.get("mode", "invoke")
            thread_id = data.get("thread_id", "default")
            input_data = {"messages": data["messages"]}
            config = {"configurable": {"thread_id": thread_id}}

            if mode == "invoke":
                # Synchronous invoke mode
                response = graph.invoke(input=input_data, config=config)
                await websocket.send_json({"response": make_serializable(response)})
            elif mode == "stream":
                # Streaming mode
                try:
                    async for event in graph.astream(input=input_data, config=config, stream_mode="values"):
                        await websocket.send_json({"stream_response": make_serializable(event)})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        # Send error message and close the WebSocket
        await websocket.send_json({"error": str(e)})
        await websocket.close()

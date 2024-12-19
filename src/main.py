import json
from fastapi import FastAPI, WebSocket
from src.agent_logic import graph  # Import the graph directly
from dotenv import load_dotenv

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to interact with the graph in real time.
    """
    await websocket.accept()
    try:
        while True:
            # Receive a message from the frontend
            data = await websocket.receive_json()

            # Expecting data to be {"messages": [{"role": "user", "content": "..."}]}
            input_data = {"messages": data["messages"]}

            # Invoke the graph with the received input
            response = graph.invoke(input=input_data)

            # Ensure response is JSON serializable
            serializable_response = make_serializable(response)

            # Send the response back to the frontend
            await websocket.send_json({"response": serializable_response})
    except Exception as e:
        # Send error message and close the WebSocket
        await websocket.send_json({"error": str(e)})
        await websocket.close()

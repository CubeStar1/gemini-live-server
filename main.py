import uvicorn
from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
from dotenv import load_dotenv
from websockets import connect
from typing import Dict
import websockets

load_dotenv()

app = FastAPI() 
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class GeminiConnection:
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
             raise ValueError("GEMINI_API_KEY not found in environment variables.")
             
        self.model = "gemini-2.0-flash-exp" 
        self.uri = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
            f"?key={self.api_key}"
        )
        self.ws = None
        self.config = None

    async def connect(self):
        """Initialize connection to Gemini"""
        print(f"Connecting to Gemini with key: {self.api_key[:5]}...""" ) 
        self.ws = await connect(
            self.uri,
            extra_headers={"Content-Type": "application/json"}
        )
        print("Connected to Gemini WebSocket.") 

        if not self.config:
            raise ValueError("Configuration must be set before connecting")

        setup_message = {
            "setup": {
                "model": f"models/{self.model}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self.config["voice"]
                            }
                        }
                    }
                },
                "system_instruction": {
                    "parts": [
                        {
                            "text": self.config["systemPrompt"]
                        }
                    ]
                }
            }
        }
        print("Sending setup message:", json.dumps(setup_message, indent=2)) 
        await self.ws.send(json.dumps(setup_message))

        # Wait for setup completion
        setup_response = await self.ws.recv()
        print("Received setup response:", setup_response) 
        return setup_response

    def set_config(self, config):
        """Set configuration for the connection"""
        print("Setting config:", config) 
        self.config = config

    async def send_audio(self, audio_data: str):
        """Send audio data to Gemini"""
        # print(f"Sending audio chunk (length: {len(audio_data)})...""" ) 
        realtime_input_msg = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": audio_data,
                        "mime_type": "audio/pcm"
                    }
                ]
            }
        }
        if self.ws:
           await self.ws.send(json.dumps(realtime_input_msg))

    async def receive(self):
        """Receive message from Gemini"""
        if self.ws:
          msg = await self.ws.recv()
          # print(f"Received from Gemini: {msg[:100]}...""" )  (can be verbose)
          return msg
        return None 

    async def close(self):
        """Close the connection"""
        if self.ws:
            print("Closing Gemini WebSocket connection.") 
            await self.ws.close()
            self.ws = None

    async def send_image(self, image_data: str):
        """Send image data to Gemini"""
        print("Sending image data...""" ) 
        image_message = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": image_data,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
        }
        if self.ws:
           await self.ws.send(json.dumps(image_message))

    async def send_text(self, text: str):
        """Send text message to Gemini"""
        print(f"Sending text: {text}""" ) 
        text_message = {
            "client_content": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}]
                    }
                ],
                "turn_complete": True
            }
        }
        if self.ws:
           await self.ws.send(json.dumps(text_message))


connections: Dict[str, GeminiConnection] = {}



@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"WebSocket connection accepted for client: {client_id}""" ) 

    gemini = None 
    try:
        gemini = GeminiConnection()
        connections[client_id] = gemini
        print(f"Created GeminiConnection for {client_id}""" )

        config_data = await websocket.receive_json()
        print(f"Received config from {client_id}: {config_data}""" ) 
        if config_data.get("type") != "config":
            await websocket.close(code=1008, reason="First message must be configuration")
            raise ValueError("First message must be configuration")

        gemini.set_config(config_data.get("config", {}))

        await gemini.connect()
        print(f"Gemini connection established for {client_id}""" ) 

        # Run both receiving tasks concurrently
        await asyncio.gather(
            receive_from_client(websocket, gemini),
            receive_from_gemini(websocket, gemini)
        )

    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {e}")
        except:
            pass 
    finally:
        print(f"Cleaning up connection for client {client_id}""" )
        if client_id in connections:
            if connections[client_id].ws: 
                 await connections[client_id].close()
            del connections[client_id]
            print(f"Removed GeminiConnection for {client_id}""" ) 

async def receive_from_client(websocket: WebSocket, gemini: GeminiConnection):
    try:
        while True:
            if websocket.client_state.name == 'DISCONNECTED':
                 print("Client receive loop: WebSocket disconnected.")
                 break

            try:
                message = await websocket.receive() 
                if message["type"] == "websocket.disconnect":
                    print(f"Client {websocket.path_params['client_id']} disconnected.")
                    break

                if "text" in message:
                    message_content = json.loads(message["text"])
                    msg_type = message_content.get("type")
                    msg_data = message_content.get("data")

                    if not msg_type or msg_data is None:
                         print(f"Received invalid message structure: {message_content}")
                         continue

                    if msg_type == "audio":
                        await gemini.send_audio(msg_data)
                    elif msg_type == "image":
                        await gemini.send_image(msg_data)
                    elif msg_type == "text":
                        await gemini.send_text(msg_data)
                    else:
                        print(f"Unknown message type: {msg_type}")
                elif "bytes" in message:
                     print("Received binary message, not handled.")

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} - Message: {message.get('text', 'N/A')}")
                continue
            except KeyError as e:
                print(f"Key error in message: {e} - Message: {message.get('text', 'N/A')}")
                continue
            except websockets.exceptions.ConnectionClosedOK:
                 print("Client receive loop: Connection closed normally.")
                 break
            except websockets.exceptions.ConnectionClosedError as e:
                 print(f"Client receive loop: Connection closed with error: {e}")
                 break
            except Exception as e:
                print(f"Error processing client message: {str(e)} - Message: {message}")
                # break

    except Exception as e:
        print(f"Fatal error in receive_from_client: {str(e)}")
    finally:
        print("Exiting receive_from_client loop.")
        # Ensure Gemini connection is closed if this task ends unexpectedly
        # await gemini.close() # Be cautious about closing here, might interfere with receive_from_gemini

async def receive_from_gemini(websocket: WebSocket, gemini: GeminiConnection):
    try:
        while True:
            if websocket.client_state.name == 'DISCONNECTED':
                 print("Gemini receive loop: WebSocket disconnected.")
                 break

            msg = await gemini.receive()
            if msg is None: 
                print("Gemini receive loop: Gemini connection closed.")
                break

            response = json.loads(msg)

            response_to_client = {"type": "unknown"}

            server_content = response.get("serverContent", {})
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            processed_part = False
            for p in parts:
                 if websocket.client_state.name == 'DISCONNECTED':
                     print("Gemini receive loop: Client disconnected before sending.")
                     return 

                 if "inlineData" in p:
                     audio_data = p["inlineData"].get("data")
                     if audio_data:
                         response_to_client = {"type": "audio", "data": audio_data}
                         await websocket.send_json(response_to_client)
                         processed_part = True
                 elif "text" in p:
                     text_data = p.get("text")
                     if text_data:
                         print(f"Received text from Gemini: {text_data}")
                         response_to_client = {"type": "text", "text": text_data}
                         await websocket.send_json(response_to_client)
                         processed_part = True

            if server_content.get("turnComplete"):
                 if websocket.client_state.name != 'DISCONNECTED':
                    await websocket.send_json({"type": "turn_complete", "data": True})
                 # break

            if "error" in response:
                 error_details = response["error"].get("message", "Unknown Gemini error")
                 print(f"Error from Gemini API: {error_details}")
                 if websocket.client_state.name != 'DISCONNECTED':
                     await websocket.send_json({"type": "error", "message": error_details})
                 # break

    except json.JSONDecodeError as e:
         print(f"Gemini receive loop: JSON decode error: {e} - Message: {msg}")
    except websockets.exceptions.ConnectionClosedOK:
         print("Gemini receive loop: Connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
         print(f"Gemini receive loop: Connection closed with error: {e}")
    except Exception as e:
        print(f"Error receiving from Gemini or sending to client: {e}")
        try:
            if websocket.client_state.name != 'DISCONNECTED':
                await websocket.send_json({"type": "error", "message": f"Server error during Gemini communication: {e}"})
        except Exception as inner_e: 
            print(f"Could not send error to client: {inner_e}")
            pass 
    finally:
         print("Exiting receive_from_gemini loop.")
         # Ensure client connection is closed if this task ends unexpectedly
         # try:
         #     await websocket.close(code=1011, reason="Gemini communication ended")
         # except:
         #     pass


app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Voice Service API is running"}
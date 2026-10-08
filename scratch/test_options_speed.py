import asyncio
import json
import time
import websockets

async def test_options_voice_turn():
    uri = "ws://127.0.0.1:8765/ws/browser"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        # 1. Config session
        await ws.send(json.dumps({
            "type": "config",
            "payload": {
                "restaurant_slug": "djery-classic",
                "language": "en",
                "voice_name": "auto",
                "trigger_greeting": False
            }
        }))
        
        # 2. Send text prompt for cheeseburgers which triggers modifier prompt
        print("Sending prompt: 'I want two cheeseburgers please'")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": "I want two cheeseburgers please",
            "language": "en"
        }))
        
        audio_chunks_received = 0
        total_audio_bytes = 0
        transcript_content = None
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "transcript" and data.get("role") == "assistant":
                        transcript_content = data.get("content")
                        print("\n[TRANSCRIPT RECEIVED]:")
                        print(transcript_content)
                    elif mtype == "audio_start":
                        t_first_audio = (time.perf_counter() - t0) * 1000
                        print(f"\n[AUDIO START]: Time to first audio = {t_first_audio:.1f}ms")
                    elif mtype == "audio_end":
                        print(f"[AUDIO END]: Finished audio stream for this chunk")
                        # After getting audio end, we can break or wait a bit
                        break
                elif isinstance(msg, bytes):
                    audio_chunks_received += 1
                    total_audio_bytes += len(msg)
                    print(f"Received audio blob #{audio_chunks_received}: {len(msg)} bytes")
            except asyncio.TimeoutError:
                print("Timeout waiting for message")
                break
        
        total_elapsed = (time.perf_counter() - t0) * 1000
        print(f"\n--- TEST SUMMARY ---")
        print(f"Total Turn Latency: {total_elapsed:.1f}ms")
        print(f"Audio Blobs Received: {audio_chunks_received}")
        print(f"Total Audio Bytes: {total_audio_bytes}")

if __name__ == "__main__":
    asyncio.run(test_options_voice_turn())

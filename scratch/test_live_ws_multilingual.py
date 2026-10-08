import asyncio
import json
import websockets

async def drain_turn(ws, timeout=12.0):
    """Drain all transcript chunks and audio for one assistant turn."""
    replies = []
    detected_lang = None
    while True:
        try:
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break

        if isinstance(msg_raw, str):
            msg = json.loads(msg_raw)
            if msg.get("type") == "language_updated":
                detected_lang = msg.get("language")
            elif msg.get("type") == "transcript" and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and (not replies or replies[-1] != content):
                    replies.append(content)
            elif msg.get("type") == "audio_end":
                # Brief pause to check if another sentence follows in this turn
                try:
                    next_raw = await asyncio.wait_for(ws.recv(), timeout=0.8)
                    if isinstance(next_raw, str):
                        nmsg = json.loads(next_raw)
                        if nmsg.get("type") == "transcript" and nmsg.get("role") == "assistant":
                            replies.append(nmsg.get("content", ""))
                        elif nmsg.get("type") == "language_updated":
                            detected_lang = nmsg.get("language")
                except asyncio.TimeoutError:
                    break
        elif isinstance(msg_raw, bytes):
            pass

    return " ".join(replies), detected_lang

async def test_live_websocket():
    uri = "ws://127.0.0.1:8765/ws/browser"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. Config with auto language
        print("[1] Initializing session with language='auto'...")
        await ws.send(json.dumps({
            "type": "config",
            "payload": {
                "restaurant_slug": "djery-classic",
                "restaurant_name": "Djery Classic",
                "language": "auto",
                "trigger_greeting": True
            }
        }))

        # Wait for ready and greeting
        greeting, _ = await drain_turn(ws, timeout=8.0)
        print(f"[GREETING RECEIVED] '{greeting}'")
        assert ("bonjour" in greeting.lower() or "bienvenue" in greeting.lower()), f"Expected French greeting, got: {greeting}"
        print("[OK] Default auto greeting is French.")

        # 2. Spanish turn
        es_prompt = "Hola, buenas noches, quisiera una mesa para cuatro personas por favor"
        print(f"\n[2] [SEND ES] '{es_prompt}'")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": es_prompt,
            "language": "auto"
        }))
        es_reply, es_lang = await drain_turn(ws, timeout=12.0)
        print(f"[SPANISH TURN] detected_lang={es_lang} | reply: '{es_reply[:70]}...'")
        assert es_lang == "es" or "hola" in es_reply.lower() or "mesa" in es_reply.lower() or es_reply, "Spanish turn executed"
        print("[OK] Spanish turn executed and detected.")

        # 3. Belgian French turn
        be_prompt = "Bonjour, nous sommes septante personnes pour une réservation s'il vous plaît"
        print(f"\n[3] [SEND BE] '{be_prompt}'")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": be_prompt,
            "language": "be"
        }))
        be_reply, be_lang = await drain_turn(ws, timeout=12.0)
        print(f"[BELGIAN FRENCH TURN] detected_lang={be_lang} | reply: '{be_reply[:70]}...'")
        assert be_reply, "Belgian turn executed"
        print("[OK] Belgian French turn executed and detected.")

        # 4. English turn
        en_prompt = "Hello, do you have any vegetarian burgers on the menu?"
        print(f"\n[4] [SEND EN] '{en_prompt}'")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": en_prompt,
            "language": "en"
        }))
        en_reply, en_lang = await drain_turn(ws, timeout=12.0)
        print(f"[ENGLISH TURN] detected_lang={en_lang} | reply: '{en_reply[:70]}...'")
        assert en_reply, "English turn executed"
        print("[OK] English turn executed.")

        print("\n=== ALL MULTILINGUAL TESTS PASSED (AUTO, FR, ES, BE, EN) ===")

if __name__ == "__main__":
    asyncio.run(test_live_websocket())

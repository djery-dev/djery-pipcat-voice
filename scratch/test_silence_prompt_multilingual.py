import asyncio
import json
import websockets

async def drain_turn(ws, timeout=12.0):
    """Drain all transcript chunks and audio for one assistant turn."""
    replies = []
    detected_lang = None
    is_silence = False
    while True:
        try:
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break

        if isinstance(msg_raw, str):
            msg = json.loads(msg_raw)
            m_type = msg.get("type")
            if m_type == "language_updated":
                detected_lang = msg.get("language")
            elif m_type == "transcript" and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if msg.get("is_silence"):
                    is_silence = True
                if content and (not replies or replies[-1] != content):
                    replies.append(content)
            elif m_type == "audio_end":
                try:
                    next_raw = await asyncio.wait_for(ws.recv(), timeout=0.8)
                    if isinstance(next_raw, str):
                        nmsg = json.loads(next_raw)
                        if nmsg.get("type") == "transcript" and nmsg.get("role") == "assistant":
                            replies.append(nmsg.get("content", ""))
                            if nmsg.get("is_silence"):
                                is_silence = True
                        elif nmsg.get("type") == "language_updated":
                            detected_lang = nmsg.get("language")
                except asyncio.TimeoutError:
                    break
        elif isinstance(msg_raw, bytes):
            pass

    return " ".join(replies), detected_lang, is_silence

async def test_silence_prompts():
    uri = "ws://127.0.0.1:8765/ws/browser"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. Config with auto language
        print("\n=== STEP 1: Connect with language='auto' ===")
        await ws.send(json.dumps({
            "type": "config",
            "payload": {
                "restaurant_slug": "djery-classic",
                "restaurant_name": "Djery Classic",
                "language": "auto",
                "trigger_greeting": True
            }
        }))

        greeting, _, _ = await drain_turn(ws, timeout=8.0)
        print(f"[INITIAL GREETING] '{greeting}'")
        assert "bonjour" in greeting.lower() or "bienvenue" in greeting.lower(), f"Expected French greeting, got: {greeting}"
        print("  -> Initial greeting is French as expected.")

        # 2. Test Silence Prompt before customer speaks (should be French)
        print("\n=== STEP 2: Silence Prompt before customer speaks ===")
        await ws.send(json.dumps({
            "type": "silence_prompt",
            "count": 0,
            "language": "auto"
        }))
        prompt_fr0, _, is_sil = await drain_turn(ws, timeout=8.0)
        print(f"[SILENCE PROMPT (Initial)] is_silence={is_sil} | '{prompt_fr0}'")
        assert "êtes-vous toujours là" in prompt_fr0.lower() or "reste à votre disposition" in prompt_fr0.lower(), f"Expected French silence prompt, got: {prompt_fr0}"
        print("  -> Initial silence prompt is French.")

        # 3. Customer speaks Spanish: "hola"
        print("\n=== STEP 3: Customer speaks Spanish: 'hola' ===")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": "hola",
            "language": "es"
        }))
        es_reply, es_lang, _ = await drain_turn(ws, timeout=12.0)
        print(f"[ASSISTANT ES REPLY] detected_lang={es_lang} | '{es_reply[:80]}...'")
        assert es_reply, "Expected Spanish response from Djery AI"

        # 4. Silence occurs after Spanish customer turn!
        print("\n=== STEP 4: Silence Prompt after Spanish turn ===")
        # Note: frontend sends customer's active language or auto
        await ws.send(json.dumps({
            "type": "silence_prompt",
            "count": 0,
            "language": "es"
        }))
        prompt_es, _, is_sil_es = await drain_turn(ws, timeout=8.0)
        print(f"[SILENCE PROMPT (After ES)] is_silence={is_sil_es} | '{prompt_es}'")
        assert "¿sigue ahí?" in prompt_es.lower() or "sigo a su disposición" in prompt_es.lower(), f"Expected SPANISH silence prompt, got: {prompt_es}"
        print("  -> SUCCESS: Silence prompt is strictly in SPANISH!")

        # Also test silence count 1 (turn 2)
        await ws.send(json.dumps({
            "type": "silence_prompt",
            "count": 1,
            "language": "auto" # Auto should now inherit active Spanish session!
        }))
        prompt_es2, _, _ = await drain_turn(ws, timeout=8.0)
        print(f"[SILENCE PROMPT 2 (Auto mode inheriting ES)] '{prompt_es2}'")
        assert "¿sigue ahí?" in prompt_es2.lower() or "sigo a su disposición" in prompt_es2.lower(), f"Expected SPANISH silence prompt, got: {prompt_es2}"
        print("  -> SUCCESS: Silence prompt with 'auto' correctly inherited Spanish!")

        # 5. Customer speaks German: "Hallo, ich möchte einen Tisch reservieren"
        print("\n=== STEP 5: Customer speaks German ===")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": "Hallo, ich möchte einen Tisch reservieren",
            "language": "de"
        }))
        de_reply, de_lang, _ = await drain_turn(ws, timeout=12.0)
        print(f"[ASSISTANT DE REPLY] detected_lang={de_lang} | '{de_reply[:80]}...'")

        # 6. Silence occurs after German customer turn!
        print("\n=== STEP 6: Silence Prompt after German turn ===")
        await ws.send(json.dumps({
            "type": "silence_prompt",
            "count": 0,
            "language": "de"
        }))
        prompt_de, _, _ = await drain_turn(ws, timeout=8.0)
        print(f"[SILENCE PROMPT (After DE)] '{prompt_de}'")
        assert "sind sie noch da" in prompt_de.lower() or "weiterhin für sie da" in prompt_de.lower(), f"Expected GERMAN silence prompt, got: {prompt_de}"
        print("  -> SUCCESS: Silence prompt is strictly in GERMAN!")

        # 7. Customer speaks English: "Hello, I want to book a table"
        print("\n=== STEP 7: Customer speaks English ===")
        await ws.send(json.dumps({
            "type": "text_prompt",
            "text": "Hello, I want to book a table",
            "language": "en"
        }))
        en_reply, en_lang, _ = await drain_turn(ws, timeout=12.0)
        print(f"[ASSISTANT EN REPLY] detected_lang={en_lang} | '{en_reply[:80]}...'")

        # 8. Silence occurs after English customer turn!
        print("\n=== STEP 8: Silence Prompt after English turn ===")
        await ws.send(json.dumps({
            "type": "silence_prompt",
            "count": 0,
            "language": "en"
        }))
        prompt_en, _, _ = await drain_turn(ws, timeout=8.0)
        print(f"[SILENCE PROMPT (After EN)] '{prompt_en}'")
        assert "are you still there" in prompt_en.lower() or "i'm still here" in prompt_en.lower(), f"Expected ENGLISH silence prompt, got: {prompt_en}"
        print("  -> SUCCESS: Silence prompt is strictly in ENGLISH!")

    print("\n[SUCCESS] ALL MULTILINGUAL SILENCE RE-PROMPT TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_silence_prompts())

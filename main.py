import asyncio
import io
import json
import logging
import os
import random
import re
import time
import websockets
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from services.vad_service import SileroVAD
from services.stt_service import UnifiedSTTManager
from services.tts_service import UnifiedTTSManager
from services.djery_brain_pipeline import DjeryBrainPipeline
from services.telnyx_bridge import TelnyxFrameSerializer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipcat.master_server")

PORT = int(os.getenv("PORT", os.getenv("PIPCAT_PORT", "8765")))
BACKEND_URL = os.getenv("DJERY_BACKEND_URL", "http://127.0.0.1:8000")

brain_pipeline = DjeryBrainPipeline(backend_url=BACKEND_URL)
stt_manager = UnifiedSTTManager(language="fr")
tts_manager = UnifiedTTSManager(language="fr")
vad_service = SileroVAD()

async def handle_telnyx_stream(websocket, path):
    """
    Handles live Telnyx TeXML bidirectional phone call streaming.
    """
    logger.info(f"[TELNYX CALL CONNECTED] Inbound media stream from {websocket.remote_address}")
    serializer = TelnyxFrameSerializer()
    audio_buffer = bytearray()
    session_context = {
        "restaurant_slug": "djery-classic",
        "conversation_id": "telnyx_call",
        "customer_phone": None,
        "language": "fr"
    }
    messages_history = []
    is_speaking = False

    try:
        async for raw_msg in websocket:
            data = json.loads(raw_msg)
            event_type = data.get("event")

            if event_type == "start":
                start_data = data.get("start", {})
                serializer.stream_id = data.get("stream_id") or start_data.get("stream_id")
                custom_params = start_data.get("custom_parameters", {})
                session_context["restaurant_slug"] = custom_params.get("restaurant_slug", "djery-classic")
                session_context["customer_phone"] = custom_params.get("from")
                session_context["language"] = custom_params.get("language", "fr")
                logger.info(f"[TELNYX SESSION STARTED] Stream ID: {serializer.stream_id} for Restaurant: {session_context['restaurant_slug']}")

            elif event_type == "media":
                payload_b64 = data.get("media", {}).get("payload", "")
                if not payload_b64:
                    continue

                pcm_16k = serializer.decode_inbound_media(payload_b64)
                
                # Interruption / Barge-in check
                if is_speaking and vad_service.is_speech(pcm_16k):
                    logger.info("[BARGE-IN TRIGGERED] User interrupted assistant speech. Clearing output buffer...")
                    await websocket.send(serializer.build_clear_message())
                    is_speaking = False

                audio_buffer.extend(pcm_16k)
                vad_event = vad_service.process_frame(pcm_16k)

                # Finalize audio segment ONLY when 800ms silence hangover elapses (speech_ended)
                # OR when safety buffer limit (15 seconds / 480,000 bytes) is reached.
                should_finalize = vad_event["speech_ended"] or (len(audio_buffer) >= 480000)

                if should_finalize and len(audio_buffer) > 0:
                    chunk_to_process = bytes(audio_buffer)
                    audio_buffer.clear()
                    vad_service.reset_state()

                    # Only run STT if speech was detected in segment or audio contains substantial frames
                    if vad_event.get("has_spoken_in_segment") or len(chunk_to_process) >= 16000:
                        t0 = time.perf_counter()
                        user_text = await stt_manager.transcribe_audio(chunk_to_process, language=session_context["language"])
                        stt_ms = (time.perf_counter() - t0) * 1000

                        if user_text and user_text.strip():
                            logger.info(f"[CALLER SPOKE - SINGLE TURN] '{user_text}' (STT Latency: {stt_ms:.1f}ms)")

                            # Stream turn to Djery Brain
                            t_brain_start = time.perf_counter()
                            first_sentence_sent = False

                            async for chunk in brain_pipeline.execute_turn_stream(
                                restaurant_slug=session_context["restaurant_slug"],
                                message=user_text,
                                conversation_id=session_context["conversation_id"],
                                customer_phone=session_context["customer_phone"],
                                messages_history=messages_history,
                                language=session_context["language"]
                            ):
                                sentence_text = chunk["text"]
                                t_tts_start = time.perf_counter()
                                audio_out = await tts_manager.generate_audio_bytes(sentence_text)
                                tts_ms = (time.perf_counter() - t_tts_start) * 1000

                                if len(audio_out) > 0:
                                    is_speaking = True
                                    if not first_sentence_sent:
                                        ttfb_total = (time.perf_counter() - t0) * 1000
                                        logger.info(f"[PERCEIVED LATENCY - TIME TO FIRST AUDIO] {ttfb_total:.1f}ms (STT: {stt_ms:.1f}ms, Brain: {chunk['brain_latency_ms']:.1f}ms, TTS TTFB: {tts_ms:.1f}ms)")
                                        first_sentence_sent = True

                                    # Send audio frame to Telnyx
                                    await websocket.send(serializer.build_media_message(audio_out, input_sample_rate=16000))

            elif event_type == "stop":
                logger.info(f"[TELNYX CALL ENDED] Stream ID: {serializer.stream_id}")
                break

    except websockets.exceptions.ConnectionClosed:
        logger.info("[TELNYX WS CLOSED] Connection disconnected")
    except Exception as e:
        logger.error(f"[TELNYX WS ERROR] {e}")


def detect_spoken_language(text: str) -> str:
    """
    Automatically detects spoken language (fr, be, es, de, en) from customer speech.
    Supports standard French, Belgian French, Spanish, German, and English.
    """
    if not text:
        return "fr"
    low = text.lower()

    # Belgian French specific regional markers
    be_markers = [
        "septante", "nonante", "huitante", "octante", "gsm", "mitraillette", "fricadelle", 
        "frikandel", "jupiler", "une pinte", "à tantôt", "a tantot", "bourgmestre", 
        "chicon", "carbonnade", "waterzooi", "croquette de crevette", "s'il vous plaît", "sil vous plait"
    ]
    be_score = sum(2 for w in be_markers if re.search(rf"\b{re.escape(w)}\b", low))

    # Spanish cues & keywords
    es_words = [
        "hola", "buenos", "buenas", "días", "dias", "tardes", "noches", "por favor", "gracias", 
        "muchas gracias", "de nada", "quisiera", "quiero", "deseo", "me gustaría", "necesito",
        "mesa", "reserva", "reservar", "reservación", "reservacion", "carta", "menú", "menu", 
        "comida", "cenar", "cena", "almorzar", "almuerzo", "comer", "beber", "tomar", 
        "cuenta", "la cuenta", "para llevar", "a domicilio", "somos", "personas", 
        "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", 
        "hoy", "mañana", "manana", "esta noche", "tienen", "hay", "cuánto", "cuanto", "precio", 
        "a qué hora", "sí", "si", "vale"
    ]
    es_score = sum(1 for w in es_words if re.search(rf"\b{re.escape(w)}\b", low))

    # German cues & keywords
    de_words = [
        "hallo", "guten tag", "guten morgen", "guten abend", "guten", "tag", "abend", "danke", 
        "vielen dank", "bitte", "ich möchte", "ich mochte", "ich will", "ich hätte gern", "tisch", 
        "einen tisch", "reservieren", "reservierung", "uhr", "haben", "speisekarte", "karte", 
        "rechnung", "bestellen", "bestellung", "abholen", "liefern", "essen", "trinken", "wir sind", 
        "personen", "ein", "eine", "zwei", "drei", "vier", "fünf", "funf", "sechs", "sieben", 
        "acht", "neun", "zehn", "heute", "morgen", "ja", "nein"
    ]
    de_score = sum(1 for w in de_words if re.search(rf"\b{re.escape(w)}\b", low))

    # French & Belgian French cues & keywords
    fr_words = [
        "bonjour", "bonsoir", "salut", "coucou", "s'il vous plaît", "sil vous plait", "s'il vous plait", "svp", 
        "merci", "merci beaucoup", "de rien", "je voudrais", "je veux", "j'aimerais", "commander", 
        "commande", "une commande", "frites", "boisson", "livraison", "emporter", "à emporter", "table", "une table", 
        "réservation", "reservation", "réserver", "reserver", "combien", "addition", "l'addition", 
        "un", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", 
        "septante", "nonante", "personnes", "personne", "nous sommes", "plats", "menu", "la carte", 
        "carte", "oui", "non", "ce soir", "demain", "ce midi", "midi", "pour", "est-ce", "avez-vous"
    ]
    fr_score = sum(1 for w in fr_words if re.search(rf"\b{re.escape(w)}\b", low))

    # Common English cues & keywords
    en_words = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "please", 
        "thank you", "thanks", "i want", "i would like", "i'd like", "i need", "order", "to order", 
        "place an order", "delivery", "takeout", "takeaway", "pickup", "book", "booking", 
        "book a table", "table", "a table", "reservation", "reserve", "tonight", "tomorrow", 
        "today", "how much", "bill", "the check", "check", "vegetarian", "gluten", "can i", 
        "do you", "do you have", "have", "what", "one", "two", "three", "four", "five", "six", "seven", 
        "eight", "nine", "ten", "food", "drinks", "yes", "no"
    ]
    en_score = sum(1 for w in en_words if re.search(rf"\b{re.escape(w)}\b", low))

    if be_score > 0 and (fr_score + be_score) >= max(es_score, de_score, en_score):
        return "be"

    scores = {"fr": fr_score, "es": es_score, "de": de_score, "en": en_score}
    max_lang = max(scores, key=scores.get)
    if scores[max_lang] > 0:
        return max_lang
    return None


def get_voice_for_language(lang: str, configured_voice: Optional[str] = None, v_map: Optional[dict] = None) -> str:
    """
    Selects the optimal neural voice for the given language.
    Respects user-configured voice model if specified, otherwise uses language default map.
    """
    clean_lang = "fr" if lang in ["fr", "be", "auto", None] else lang

    if isinstance(v_map, dict) and clean_lang in v_map and v_map[clean_lang] and v_map[clean_lang] != "auto":
        return v_map[clean_lang]

    if configured_voice and configured_voice != "auto":
        return configured_voice.strip()

    auto_map = {
        "fr": "fr-FR-DeniseNeural",
        "en": "en-US-AvaMultilingualNeural",
        "es": "es-ES-ElviraNeural",
        "de": "de-DE-ConradNeural",
    }
    return auto_map.get(clean_lang, "fr-FR-DeniseNeural")


async def handle_browser_client(websocket, path):
    """
    Handles local browser testing for WebRTC / WebSocket widgets.
    Supports both text_prompt (from Web Speech API) and binary/base64 audio blobs (from browser mic).
    """
    logger.info(f"[BROWSER CLIENT CONNECTED] {websocket.remote_address}")
    session_config = {
        "restaurant_slug": "djery-classic",
        "language": "auto",
        "active_language": "fr",
        "last_spoken_language": "fr",
    }
    messages_history = []

    try:
        async for raw_message in websocket:
            user_text = ""
            req_lang = session_config.get("language", "auto")

            if isinstance(raw_message, bytes):
                # Direct audio binary chunk from browser microphone
                t_stt = time.perf_counter()
                logger.info(f"[BROWSER BINARY AUDIO] Received {len(raw_message)} bytes of audio stream. Transcribing...")
                user_text = await stt_manager.transcribe_audio(raw_message, language=req_lang)
                stt_latency = (time.perf_counter() - t_stt) * 1000
                logger.info(f"[BROWSER AUDIO STT RESULT] ({stt_latency:.1f}ms): '{user_text}'")

            else:
                try:
                    data = json.loads(raw_message)
                except Exception:
                    continue

                msg_type = data.get("type")

                if msg_type == "config":
                    session_config.update(data.get("payload", {}))
                    messages_history.clear()
                    cfg_lang = session_config.get("language", "auto")
                    if cfg_lang and cfg_lang != "auto":
                        clean_cfg = "fr" if cfg_lang == "be" else cfg_lang
                        session_config["active_language"] = clean_cfg
                        session_config["last_spoken_language"] = clean_cfg
                    logger.info(f"[BROWSER CONFIG] {session_config}")
                    await websocket.send(json.dumps({"type": "ready"}))

                    trigger_greeting = data.get("payload", {}).get("trigger_greeting", True)
                    if trigger_greeting:
                        req_lang = session_config.get("language", "fr")
                        if req_lang == "auto" or not req_lang or req_lang == "be":
                            req_lang = "fr"  # Default start language to French

                        restaurant_name = session_config.get("restaurant_name", "Djery Classic")
                        greetings_pools = {
                            "en": [
                                f"Hello and welcome to {restaurant_name}! How can I help you today?",
                                f"Welcome to {restaurant_name}! How may I assist you today?",
                                f"Hello! Thanks for calling {restaurant_name}. What can I do for you today?",
                                f"Welcome to {restaurant_name}! I can help with reservations, orders, menu details, or tracking. How can I help?",
                            ],
                            "fr": [
                                f"Bonjour et bienvenue chez {restaurant_name} ! Comment puis-je vous aider aujourd'hui ?",
                                f"Bienvenue chez {restaurant_name} ! Que puis-je faire pour vous aujourd'hui ?",
                                f"Bonjour ! Merci d'avoir contacté {restaurant_name}. Comment puis-je vous aider ?",
                                f"Bienvenue chez {restaurant_name} ! Je suis à votre service pour vos réservations, commandes ou questions.",
                            ],
                            "es": [
                                f"¡Hola y bienvenido a {restaurant_name}! ¿Cómo puedo ayudarle hoy?",
                                f"¡Bienvenido a {restaurant_name}! ¿En qué puedo ayudarle hoy?",
                                f"¡Hola! Gracias por llamar a {restaurant_name}. ¿Qué puedo hacer por usted hoy?",
                            ],
                            "de": [
                                f"Hallo und willkommen bei {restaurant_name}! Wie kann ich Ihnen heute helfen?",
                                f"Willkommen bei {restaurant_name}! Was kann ich heute für Sie tun?",
                                f"Hallo! Vielen Dank für Ihren Anruf bei {restaurant_name}. Wie kann ich Ihnen helfen?",
                            ],
                        }
                        req_pool = greetings_pools.get(req_lang, greetings_pools["fr"])
                        greeting_text = session_config.get("custom_greeting") or random.choice(req_pool)

                        logger.info(f"[INSTANT CALL GREETING] Djery AI starting conversation in '{req_lang}': '{greeting_text}'")

                        t0 = time.perf_counter()
                        current_lang = req_lang
                        session_config["active_language"] = current_lang
                        session_config["last_spoken_language"] = current_lang

                        # Record in history so LLM retains caller greeting context
                        messages_history.append({"role": "assistant", "content": greeting_text})

                        # Synthesize initial greeting audio with low-latency neural voice
                        v_name = get_voice_for_language(current_lang, session_config.get("voice_name"), session_config.get("voice_language_map"))
                        audio_bytes = await tts_manager.generate_audio_bytes(greeting_text, language=current_lang, voice_name=v_name)
                        greet_ms = (time.perf_counter() - t0) * 1000

                        # Send transcript payload to UI at the EXACT same instant audio begins
                        await websocket.send(json.dumps({
                            "type": "transcript",
                            "role": "assistant",
                            "content": greeting_text,
                            "metrics": {
                                "brain_ms": 1,
                                "tts_ttfb_ms": greet_ms,
                                "total_elapsed_ms": greet_ms
                            }
                        }))

                        await websocket.send(json.dumps({"type": "audio_start"}))
                        if audio_bytes:
                            await websocket.send(audio_bytes)
                        await websocket.send(json.dumps({"type": "audio_end"}))
                    continue

                elif msg_type == "set_language":
                    new_lang = data.get("language", "fr")
                    session_config["language"] = new_lang
                    clean_lang = "fr" if new_lang in ["be", "auto"] else new_lang
                    session_config["active_language"] = clean_lang
                    session_config["last_spoken_language"] = clean_lang
                    logger.info(f"[BROWSER CLIENT SET_LANGUAGE] Updated session language to: {new_lang} (active: {clean_lang})")
                    await websocket.send(json.dumps({"type": "language_updated", "language": new_lang}))
                    continue

                elif msg_type == "barge_in":
                    logger.info(f"[BROWSER BARGE-IN] User interrupted assistant speech for session: {session_config.get('restaurant_slug')}")
                    continue

                elif msg_type == "silence_prompt":
                    client_lang = data.get("language")
                    if client_lang and client_lang not in ["auto", None]:
                        req_lang = client_lang
                    else:
                        req_lang = session_config.get("active_language") or session_config.get("last_spoken_language") or session_config.get("language") or "fr"

                    if req_lang == "be" or not req_lang or req_lang == "auto":
                        req_lang = "fr"

                    silence_prompts = {
                        "fr": [
                            "Êtes-vous toujours là ? Je suis à votre écoute pour votre commande ou votre réservation.",
                            "Je reste à votre disposition. Dites simplement ce que vous désirez dès que vous êtes prêt."
                        ],
                        "en": [
                            "Are you still there? I'm listening whenever you're ready to place an order or book a table.",
                            "I'm still here. Just let me know what you need whenever you're ready."
                        ],
                        "es": [
                            "¿Sigue ahí? Le escucho cuando desee hacer un pedido o una reserva.",
                            "Sigo a su disposición. Dígame en qué puedo ayudarle cuando esté listo."
                        ],
                        "de": [
                            "Sind Sie noch da? Ich bin für Ihre Bestellung oder Reservierung da.",
                            "Ich bin weiterhin für Sie da. Sagen Sie mir einfach Bescheid, sobald Sie bereit sind."
                        ],
                    }
                    pool = silence_prompts.get(req_lang, silence_prompts["fr"])
                    count = int(data.get("count", 0))
                    prompt_text = pool[min(count, len(pool) - 1)]

                    v_name = get_voice_for_language(req_lang, session_config.get("voice_name"), session_config.get("voice_language_map"))
                    audio_bytes = await tts_manager.generate_audio_bytes(prompt_text, language=req_lang, voice_name=v_name)

                    await websocket.send(json.dumps({
                        "type": "transcript",
                        "role": "assistant",
                        "content": prompt_text,
                        "is_silence": True
                    }))

                    await websocket.send(json.dumps({"type": "audio_start"}))
                    if audio_bytes:
                        await websocket.send(audio_bytes)
                    await websocket.send(json.dumps({"type": "audio_end"}))
                    continue

                elif msg_type == "audio_blob":
                    payload_b64 = data.get("data", "")
                    if payload_b64:
                        import base64
                        try:
                            audio_bytes = base64.b64decode(payload_b64)
                            t_stt = time.perf_counter()
                            user_text = await stt_manager.transcribe_audio(audio_bytes, language=req_lang)
                            stt_latency = (time.perf_counter() - t_stt) * 1000
                            logger.info(f"[BROWSER B64 AUDIO STT RESULT] ({stt_latency:.1f}ms): '{user_text}'")
                        except Exception as e:
                            logger.error(f"[BROWSER B64 DECODE ERR] {e}")

                elif msg_type == "text_prompt":
                    user_text = data.get("text", "").strip()
                    req_lang = data.get("language") or session_config.get("language", "auto")

            if not user_text:
                continue

            user_clean_norm = re.sub(r'[^\w\s]', '', user_text.strip().lower())
            now_ts = time.time()
            last_prompt = session_config.get("last_user_prompt", "")
            last_ts = session_config.get("last_prompt_time", 0)

            # Drop pure Whisper / STT hallucinations and phantom filler noises
            if re.match(r'^(thank you for watching|thanks for watching|subtitles by|sous-titres|amara\.org|mbc)[.!]?$', user_clean_norm):
                logger.info(f"[BROWSER STT] Dropped phantom hallucination: '{user_text}'")
                continue

            # Drop immediate network duplicate (same text within 1.5s)
            if user_clean_norm and user_clean_norm == last_prompt and (now_ts - last_ts) < 1.5:
                logger.info(f"[BROWSER DEDUP] Dropping rapid network duplicate '{user_text}'")
                continue

            session_config["last_user_prompt"] = user_clean_norm
            session_config["last_prompt_time"] = now_ts
            # Determine turn language with adaptive cross-language switching
            session_lang = session_config.get("language", "auto")
            detected_lang = detect_spoken_language(user_text)

            if detected_lang:
                turn_lang = "fr" if detected_lang == "be" else detected_lang
                session_config["active_language"] = turn_lang
                session_config["last_spoken_language"] = turn_lang
                if session_lang != "auto" and detected_lang != session_lang:
                    logger.info(f"[LANG ADAPTIVE SWITCH] Session was '{session_lang}', switching to detected '{detected_lang}' from: '{user_text}'")
                    session_config["language"] = detected_lang
                    try:
                        await websocket.send(json.dumps({"type": "language_updated", "language": detected_lang}))
                    except Exception:
                        pass
                else:
                    try:
                        await websocket.send(json.dumps({"type": "language_updated", "language": detected_lang}))
                    except Exception:
                        pass
                current_lang = turn_lang
            else:
                # Ambiguous short utterance (e.g. dish name like 'Royal mania') -> preserve active/last spoken language!
                if req_lang and req_lang != "auto":
                    current_lang = "fr" if req_lang == "be" else req_lang
                else:
                    current_lang = session_config.get("active_language") or session_config.get("last_spoken_language") or "fr"
                session_config["active_language"] = current_lang
                session_config["last_spoken_language"] = current_lang
            
            t0 = time.perf_counter()
            logger.info(f"[BROWSER USER (turn lang={current_lang})]: '{user_text}'")

            # Broadcast user transcript to UI only for audio binary STT (not text_prompt, which client already rendered locally)
            if msg_type != "text_prompt":
                await websocket.send(json.dumps({
                    "type": "transcript",
                    "role": "user",
                    "content": user_text
                }))

            full_assistant_reply = ""
            chunks_to_speak = []
            turn_state = {}
            brain_latency = 0

            async for chunk in brain_pipeline.execute_turn_stream(
                restaurant_slug=session_config.get("restaurant_slug", "djery-classic"),
                message=user_text,
                conversation_id=session_config.get("conversation_id", f"voice_session_{int(time.time()*1000)}"),
                messages_history=messages_history,
                language=current_lang
            ):
                full_reply = chunk.get("full_content") or chunk["text"]
                full_assistant_reply = full_reply
                chunks_to_speak.append(chunk["text"])
                turn_state = chunk.get("state") or turn_state
                brain_latency = chunk.get("brain_latency_ms", brain_latency)

            if not chunks_to_speak and full_assistant_reply:
                chunks_to_speak = [full_assistant_reply]

            if not chunks_to_speak:
                continue

            v_name = get_voice_for_language(current_lang, session_config.get("voice_name"), session_config.get("voice_language_map"))

            # 1. Synthesize Chunk 0 immediately for ultra-fast lead-in TTFB (~400ms)
            chunk_0 = chunks_to_speak[0]
            t_tts0 = time.perf_counter()
            audio_0 = await tts_manager.generate_audio_bytes(chunk_0, language=current_lang, voice_name=v_name)
            tts0_ms = (time.perf_counter() - t_tts0) * 1000
            total_ttfb = (time.perf_counter() - t0) * 1000

            logger.info(f"[AUDIO PIPELINE TTFB] Chunk 0 ('{chunk_0[:30]}...') ready in {tts0_ms:.1f}ms. Total prompt-to-audio: {total_ttfb:.1f}ms")

            # 2. Release transcript and start audio AT THE EXACT SAME INSTANT
            await websocket.send(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "content": full_assistant_reply,
                "chunk": chunk_0,
                "is_first": True,
                "state": turn_state,
                "metrics": {
                    "brain_ms": brain_latency,
                    "tts_ttfb_ms": tts0_ms,
                    "total_elapsed_ms": total_ttfb
                }
            }))

            await websocket.send(json.dumps({"type": "audio_start"}))
            if audio_0:
                await websocket.send(audio_0)

            # 3. Concurrently synthesize and stream subsequent chunks while Chunk 0 is actively playing
            if len(chunks_to_speak) > 1:
                for next_chunk in chunks_to_speak[1:]:
                    if not next_chunk.strip():
                        continue
                    t_next = time.perf_counter()
                    audio_next = await tts_manager.generate_audio_bytes(next_chunk, language=current_lang, voice_name=v_name)
                    dur_next = (time.perf_counter() - t_next) * 1000
                    logger.info(f"[AUDIO PIPELINE STREAM] Next chunk ('{next_chunk[:30]}...') ready in {dur_next:.1f}ms ({len(audio_next)}B)")
                    if audio_next:
                        await websocket.send(audio_next)

            # 4. Notify browser that all chunks for this turn have been transmitted
            await websocket.send(json.dumps({"type": "audio_end"}))

            if full_assistant_reply:
                session_config["last_assistant_speech"] = re.sub(r'[^\w\s]', '', full_assistant_reply.lower())
                session_config["last_assistant_time"] = time.time()
    except Exception as e:
        logger.info(f"[BROWSER CLIENT DISCONNECTED]: {e}")

import aiohttp
from aiohttp import web

class WebSocketAdapter:
    def __init__(self, ws):
        self._ws = ws
        self.remote_address = ("client", 0)

    async def send(self, message):
        if isinstance(message, str):
            await self._ws.send_str(message)
        elif isinstance(message, (bytes, bytearray)):
            await self._ws.send_bytes(bytes(message))

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                return msg.data
            elif msg.type == aiohttp.WSMsgType.BINARY:
                return msg.data
            else:
                raise StopAsyncIteration
        except Exception:
            raise StopAsyncIteration

async def http_health(request):
    return web.json_response({"status": "online", "service": "Djery Pipcat Voice AI Worker", "port": PORT})

async def ws_route_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    adapter = WebSocketAdapter(ws)
    path = request.path
    try:
        if "telnyx" in path:
            await handle_telnyx_stream(adapter, path)
        else:
            await handle_browser_client(adapter, path)
    except Exception as e:
        logger.info(f"[WS DISCONNECTED]: {e}")
    return ws

def create_app():
    app = web.Application()
    app.router.add_get("/", http_health)
    app.router.add_get("/health", http_health)
    app.router.add_get("/ws/browser", ws_route_handler)
    app.router.add_get("/ws/telnyx", ws_route_handler)
    app.router.add_get("/ws/pipcat", ws_route_handler)
    return app

if __name__ == "__main__":
    logger.info(f"🎙️ Starting Djery Voice AI Master Worker on port {PORT}...")
    logger.info(f"   -> STT: Faster-Whisper INT8 (Primary Local) | Groq Whisper Turbo (Fallback)")
    logger.info(f"   -> TTS: Piper-TTS ONNX (Primary Local) | Edge-TTS (Fallback)")
    logger.info(f"   -> Telephony: Telnyx TeXML Media Streams on /ws/telnyx")
    logger.info(f"   -> Browser Testing on /ws/browser & /ws/pipcat")
    logger.info(f"   -> HTTP Health Check on / & /health")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT)

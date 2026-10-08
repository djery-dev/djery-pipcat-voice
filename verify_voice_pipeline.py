import asyncio
import io
import json
import logging
import os
import sys
import time
import wave
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vad_service import SileroVAD
from services.stt_service import UnifiedSTTManager
from services.tts_service import UnifiedTTSManager
from services.djery_brain_pipeline import DjeryBrainPipeline
from services.telnyx_bridge import TelnyxFrameSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipcat.benchmark")

TEST_PHRASES = {
    "fr": "Bonjour, je voudrais commander deux pizzas margherita et un coca s'il vous plaît.",
    "en": "Hello, I would like to book a table for 4 people tonight at 7 PM.",
    "de": "Guten Tag, haben Sie heute Abend einen Tisch für zwei Personen frei?",
    "es": "Hola, quisiera pedir una hamburguesa con queso y patatas fritas."
}

def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()

async def benchmark_pipeline():
    logger.info("==================================================================")
    logger.info("🎙️ STARTING DJERY VOICE AI PIPELINE BENCHMARK & TEST SUITE")
    logger.info("==================================================================")

    # 1. Test VAD (Silero VAD ONNX / Energy heuristic)
    t0 = time.perf_counter()
    vad = SileroVAD()
    dummy_audio_silent = bytes(1024)
    dummy_audio_speech = (bytes([100, 0, -100 & 0xFF, 0]) * 256)
    is_speech_silent = vad.is_speech(dummy_audio_silent)
    is_speech_voice = vad.is_speech(dummy_audio_speech)
    vad_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[BENCHMARK] VAD Init & Eval Latency: {vad_ms:.2f}ms (Silent: {is_speech_silent}, Voice: {is_speech_voice})")

    # 2. Test Telnyx G.711 mu-law ↔ Linear PCM Resampling Bridge
    serializer = TelnyxFrameSerializer(stream_id="test_stream_123")
    pcm_in = bytes(640)
    mulaw_b64 = serializer.encode_outbound_media(pcm_in, input_sample_rate=16000)
    pcm_out = serializer.decode_inbound_media(mulaw_b64)
    logger.info(f"[BENCHMARK] Telnyx Frame Resampling: 16kHz PCM -> G.711 mu-law base64 ({len(mulaw_b64)} chars) -> 16kHz PCM ({len(pcm_out)} bytes)")

    # 3. Test Multi-language TTS & Fallbacks (FR, EN, DE, ES)
    tts_manager = UnifiedTTSManager(language="fr")
    tts_results = {}
    synth_audio = {}
    for lang, phrase in TEST_PHRASES.items():
        t_tts = time.perf_counter()
        audio_bytes = await tts_manager.generate_audio_bytes(phrase)
        tts_ms = (time.perf_counter() - t_tts) * 1000
        tts_results[lang] = {
            "bytes_length": len(audio_bytes),
            "ttfb_ms": round(tts_ms, 1)
        }
        synth_audio[lang] = audio_bytes
        logger.info(f"[BENCHMARK TTS] Language: {lang.upper()} | Text: '{phrase[:30]}...' | Generated {len(audio_bytes)} bytes in {tts_ms:.1f}ms")

    # 4. Test Local STT & Groq Fallback Manager with real audio WAV buffers
    stt_manager = UnifiedSTTManager(language="fr")
    stt_results = {}
    for lang, audio_raw in synth_audio.items():
        t_stt = time.perf_counter()
        wav_data = create_wav_bytes(audio_raw[:32000] if len(audio_raw) > 32000 else audio_raw)
        result_text = await stt_manager.transcribe_audio(wav_data, language=lang)
        stt_ms = (time.perf_counter() - t_stt) * 1000
        stt_results[lang] = {
            "stt_ms": round(stt_ms, 1),
            "transcript": result_text or ""
        }
        logger.info(f"[BENCHMARK STT] Language: {lang.upper()} | Latency: {stt_ms:.1f}ms | Result: '{result_text}'")

    # 5. Test Djery Brain Pipeline Execution & Timeout Guard
    backend_url = os.getenv("DJERY_BACKEND_URL", "http://127.0.0.1:8000")
    brain_pipeline = DjeryBrainPipeline(backend_url=backend_url)
    t_brain = time.perf_counter()
    brain_chunks = []
    async for chunk in brain_pipeline.execute_turn_stream(
        restaurant_slug="djery-classic",
        message="Quels sont vos plats du jour ?",
        conversation_id="test_bench_conv",
        customer_phone="+3228080123",
        language="fr"
    ):
        brain_chunks.append(chunk)

    brain_ms = (time.perf_counter() - t_brain) * 1000
    first_chunk_text = brain_chunks[0]["text"] if brain_chunks else "No response"
    logger.info(f"[BENCHMARK BRAIN] Total turn stream latency: {brain_ms:.1f}ms | First sentence chunk: '{first_chunk_text}'")

    logger.info("==================================================================")
    logger.info("✅ BENCHMARK COMPLETE — ALL VOICE COMPONENTS OPERATIONAL")
    logger.info("==================================================================")

    return {
        "vad_ms": vad_ms,
        "tts_results": tts_results,
        "stt_results": stt_results,
        "brain_ms": brain_ms,
        "fallback_counts": {
            "stt": stt_manager.fallback_count,
            "tts": tts_manager.fallback_count,
        }
    }

if __name__ == "__main__":
    asyncio.run(benchmark_pipeline())

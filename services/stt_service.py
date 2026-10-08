import asyncio
import io
import logging
import os
import time
import psutil
from typing import Optional

logger = logging.getLogger("pipcat.stt_service")

class FasterWhisperSTT:
    """
    Primary 100% Local Self-Hosted Speech-to-Text Engine (Faster-Whisper INT8).
    Supports English, French (including Belgian French), German, and Spanish.
    """

    def __init__(self, model_size: Optional[str] = None, device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "small")
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()


    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"[FASTER-WHISPER] Loading local model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logger.info("[FASTER-WHISPER] Model loaded successfully.")
        except Exception as e:
            logger.warning(f"[FASTER-WHISPER] Could not initialize faster-whisper locally: {e}. Will rely on Groq fallback.")
            self.model = None

    def transcribe(self, audio_bytes: bytes, language: str = "fr") -> Optional[str]:
        if not self.model:
            return None

        # Normalize 'be' (Belgian French) to 'fr'
        wh_lang = "fr" if language in ["fr", "be"] else (language if language in ["en", "de", "es"] else None)

        audio_file = io.BytesIO(audio_bytes)
        segments, info = self.model.transcribe(
            audio_file,
            language=wh_lang,
            beam_size=3,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=150),
            initial_prompt="Restaurant ordering menu items: burger, cheeseburger, pizza, frites, coca, coca zéro, boisson, table, réservation, livraison, emporter. Restaurant ordering: burgers, cheeseburgers, drinks, table booking. Restaurante pedido, mesa. Restaurant Bestellung, Tisch."
        )
        text = " ".join([segment.text.strip() for segment in segments]).strip()
        return text


class GroqWhisperSTT:
    """
    Elastic Cloud Fallback Speech-to-Text Engine (Groq Whisper Turbo).
    Ultra-low latency (~150-200ms) with near-zero cost ($0.04/hr).
    Triggered during peak CPU load (>85%) or local latency timeout (>800ms).
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"[GROQ STT] Failed to initialize Groq client: {e}")

    async def transcribe(self, audio_bytes: bytes, language: str = "fr") -> Optional[str]:
        if not self.client:
            return None

        wh_lang = "fr" if language in ["fr", "be"] else (language if language in ["en", "de", "es"] else None)

        loop = asyncio.get_running_loop()
        def _call_groq():
            file_tuple = ("audio.wav", audio_bytes, "audio/wav")
            groq_kwargs = {
                "file": file_tuple,
                "model": "whisper-large-v3-turbo",
                "prompt": "Restaurant ordering menu items: burger, cheeseburger, pizza, frites, coca, coca zéro, boisson, table, réservation, livraison, emporter, burger, cheeseburger, drinks. Restaurante pedido, mesa. Restaurant Bestellung.",
                "response_format": "json",
                "temperature": 0.0
            }
            if wh_lang:
                groq_kwargs["language"] = wh_lang

            transcription = self.client.audio.transcriptions.create(**groq_kwargs)
            return transcription.text.strip()

        try:
            return await loop.run_in_executor(None, _call_groq)
        except Exception as e:
            logger.error(f"[GROQ STT] Fallback transcription failed: {e}")
            return None


class UnifiedSTTManager:
    """
    Unified STT Manager with CPU Guard & Latency-Driven Failover.
    """

    def __init__(self, language: str = "fr"):
        self.language = language
        self.local_whisper = FasterWhisperSTT(model_size=os.getenv("WHISPER_MODEL_SIZE", "small"), device="cpu", compute_type="int8")

        self.groq_fallback = GroqWhisperSTT()
        self.fallback_count = 0
        self.latency_history = []

    @staticmethod
    def _clean_hallucinated_silence(text: str) -> str:
        if not text:
            return ""
        norm = text.strip().lower()
        # Strictly filter real subtitle artifacts, NOT genuine customer words like 'merci' or 'thanks'
        hallucinations = {
            "thank you for watching.", "thanks for watching!", "thanks for watching.",
            "thank you for watching!", "thank you for watching", "merci d'avoir regardé cette vidéo.",
            "sous-titres réalisés par la communauté d'amara.org", "amara.org", "subtitles by..."
        }
        if norm in hallucinations or norm.startswith("sous-titres") or norm.startswith("subtitles by") or norm.startswith("amara.org"):
            logger.info(f"[WHISPER HALLUCINATION FILTERED] Dropped silence artifact: '{text}'")
            return ""
        return text

    async def transcribe_audio(self, audio_bytes: bytes, language: str = "fr") -> str:
        start_time = time.perf_counter()
        cpu_usage = psutil.cpu_percent(interval=None)

        # 1. Overload Check: If CPU > 85%, immediately route to Groq Whisper Turbo
        if cpu_usage > 85.0 and self.groq_fallback.client:
            self.fallback_count += 1
            logger.warning(f"[STT OVERLOAD # {self.fallback_count}] CPU at {cpu_usage}%. Routing to Groq Whisper Turbo...")
            text = await self.groq_fallback.transcribe(audio_bytes, language=language)
            stt_ms = (time.perf_counter() - start_time) * 1000
            cleaned_text = self._clean_hallucinated_silence(text or "")
            logger.info(f"[GROQ STT] Transcribed in {stt_ms:.1f}ms: '{cleaned_text}'")
            return cleaned_text

        # 2. Try Primary: Local Faster-Whisper
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, lambda: self.local_whisper.transcribe(audio_bytes, language=language))
            stt_ms = (time.perf_counter() - start_time) * 1000
            self.latency_history.append(stt_ms)

            # If local took too long (>800ms) and yielded empty, attempt Groq
            if (stt_ms > 800.0 or not text) and self.groq_fallback.client:
                self.fallback_count += 1
                logger.warning(f"[STT LATENCY EXCEEDED # {self.fallback_count}] Local took {stt_ms:.1f}ms. Falling back to Groq...")
                groq_text = await self.groq_fallback.transcribe(audio_bytes, language=language)
                if groq_text:
                    cleaned_groq = self._clean_hallucinated_silence(groq_text)
                    return cleaned_groq

            cleaned_text = self._clean_hallucinated_silence(text or "")
            logger.info(f"[FASTER-WHISPER] Transcribed in {stt_ms:.1f}ms: '{cleaned_text}'")
            return cleaned_text

        except Exception as e:
            self.fallback_count += 1
            logger.warning(f"[STT FAILOVER # {self.fallback_count}] Local Faster-Whisper failed ({e}). Attempting Groq fallback...")
            text = await self.groq_fallback.transcribe(audio_bytes, language=language)
            cleaned_text = self._clean_hallucinated_silence(text or "")
            return cleaned_text


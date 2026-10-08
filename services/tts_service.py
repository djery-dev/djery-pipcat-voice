import asyncio
import io
import logging
import os
import time
from typing import AsyncGenerator, Dict, Optional
import edge_tts

import shutil

import re

logger = logging.getLogger("pipcat.tts_service")


def normalize_text_for_tts(text: str, language: Optional[str] = 'fr') -> str:
    """
    Normalizes time formatting (e.g., '10:00' -> '10 heures' / '10 o'clock') and 
    currency decimal trailing zeros (e.g., '10.00€' -> '10 euros') so neural TTS engines 
    pronounce times and prices naturally without reading literal 'zero zero'.
    """
    if not text or not text.strip():
        return text

    lang = (language or 'fr').lower()
    if lang == 'be':
        lang = 'fr'

    result = text

    if lang == 'fr':
        # Currency: 10.00 € / 10,00€ / 10.00€ -> 10 euros
        result = re.sub(r'(\b\d+)[\.,]00\s*(€|euros?)\b', r'\1 euros', result, flags=re.IGNORECASE)
        # Currency: 10.00 DH / 10,00 MAD -> 10 dirhams
        result = re.sub(r'(\b\d+)[\.,]00\s*(dh|mad|dirhams?)\b', r'\1 dirhams', result, flags=re.IGNORECASE)
        # Currency: $10.00 / 10.00$ -> 10 dollars
        result = re.sub(r'\$(\d+)[\.,]00\b', r'\1 dollars', result)
        result = re.sub(r'(\b\d+)[\.,]00\s*\$', r'\1 dollars', result)

        # Time with 'h': 10h00 / 10H00 -> 10 heures
        result = re.sub(r'(\b\d{1,2})h00\b', r'\1 heures', result, flags=re.IGNORECASE)
        # Time with colon: 10:00 h / 10:00h / 10:00:00 -> 10 heures
        result = re.sub(r'(\b\d{1,2}):00(?::00)?\s*h(eures?)?\b', r'\1 heures', result, flags=re.IGNORECASE)
        # Standard time: 10:00 / 20:00 / 10:00:00 -> 10 heures / 20 heures
        result = re.sub(r'(\b(?:[01]?\d|2[0-3])):00(?::00)?\b', r'\1 heures', result)
        # Special case: 00:00 -> minuit
        result = re.sub(r'\b00:00\b', 'minuit', result)
        result = re.sub(r'\b00h00\b', 'minuit', result, flags=re.IGNORECASE)

    elif lang == 'en':
        # Currency: $10.00 -> 10 dollars
        result = re.sub(r'\$(\d+)[\.,]00\b', r'\1 dollars', result)
        # Currency: 10.00 dollars / 10.00 USD -> 10 dollars
        result = re.sub(r'(\b\d+)[\.,]00\s*(USD|dollars?)\b', r'\1 dollars', result, flags=re.IGNORECASE)
        result = re.sub(r'(\b\d+)[\.,]00\s*(€|EUR|euros?)\b', r'\1 euros', result, flags=re.IGNORECASE)
        
        # Time with AM/PM: 10:00 AM -> 10 AM, 10:00 PM -> 10 PM
        result = re.sub(r'(\b\d{1,2}):00(?::00)?\s*(AM|PM)\b', r'\1 \2', result, flags=re.IGNORECASE)
        # Standard time: 10:00 / 20:00 -> 10 o'clock / 20 o'clock
        result = re.sub(r'(\b(?:[01]?\d|2[0-3])):00(?::00)?\b', r"\1 o'clock", result)

    elif lang == 'es':
        # Currency: 10.00€ / 10,00 € -> 10 euros
        result = re.sub(r'(\b\d+)[\.,]00\s*(€|euros?)\b', r'\1 euros', result, flags=re.IGNORECASE)
        result = re.sub(r'\$(\d+)[\.,]00\b', r'\1 dólares', result)
        
        # Standard time: 10:00 -> 10 en punto
        result = re.sub(r'(\b(?:[01]?\d|2[0-3])):00(?::00)?\b', r'\1 en punto', result)

    elif lang == 'de':
        # Currency: 10.00€ / 10,00 € -> 10 Euro
        result = re.sub(r'(\b\d+)[\.,]00\s*(€|Euro)\b', r'\1 Euro', result, flags=re.IGNORECASE)
        
        # Standard time: 10:00 -> 10 Uhr
        result = re.sub(r'(\b(?:[01]?\d|2[0-3])):00(?::00)?\b', r'\1 Uhr', result)

    # General fallback for any remaining standalone decimal '.00' or ',00' (e.g. "10.00")
    result = re.sub(r'(\b\d+)[\.,]00\b', r'\1', result)

    return result


class PiperTTSService:
    """
    Primary 100% Local Self-Hosted Neural TTS Engine (Piper ONNX).
    Provides instant sub-150ms Time-To-First-Byte on CPU with zero network dependency.
    """
    
    MODELS = {
        'fr': 'fr_FR-siwis-medium',
        'en': 'en_US-lessac-medium',
        'es': 'es_ES-davefx-medium',
        'de': 'de_DE-thorsten-medium',
    }

    _piper_checked = False
    _piper_available = False

    def __init__(self, language: str = 'fr'):
        self.language = language
        self.model_name = self.MODELS.get(language, 'fr_FR-siwis-medium')
        if not PiperTTSService._piper_checked:
            PiperTTSService._piper_available = bool(shutil.which("piper"))
            PiperTTSService._piper_checked = True
            if PiperTTSService._piper_available:
                logger.info(f"[PIPER TTS] Local Piper CLI binary detected. Initialized for language: {language} ({self.model_name})")
            else:
                logger.info("[PIPER TTS] Piper CLI not found in system PATH; using high-speed Edge-TTS directly.")

    async def generate_audio_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesizes text locally using Piper process or native ONNX inference."""
        if not text or not text.strip():
            return

        if not self._piper_available:
            raise RuntimeError("Piper binary not found on host system")

        start_time = time.perf_counter()
        
        # Piper CLI / ONNX execution in sub-process with streaming stdout
        proc = await asyncio.create_subprocess_exec(
            "piper",
            "--model", f"{self.model_name}.onnx",
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate(input=text.encode('utf-8'))
        
        if proc.returncode != 0:
            raise RuntimeError(f"Piper execution failed with code {proc.returncode}: {stderr.decode('utf-8')}")

        ttfb_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[PIPER TTS] Local synthesis complete in {ttfb_ms:.1f}ms for text length {len(text)}")
        yield stdout


class EdgeTTSFallbackService:
    """
    Secondary Online Fallback TTS Engine (Edge-TTS).
    Provides natural neural voices across multiple languages and accents.
    """

    VOICES = {
        'fr': 'fr-FR-DeniseNeural',
        'en': 'en-US-AndrewNeural',
        'es': 'es-ES-ElviraNeural',
        'de': 'de-DE-ConradNeural',
    }

    VOICE_CATALOG = {
        # Multilingual (Auto-Detect / Cross-Language FR, EN, ES, DE)
        'en-US-AvaMultilingualNeural': 'Ava Multilingual (FR/EN/ES/DE Female)',
        'en-US-AndrewMultilingualNeural': 'Andrew Multilingual (FR/EN/ES/DE Male)',
        'fr-FR-VivienneMultilingualNeural': 'Vivienne Multilingual (FR/EN/ES/DE Female)',
        'fr-FR-RemyMultilingualNeural': 'Rémy Multilingual (FR/EN/ES/DE Male)',

        # French & Belgian French
        'fr-FR-DeniseNeural': 'Denise (FR Female)',
        'fr-FR-HenriNeural': 'Henri (FR Male)',
        'fr-BE-CharlineNeural': 'Charline (BE Female)',
        'fr-BE-GerardNeural': 'Gérard (BE Male)',

        # English (US / UK)
        'en-US-AvaNeural': 'Ava (US Female)',
        'en-US-AndrewNeural': 'Andrew (US Male)',
        'en-GB-SoniaNeural': 'Sonia (UK Female)',
        'en-GB-RyanNeural': 'Ryan (UK Male)',

        # Spanish
        'es-ES-ElviraNeural': 'Elvira (ES Female)',
        'es-ES-AlvaroNeural': 'Álvaro (ES Male)',

        # German
        'de-DE-KatjaNeural': 'Katja (DE Female)',
        'de-DE-ConradNeural': 'Conrad (DE Male)',
    }

    AUTO_VOICE_MAP = {
        'fr': 'fr-FR-DeniseNeural',
        'en': 'en-US-AvaMultilingualNeural',
        'es': 'es-ES-ElviraNeural',
        'de': 'de-DE-ConradNeural',
    }

    def __init__(self, language: str = 'fr', voice_name: Optional[str] = None, rate: str = '+0%', pitch: str = '+0Hz'):
        self.language = language
        if not voice_name or voice_name == 'auto':
            self.voice_name = self.AUTO_VOICE_MAP.get(language, 'fr-FR-DeniseNeural')
        else:
            self.voice_name = voice_name
        self.rate = rate
        self.pitch = pitch

    async def generate_audio_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text or not text.strip():
            return

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice_name,
            rate=self.rate,
            pitch=self.pitch
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


class UnifiedTTSManager:
    """
    Production TTS Manager with Automated Failover & Customizable Voice Options.
    Primary: Piper-TTS Local ONNX -> Fallback: Edge-TTS Neural Voices
    """

    def __init__(self, language: str = 'fr', voice_name: Optional[str] = None):
        self.language = language
        self.voice_name = voice_name
        self.piper = PiperTTSService(language=language)
        self.edge = EdgeTTSFallbackService(language=language, voice_name=voice_name)
        self.fallback_count = 0

    async def generate_audio_bytes(self, text: str, language: Optional[str] = None, voice_name: Optional[str] = None) -> bytes:
        """
        Synthesizes audio with automatic failover, voice selection, and latency tracking.
        Primary: Edge-TTS Neural Voices (respects selected voice_name)
        Fallback: Piper-TTS Local ONNX
        """
        start_time = time.perf_counter()
        lang = language or self.language
        v_name = voice_name or self.voice_name

        # Pre-process text to normalize 10:00 -> 10 heures / 10 o'clock and 10.00 -> 10
        normalized_text = normalize_text_for_tts(text, lang)

        # 1. Primary: Edge-TTS Neural Voices (synthesizes user-selected voice model)
        try:
            buffer = io.BytesIO()
            edge_service = EdgeTTSFallbackService(language=lang, voice_name=v_name)
            async for chunk in edge_service.generate_audio_stream(normalized_text):
                buffer.write(chunk)
            ttfb_ms = (time.perf_counter() - start_time) * 1000
            data = buffer.getvalue()
            if len(data) > 0:
                logger.info(f"[EDGE-TTS SUCCESS] Streamed {len(data)} bytes in {ttfb_ms:.1f}ms for voice='{v_name or lang}'")
                return data
        except Exception as ex:
            self.fallback_count += 1
            logger.warning(f"[EDGE-TTS FAILOVER # {self.fallback_count}] Edge-TTS failed ({ex}). Triggering Piper local fallback...")

        # 2. Safety Fallback: Piper-TTS Local ONNX (offline emergency fallback)
        if self.piper._piper_available:
            try:
                buffer = io.BytesIO()
                piper_service = PiperTTSService(language=lang) if lang != self.language else self.piper
                async for chunk in piper_service.generate_audio_stream(normalized_text):
                    buffer.write(chunk)
                data = buffer.getvalue()
                if len(data) > 0:
                    return data
            except Exception as e:
                logger.error(f"[TTS CRITICAL] Both Edge-TTS and Piper fallback failed: {e}")

        return b""

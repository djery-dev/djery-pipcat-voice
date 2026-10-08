import aiohttp
import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

logger = logging.getLogger("pipcat.djery_brain")

class DjeryBrainPipeline:
    """
    Connects Pipecat Voice Agent directly to Laravel Djery Brain API.
    Handles isolated stopwatch latency tracking, sentence chunking, and timeout safeguards.
    """

    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url.rstrip('/')
        self._filler_index = 0

    async def execute_turn_stream(
        self,
        restaurant_slug: str,
        message: str,
        conversation_id: str,
        customer_phone: Optional[str] = None,
        messages_history: List[Dict[str, Any]] = None,
        language: str = "fr",
        is_initial_greeting: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes turn against Djery Brain API and yields sentence chunks for instant TTS streaming.
        """
        start_brain = time.perf_counter()
        endpoint = f"{self.backend_url}/api/v1/chat/test"

        if messages_history is None:
            messages_history = []

        request_history = list(messages_history)
        if not is_initial_greeting:
            messages_history.append({"role": "user", "content": message})
            request_history.append({"role": "user", "content": message})
        else:
            request_history.append({"role": "user", "content": message})

        payload = {
            "restaurant_slug": restaurant_slug,
            "messages": request_history,
            "conversation_id": conversation_id,
            "customer_phone": customer_phone,
            "language": language,
            "mode": "voice"
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, headers=headers, timeout=10.0) as resp:
                    brain_ms = (time.perf_counter() - start_brain) * 1000

                    if resp.status == 200:
                        data = await resp.json()
                        assistant_text = data.get("content", "").strip()
                        if assistant_text:
                            messages_history.append({"role": "assistant", "content": assistant_text})
                        logger.info(f"[DJERY BRAIN ISOLATED LATENCY] Brain (LLM + Tools) completed in {brain_ms:.1f}ms")

                        # Smart Voice Chunking: keeps questions and choices unified to prevent gaps/silence
                        sentences = self.chunk_assistant_text_for_speech(assistant_text, language)
                        if not sentences and assistant_text:
                            sentences = [self.clean_text_for_speech(assistant_text, language)]
                        total_sentences = len(sentences)
                        for idx, clean_sentence in enumerate(sentences):
                            yield {
                                "type": "sentence",
                                "text": clean_sentence,
                                "full_content": assistant_text,
                                "is_first": idx == 0,
                                "is_last": idx == total_sentences - 1,
                                "brain_latency_ms": brain_ms,
                                "state": data.get("state", {})
                            }
                    else:
                        error_text = await resp.text()
                        logger.error(f"[DJERY BRAIN] Backend error {resp.status}: {error_text}")
                        # Safeguard: Yield emergency fallback
                        yield {
                            "type": "sentence",
                            "text": self.get_timeout_filler(language),
                            "brain_latency_ms": brain_ms,
                            "error": True
                        }

        except asyncio.TimeoutError:
            brain_ms = (time.perf_counter() - start_brain) * 1000
            logger.warning(f"[DJERY BRAIN TIMEOUT] Laravel API exceeded 10.0s ({brain_ms:.1f}ms). Playing instant filler audio...")
            yield {
                "type": "sentence",
                "text": self.get_timeout_filler(language),
                "brain_latency_ms": brain_ms,
                "timeout": True
            }
        except Exception as e:
            logger.error(f"[DJERY BRAIN EXCEPTION] Failed to query Brain: {e}")
            yield {
                "type": "sentence",
                "text": self.get_timeout_filler(language),
                "error": True
            }

    def get_timeout_filler(self, language: str) -> str:
        fillers = {
            "fr": [
                "Un instant s'il vous plaît, je vérifie ces détails.",
                "Laissez-moi vérifier ça pour vous.",
                "Je m'en occupe tout de suite.",
                "Je vérifie ça immédiatement.",
            ],
            "en": [
                "Just a moment please, let me check those details for you.",
                "Let me check that for you.",
                "I'll take care of that right away.",
                "Checking that for you right now.",
            ],
            "es": [
                "Un momento por favor, déjame verificar esos detalles.",
                "Déjame comprobar eso por ti.",
                "Me encargo de eso de inmediato.",
                "Lo verifico inmediatamente.",
            ],
            "de": [
                "Einen Moment bitte, ich prüfe die Details für Sie.",
                "Lassen Sie mich das für Sie überprüfen.",
                "Ich kümmere mich sofort darum.",
                "Ich überprüfe das sofort.",
            ],
        }
        lang_key = (language or "fr").lower()
        options = fillers.get(lang_key, fillers["fr"])
        phrase = options[self._filler_index % len(options)]
        self._filler_index += 1
        return phrase

    @staticmethod
    def clean_text_for_speech(text: str, language: str = 'fr') -> str:
        """
        Transforms formatted restaurant text (with markdown, checkout URLs, UUIDs, bullet points, prices, etc.)
        into natural, fluent spoken language for neural TTS with minimum latency.
        """
        if not text:
            return ""
        
        t = text
        # 1. Clean markdown formatting
        t = re.sub(r'[*_`#]', '', t)

        # 2. Never recite long checkout/web URLs in voice calls; replace with natural conversational phrase
        lang = (language or 'fr').lower()
        if lang.startswith('fr') or lang == 'be':
            t = re.sub(r'(?:avec|par|via)?\s*ce lien\s*:\s*https?://\S+', 'avec le lien sécurisé envoyé par SMS', t, flags=re.IGNORECASE)
            t = re.sub(r'https?://\S+', 'le lien sécurisé envoyé par SMS', t, flags=re.IGNORECASE)
        elif lang.startswith('es'):
            t = re.sub(r'(?:con|por|a través de)?\s*este enlace\s*:\s*https?://\S+', 'con el enlace seguro enviado por SMS', t, flags=re.IGNORECASE)
            t = re.sub(r'https?://\S+', 'el enlace seguro enviado por SMS', t, flags=re.IGNORECASE)
        elif lang.startswith('de'):
            t = re.sub(r'(?:mit|über)?\s*diesem Link\s*:\s*https?://\S+', 'mit dem sicheren Link per SMS', t, flags=re.IGNORECASE)
            t = re.sub(r'https?://\S+', 'dem sicheren Link per SMS', t, flags=re.IGNORECASE)
        else:  # en / default
            t = re.sub(r'(?:with|via)?\s*this link\s*:\s*https?://\S+', 'with the secure link sent by SMS', t, flags=re.IGNORECASE)
            t = re.sub(r'https?://\S+', 'the secure link sent by SMS', t, flags=re.IGNORECASE)

        # 3. Shorten UUIDs to the first 4 characters for voice calls (e.g. 01a0cf65-dbd5-... -> 01a0)
        t = re.sub(r'\b([0-9a-fA-F]{4})[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', r'\1', t)

        # 4. Strip trailing .00 or ,00 from prices/numbers (e.g. 150.00 € -> 150 €, 150,00 € -> 150 €, 150.00$ -> 150$)
        # so TTS speaks "150 euros" instead of "150 point zero zero euros" / "150 virgule zéro zéro euros"
        t = re.sub(r'(\b\d+)[.,]00(?!\d)', r'\1', t)

        # 5. Convert currency additions (+€5, +5€, etc.) and strip option inclusion tags ("— inclus", "- included", etc.)
        if lang.startswith('fr') or lang == 'be':
            t = re.sub(r'\+\s*€?\s*(\d+(?:[.,]\d+)?)\s*€?', r'pour \1 euros de plus', t)
            t = re.sub(r'\s*[—–-]\s*\b(?:inclus|incluse|incluses)\b|\s*\(\s*(?:inclus|incluse|incluses)\s*\)', '', t, flags=re.IGNORECASE)
        elif lang.startswith('es'):
            t = re.sub(r'\+\s*€?\s*(\d+(?:[.,]\d+)?)\s*€?', r'por \1 euros más', t)
            t = re.sub(r'\s*[—–-]\s*\b(?:incluido|incluida|incluidos|incluidas)\b|\s*\(\s*(?:incluido|incluida|incluidos|incluidas)\s*\)', '', t, flags=re.IGNORECASE)
        elif lang.startswith('de'):
            t = re.sub(r'\+\s*€?\s*(\d+(?:[.,]\d+)?)\s*€?', r'für \1 Euro extra', t)
            t = re.sub(r'\s*[—–-]\s*\b(?:inklusive|inklusiv)\b|\s*\(\s*(?:inklusive|inklusiv)\s*\)', '', t, flags=re.IGNORECASE)
        else:  # en / default
            t = re.sub(r'\+\s*€?\s*(\d+(?:[.,]\d+)?)\s*€?', r'for \1 euros extra', t)
            t = re.sub(r'\s*[—–-]\s*\b(?:included)\b|\s*\(\s*(?:included)\s*\)', '', t, flags=re.IGNORECASE)

        # 6. Clean bullets, dashes, line breaks into natural pauses
        t = re.sub(r'[\r\n]+', ' ', t)
        t = re.sub(r'\s*[•\t]+\s*', ', ', t)
        t = re.sub(r'^\s*\d+[\.\)]\s*', '', t)
        t = re.sub(r'\s+\d+[\.\)]\s*', ', ', t)
        t = re.sub(r'[—–]', ',', t)

        # 6. Clean extra spaces and punctuation
        t = re.sub(r'\s*,\s*', ', ', t)
        t = re.sub(r'(?:,\s*){2,}', ', ', t)
        t = re.sub(r'\s*:\s*', ': ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        return t

    @classmethod
    def chunk_assistant_text_for_speech(cls, assistant_text: str, language: str = 'fr') -> List[str]:
        """
        Chunks assistant text into optimal speech segments:
        - If total text is short (<= 90 chars), keeps as one chunk.
        - If longer, isolates the first sentence (Chunk 0, <= 100 chars) as an ultra-fast lead-in
          for sub-400ms TTFB so caller hears voice almost immediately with zero delay.
        - Groups remaining sentences into natural chunks of ~120-180 chars.
        """
        clean_speech = cls.clean_text_for_speech(assistant_text, language)
        if not clean_speech:
            return []
        
        # If very short, return as single chunk
        if len(clean_speech) <= 90:
            return [clean_speech]

        # Split on sentence terminators (. ! ?) or colons followed by a space
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?:])\s+', clean_speech) if s.strip()]
        if not raw_sentences:
            return [clean_speech]

        chunks = []
        # Isolate crisp first sentence (<= 100 chars) for ultra-fast lead-in TTFB
        first = raw_sentences[0]
        chunks.append(first)

        # Group remaining sentences into balanced chunks of ~120-180 chars
        current = ""
        for s in raw_sentences[1:]:
            if not current:
                current = s
            elif len(current) + len(s) < 180:
                current = f"{current} {s}"
            else:
                chunks.append(current)
                current = s
        if current:
            chunks.append(current)

        return chunks if chunks else [clean_speech]



"""Identity of one phone call, read from the parameters Laravel puts in the TwiML <Stream>.

The worker never guesses: which restaurant, which language, which call. If Laravel did not say, the
call is refused instead of silently falling back to a default restaurant (which used to send every
call to "djery-classic" and share one conversation between all callers).
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

SUPPORTED_LANGUAGES = ("fr", "en", "es", "de")

GREETINGS = {
    "fr": "Bonjour et bienvenue chez {name} ! Comment puis-je vous aider aujourd'hui ?",
    "en": "Hello and welcome to {name}! How can I help you today?",
    "es": "¡Hola y bienvenido a {name}! ¿Cómo puedo ayudarle hoy?",
    "de": "Hallo und willkommen bei {name}! Wie kann ich Ihnen heute helfen?",
}

NOT_CONFIGURED = {
    "fr": "Désolé, ce service est momentanément indisponible. Merci de rappeler plus tard.",
    "en": "Sorry, this service is temporarily unavailable. Please call back later.",
    "es": "Lo sentimos, este servicio no está disponible temporalmente. Por favor, vuelva a llamar más tarde.",
    "de": "Entschuldigung, dieser Service ist vorübergehend nicht verfügbar. Bitte rufen Sie später wieder an.",
}


class CallIdentityError(ValueError):
    """The stream did not carry what is needed to run this call safely."""


@dataclass(frozen=True)
class CallContext:
    voice_session_id: str
    call_token: str
    restaurant_slug: str
    restaurant_name: str
    language: str
    caller_phone: Optional[str]

    @property
    def conversation_id(self) -> str:
        # One conversation per call, never shared between callers.
        return f"call_{self.voice_session_id}"

    def greeting(self) -> str:
        return GREETINGS[self.language].format(name=self.restaurant_name or "")

    @classmethod
    def from_start_event(cls, start: Dict[str, Any]) -> "CallContext":
        params = start.get("customParameters") or start.get("custom_parameters") or {}

        session_id = (params.get("session_id") or "").strip()
        token = (params.get("call_token") or "").strip()
        slug = (params.get("restaurant_slug") or "").strip()
        if not session_id or not token or not slug:
            raise CallIdentityError("stream is missing session_id, call_token or restaurant_slug")

        language = (params.get("language") or "fr").strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            language = "fr"

        return cls(
            voice_session_id=session_id,
            call_token=token,
            restaurant_slug=slug,
            restaurant_name=(params.get("restaurant_name") or "").strip(),
            language=language,
            caller_phone=(params.get("caller_phone") or params.get("from") or None),
        )

    @staticmethod
    def not_configured_message(language: str = "fr") -> str:
        return NOT_CONFIGURED.get(language, NOT_CONFIGURED["fr"])

import audioop
import base64
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("pipcat.telnyx_bridge")

class TelnyxFrameSerializer:
    """
    Handles bidirectional Telnyx TeXML Media Streams.
    Converts between 8kHz G.711 mu-law telephony audio and 16kHz linear PCM for STT/TTS.
    """

    def __init__(self, stream_id: Optional[str] = None):
        self.stream_id = stream_id

    def decode_inbound_media(self, payload_b64: str) -> bytes:
        """
        Decodes incoming Telnyx 8kHz mu-law audio frame to 16kHz Linear PCM.
        """
        mulaw_bytes = base64.b64decode(payload_b64)
        # 1. Convert 8kHz mu-law to 8kHz 16-bit linear PCM
        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        # 2. Resample 8kHz linear PCM to 16kHz linear PCM (Whisper format)
        pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
        return pcm_16k

    def encode_outbound_media(self, pcm_bytes: bytes, input_sample_rate: int = 16000) -> str:
        """
        Encodes linear PCM audio from TTS into 8kHz G.711 mu-law base64 string for Telnyx.
        """
        # 1. Resample to 8kHz if needed
        if input_sample_rate != 8000:
            pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, input_sample_rate, 8000, None)
        else:
            pcm_8k = pcm_bytes

        # 2. Convert 8kHz Linear PCM to 8kHz mu-law
        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
        return base64.b64encode(mulaw_bytes).decode("utf-8")

    def build_media_message(self, pcm_bytes: bytes, input_sample_rate: int = 16000) -> str:
        """
        Builds Telnyx JSON media message frame.
        """
        payload_b64 = self.encode_outbound_media(pcm_bytes, input_sample_rate)
        return json.dumps({
            "event": "media",
            "stream_id": self.stream_id,
            "media": {
                "payload": payload_b64
            }
        })

    def build_clear_message(self) -> str:
        """
        Builds Telnyx clear frame for instant interruption (barge-in).
        """
        return json.dumps({
            "event": "clear",
            "stream_id": self.stream_id
        })

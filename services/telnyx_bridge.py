import audioop
import base64
import json
import logging
import subprocess
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

    def encode_outbound_media(self, audio_bytes: bytes, input_sample_rate: int = 16000) -> str:
        """
        Encodes linear PCM audio (or compressed audio like MP3 from Edge-TTS) into 8kHz G.711 mu-law base64 string for Twilio & Telnyx.
        """
        # If input is compressed audio (e.g. MP3 from Edge-TTS), decode directly to 8kHz linear PCM using ffmpeg
        if (audio_bytes.startswith(b"ID3") or 
            audio_bytes.startswith(b"\xff\xfb") or 
            audio_bytes.startswith(b"\xff\xf3") or 
            audio_bytes.startswith(b"OggS") or 
            audio_bytes.startswith(b"RIFF")):
            try:
                proc = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", "pipe:1"],
                    input=audio_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                pcm_8k = proc.stdout
                mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
                return base64.b64encode(mulaw_bytes).decode("utf-8")
            except Exception as e:
                logger.error(f"[TELEPHONY AUDIO DECODE ERR] ffmpeg decode failed: {e}")

        # 1. Resample to 8kHz if needed
        pcm_bytes = audio_bytes
        if input_sample_rate != 8000:
            pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, input_sample_rate, 8000, None)
        else:
            pcm_8k = pcm_bytes

        # 2. Convert 8kHz Linear PCM to 8kHz mu-law
        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
        return base64.b64encode(mulaw_bytes).decode("utf-8")

    def build_media_message(self, pcm_bytes: bytes, input_sample_rate: int = 16000) -> str:
        """
        Builds Telnyx & Twilio JSON media message frame.
        """
        payload_b64 = self.encode_outbound_media(pcm_bytes, input_sample_rate)
        return json.dumps({
            "event": "media",
            "streamSid": self.stream_id,
            "stream_id": self.stream_id,
            "media": {
                "payload": payload_b64
            }
        })

    def build_clear_message(self) -> str:
        """
        Builds Telnyx & Twilio clear frame for instant interruption (barge-in).
        """
        return json.dumps({
            "event": "clear",
            "streamSid": self.stream_id,
            "stream_id": self.stream_id
        })

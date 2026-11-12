"""Test Piper TTS fallback voice — generates a WAV file you can play."""
import asyncio
import wave
from app.tts import PiperTTS


def main() -> None:
    piper = PiperTTS()
    text = (
        "Hello, this is Alvin's Piper TTS fallback voice. "
        "If you can hear this, Edge TTS has fallen back to local Piper synthesis."
    )
    pcm = piper.synthesize_pcm(text)
    if not pcm:
        print("Piper TTS synthesis failed. Is the voice model downloaded?")
        return

    out_path = "piper_test_output.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)
    print(f"Wrote {len(pcm)} bytes of PCM to {out_path}")
    print("Play it with: start piper_test_output.wav")


if __name__ == "__main__":
    main()

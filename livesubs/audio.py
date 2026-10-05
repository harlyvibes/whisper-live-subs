"""Audio capture: WASAPI loopback (whatever Windows is playing) or a microphone.

Produces 16 kHz mono float32 audio, which is what Whisper expects.
"""
from __future__ import annotations

import queue

import av
import numpy as np
import pyaudiowpatch as pyaudio

SAMPLE_RATE = 16000


def list_devices(source: str) -> list[str]:
    """Names of capture devices for the given source ("loopback" | "microphone")."""
    pa = pyaudio.PyAudio()
    try:
        names = []
        if source == "loopback":
            for d in pa.get_loopback_device_info_generator():
                names.append(d["name"])
        else:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if (d["hostApi"] == wasapi and d["maxInputChannels"] > 0
                        and not d.get("isLoopbackDevice", False)):
                    names.append(d["name"])
        return names
    finally:
        pa.terminate()


class AudioCapture:
    def __init__(self, source: str = "loopback", device_name: str = ""):
        self.source = source
        self.device_name = device_name
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=2000)
        self._resampler = None
        self.device_label = ""
        self.channels = 1
        self.rate = SAMPLE_RATE

    def _find_device(self, pa: pyaudio.PyAudio) -> dict:
        if self.source == "loopback":
            if self.device_name:
                for d in pa.get_loopback_device_info_generator():
                    if d["name"] == self.device_name:
                        return d
            return pa.get_default_wasapi_loopback()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        if self.device_name:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if (d["hostApi"] == wasapi["index"] and d["name"] == self.device_name
                        and d["maxInputChannels"] > 0):
                    return d
        return pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        dev = self._find_device(self._pa)
        self.device_label = dev["name"]
        self.channels = max(1, int(dev["maxInputChannels"]))
        self.rate = int(dev["defaultSampleRate"])
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

        def callback(in_data, frame_count, time_info, status):
            try:
                self._q.put_nowait(in_data)
            except queue.Full:  # consumer stalled; drop oldest audio
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(in_data)
                except queue.Empty:
                    pass
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=int(self.rate * 0.05),
            stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except OSError:
            pass
        self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def read(self, timeout: float = 0.1) -> np.ndarray:
        """Return all audio captured since the last call (16 kHz mono), possibly empty."""
        chunks = []
        try:
            chunks.append(self._q.get(timeout=timeout))
            while True:
                chunks.append(self._q.get_nowait())
        except queue.Empty:
            pass
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        raw = np.frombuffer(b"".join(chunks), dtype=np.float32)
        mono = raw.reshape(-1, self.channels).mean(axis=1).astype(np.float32)
        if self.rate == SAMPLE_RATE:
            return mono
        frame = av.AudioFrame.from_ndarray(mono.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = self.rate
        out = [f.to_ndarray().reshape(-1) for f in self._resampler.resample(frame)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)

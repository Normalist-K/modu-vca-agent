#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, AsyncGenerator, Dict

import aiohttp
from loguru import logger

from pipecat.audio.utils import resample_audio, create_default_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService


class StyleTTS2Service(TTSService):
    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.base_url = base_url
        self._sample_rate = sample_rate
        self._aiohttp_session = aiohttp_session
        self._resampler = create_default_resampler()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        text = text.replace('"', '').strip()
        if len(text) <= 1:
            logger.debug(f'Skipping TTS: [{text}]')
            return
        logger.debug(f"Generating TTS: [{text}]")

        url = self.base_url + "/tts"

        payload = {
            "text": text,
        }

        await self.start_ttfb_metrics()

        async with self._aiohttp_session.post(url, json=payload) as r:
            if r.status != 200:
                text = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status}, error: {text})")
                yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                return

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            buffer = bytearray()
            async for chunk in r.content.iter_chunked(1024):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    # Append new chunk to the buffer.
                    buffer.extend(chunk)

                    # Check if buffer has enough data for processing.
                    while (
                        len(buffer) >= 48000
                    ):  # Assuming at least 0.5 seconds of audio data at 24000 Hz
                        # Process the buffer up to a safe size for resampling.
                        process_data = buffer[:48000]
                        # Remove processed data from buffer.
                        buffer = buffer[48000:]

                        # XTTS uses 24000 so we need to resample to our desired rate.
                        resampled_audio = await self._resampler.resample(
                            bytes(process_data), 24000, self._sample_rate
                        )
                        # Create the frame with the resampled audio
                        frame = TTSAudioRawFrame(resampled_audio, self._sample_rate, 1)
                        yield frame

            # Process any remaining data in the buffer.
            if len(buffer) > 0:
                resampled_audio = await self._resampler.resample(
                    bytes(buffer), 24000, self._sample_rate
                    )
                frame = TTSAudioRawFrame(resampled_audio, self._sample_rate, 1)
                yield frame

            yield TTSStoppedFrame()

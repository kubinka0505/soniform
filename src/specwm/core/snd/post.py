import numpy as np
from pathlib import Path

from ...config import Settings

#-=-=-=-#

def post_process(
    audio: np.ndarray,
    cfg: Settings
) -> np.ndarray:
    """
    Apply final length adjustment, fades, and level processing.
    """
    audio = audio.astype(np.float32)
    target_len = int(cfg.sample_rate * cfg.duration)
    cur_len = audio.shape[0]

    # pad or trim
    if cur_len < target_len:
        pad_len = target_len - cur_len

        if audio.ndim == 1:
            pad_shape = (0, pad_len)
        else:
            pad_shape = ((0, pad_len), (0, 0))

        audio = np.pad(audio, pad_shape)
    else:
        audio = audio[:target_len]

    # fades
    if getattr(cfg, "fade_ms", 0) > 0 and audio.shape[0] > 1:
        fade_samples = max(1, int(cfg.sample_rate * cfg.fade_ms / 1000))
        fade_samples = min(fade_samples, audio.shape[0] // 2)

        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        # reshape fade vectors to support both 1D (mono) and 2D (multi-channel)
        if audio.ndim > 1:
            fade_in = fade_in[:, None]
            fade_out = fade_out[:, None]

        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

    if getattr(cfg, "normalize", True):
        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = audio / peak
    else:
        audio = np.clip(audio, -1.0, 1.0)

    return audio
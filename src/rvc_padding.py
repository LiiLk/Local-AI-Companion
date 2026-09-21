"""Shared helpers for trimming the silence RVC padding adds to short clips.

RVC models require a minimum input length, so short clips are padded before
conversion. The model output keeps that padding, which shows up as leading and
trailing silence. These helpers compute which output samples must be dropped to
remove exactly the padding that was added, with no dependency beyond the
standard library so the isolated RVC worker can import them too.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_trim_bounds(
    *,
    pad_left_samples: int,
    pad_right_samples: int,
    input_sample_rate: int,
    output_sample_rate: int,
    output_length: int,
) -> tuple[int, int]:
    """Return the output slice bounds that remove the input padding.

    Padding is expressed in input samples. When the output sample rate differs,
    the padding is converted at the output/input rate ratio. When there is
    nothing to trim, or the output is too short to trim safely, the full range
    ``(0, output_length)`` is returned and no audio is dropped.
    """
    if output_length <= 0:
        return 0, output_length
    if pad_left_samples <= 0 and pad_right_samples <= 0:
        return 0, output_length

    in_rate = int(input_sample_rate or 0)
    out_rate = int(output_sample_rate or 0)
    if in_rate <= 0 or out_rate <= 0:
        return 0, output_length

    ratio = out_rate / in_rate
    start = max(0, int(round(pad_left_samples * ratio)))
    end = min(output_length, output_length - int(round(pad_right_samples * ratio)))

    if start >= end:
        logger.warning(
            "RVC pad trim skipped: output too short "
            "(length=%s, start=%s, end=%s, pad_left=%s, pad_right=%s)",
            output_length,
            start,
            end,
            pad_left_samples,
            pad_right_samples,
        )
        return 0, output_length

    return start, end
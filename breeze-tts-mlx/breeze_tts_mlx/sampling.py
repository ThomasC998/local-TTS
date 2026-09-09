from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class NumpySampler:
    """Backend-independent sampler with the same operation order as HF."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        logits: np.ndarray,
        config: SamplingConfig,
        *,
        suppress_from: int | None = None,
        suppress_to: int | None = None,
        token_history: list[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> int:
        config.validate()
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")

        scores = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
        if not np.isfinite(scores).all():
            invalid = int((~np.isfinite(scores)).sum())
            raise RuntimeError(
                "model produced non-finite logits before sampling "
                f"({invalid}/{scores.size} invalid values)"
            )
        if token_history and repetition_penalty != 1.0:
            for token in np.unique(np.asarray(token_history, dtype=np.int64)):
                if 0 <= token < scores.size:
                    scores[token] = (
                        scores[token] / repetition_penalty
                        if scores[token] > 0
                        else scores[token] * repetition_penalty
                    )

        if suppress_from is not None:
            end = scores.size if suppress_to is None else min(suppress_to, scores.size)
            scores[max(0, suppress_from) : end] = -np.inf

        if not config.do_sample:
            return int(np.argmax(scores))

        scores /= config.temperature
        if config.top_k > 0 and config.top_k < scores.size:
            threshold = np.partition(scores, -config.top_k)[-config.top_k]
            scores[scores < threshold] = -np.inf

        if config.top_p < 1.0:
            order = np.argsort(scores)[::-1]
            ordered = scores[order]
            finite = np.isfinite(ordered)
            if not finite.any():
                raise RuntimeError("all logits were suppressed")
            max_score = np.max(ordered[finite])
            exp_scores = np.zeros_like(ordered, dtype=np.float64)
            exp_scores[finite] = np.exp(ordered[finite] - max_score)
            ordered_probs = exp_scores / exp_scores.sum()
            remove = np.cumsum(ordered_probs) > config.top_p
            remove[1:] = remove[:-1].copy()
            remove[0] = False
            scores[order[remove]] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            raise RuntimeError("all logits were suppressed")
        max_score = np.max(scores[finite])
        probs = np.zeros_like(scores, dtype=np.float64)
        probs[finite] = np.exp(scores[finite] - max_score)
        probs /= probs.sum()
        return int(self._rng.choice(scores.size, p=probs))

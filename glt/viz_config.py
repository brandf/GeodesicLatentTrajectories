from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VizConfig:
    enabled: bool = False
    scalar_every: int = 10
    hist_every: int = 200
    image_every: int = 1000
    sequence_index: int = 0

    def wants_scalar(self, step: int) -> bool:
        return self.enabled and self.scalar_every > 0 and step % self.scalar_every == 0

    def wants_hist(self, step: int) -> bool:
        return self.enabled and self.hist_every > 0 and step % self.hist_every == 0

    def wants_image(self, step: int) -> bool:
        return self.enabled and self.image_every > 0 and step % self.image_every == 0

    def needs_batch_data(self, step: int) -> bool:
        if not self.enabled:
            return False
        return any([
            self.wants_scalar(step),
            self.wants_hist(step),
            self.wants_image(step),
        ])

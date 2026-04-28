import logging
from abc import ABC, abstractmethod
from pathlib import Path


class PipelineStep(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        pass


class DataLoader(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def load(self, destination_dir: Path) -> list[Path]:
        pass

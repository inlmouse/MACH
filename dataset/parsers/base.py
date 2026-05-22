# dataset/parsers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseParser(ABC):
    def __init__(self, ann_file: str, image_root: str):
        self.ann_file = ann_file
        self.image_root = image_root

    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """
        Return a list of UnifiedAnnotation dicts
        """
        pass

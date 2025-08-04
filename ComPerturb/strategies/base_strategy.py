from abc import ABC, abstractmethod
from utils.prompt_utils import split_prompt, combine_prompt

class BaseStrategy(ABC):
    def __init__(self):
        pass
    
    def process_context(self, context):
        """
        Basic process for handling context
        """
        parts = split_prompt(context)
        if len(parts) < 5:
            parts.update({tag: "" for tag in ["Role", "Directive", "Additional Information", "Output Formatting", "Examples"] if tag not in parts})
        
        return self._process_parts(parts)
    
    @abstractmethod
    def _process_parts(self, parts):
        """
        Strategy for processing each part
        """
        pass 
from strategies.base_strategy import BaseStrategy
from utils.prompt_utils import combine_prompt

class CODStrategy(BaseStrategy):
    def _process_parts(self, parts):
        """
        Component Deletion Strategy
        """
        results = []
        for tag in parts.keys():
            modified_parts = parts.copy()
            modified_context = combine_prompt(modified_parts, exclude_tag=tag)
            results.append({
                'context': modified_context,
                'tag': tag
            })
        return results 
from strategies.base_strategy import BaseStrategy
from utils.prompt_utils import combine_prompt, delete_random_words

class WODStrategy(BaseStrategy):
    def _process_parts(self, parts):
        """
        Word Deletion Strategy
        """
        results = []
        for tag in parts.keys():
            modified_parts = parts.copy()
            if modified_parts[tag].strip():
                modified_parts[tag] = delete_random_words(modified_parts[tag])
            modified_context = combine_prompt(modified_parts)
            results.append({
                'context': modified_context,
                'tag': tag
            })
        return results 
from strategies.base_strategy import BaseStrategy
from utils.prompt_utils import combine_prompt, insert_random_characters

class SCIStrategy(BaseStrategy):
    def _process_parts(self, parts):
        """
        Special Character Insertion Strategy
        """
        results = []
        for tag in parts.keys():
            modified_parts = parts.copy()
            if modified_parts[tag].strip():
                modified_parts[tag] = insert_random_characters(modified_parts[tag])
            modified_context = combine_prompt(modified_parts)
            results.append({
                'context': modified_context,
                'tag': tag
            })
        return results 
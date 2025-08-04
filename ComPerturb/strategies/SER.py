from strategies.base_strategy import BaseStrategy
from utils.prompt_utils import combine_prompt, rewrite_random_sentence

class SERStrategy(BaseStrategy):
    def _process_parts(self, parts):
        """
        Sentence Rewriting Strategy
        """
        results = []
        for tag in parts.keys():
            modified_parts = parts.copy()
            if modified_parts[tag].strip():
                modified_parts[tag] = rewrite_random_sentence(modified_parts[tag])
            modified_context = combine_prompt(modified_parts)
            results.append({
                'context': modified_context,
                'tag': tag
            })
        return results 
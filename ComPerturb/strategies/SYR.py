from strategies.base_strategy import BaseStrategy
from utils.prompt_utils import combine_prompt, replace_words_with_synonyms

class SYRStrategy(BaseStrategy):
    def _process_parts(self, parts):
        """
        Synonym Replacement Strategy
        """
        results = []
        for tag in parts.keys():
            modified_parts = parts.copy()
            if modified_parts[tag].strip():
                modified_parts[tag] = replace_words_with_synonyms(modified_parts[tag])
            modified_context = combine_prompt(modified_parts)
            results.append({
                'context': modified_context,
                'tag': tag
            })
        return results 
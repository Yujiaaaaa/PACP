import re
import random
from nltk.corpus import wordnet
import nltk

def split_prompt(prompt):
    """
    Split the prompt into different parts according to tags
    """
    parts = {}
    tags = ['Role', 'Directive', 'Additional Information', 'Output Formatting', 'Examples']
    
    for tag in tags:
        pattern = re.compile(rf'<{tag}>(.*?)</{tag}>', re.DOTALL)
        match = pattern.search(prompt)
        
        if match:
            parts[tag] = match.group(1).strip()
        else:
            parts[tag] = ""
    
    return parts

def combine_prompt(parts, exclude_tag=None):
    """
    Recombine the split parts into a complete prompt
    """
    prompt = ""
    for tag, content in parts.items():
        if content and tag != exclude_tag:  
            prompt += content.strip() + " "
    return prompt.strip()

def get_variable_parts(prompt, variable_tag):
    """
    Get the variable part of the prompt that needs to be changed
    """
    parts = split_prompt(prompt)
    variable_content = parts.get(variable_tag, "")
    return variable_content

def replace_variable_part(prompt, variable_tag, new_content):
    parts = split_prompt(prompt)
    parts[variable_tag] = new_content
    return combine_prompt(parts)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def rewrite_sentence(sentence):
    words = sentence.split()
    rewritten_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            rewritten_words.append(random.choice(synonyms))
        else:
            rewritten_words.append(word)
    return " ".join(rewritten_words)

def rewrite_random_sentence(part):
    sentences = [s.strip() for s in part.split('.') if s.strip()]
    if not sentences:
        return part
    idx = random.randint(0, len(sentences) - 1)
    sentences[idx] = rewrite_sentence(sentences[idx])
    return '. '.join(sentences) + '.'

def replace_words_with_synonyms(sentence, replacement_ratio=0.2):
    words = sentence.split()
    if not words:
        return sentence
    
    # Calculate number of words to replace based on percentage
    num_replacements = max(1, int(len(words) * replacement_ratio))
    
    word_indices = list(range(len(words)))
    random.shuffle(word_indices)
    replaced_count = 0
    
    for word_index in word_indices:
        if replaced_count >= num_replacements:
            break
        
        original_word = words[word_index]
        synonyms = get_synonyms(original_word)
        
        if synonyms:
            new_word = random.choice(synonyms)
            words[word_index] = new_word
            replaced_count += 1

    return " ".join(words)

def delete_random_words(sentence, deletion_ratio=0.2):
    words = sentence.split()
    if not words:
        return sentence
    
    # Calculate number of words to delete based on percentage
    num_deletions = max(1, int(len(words) * deletion_ratio))
    
    # Ensure we don't delete all words
    if len(words) <= num_deletions:
        return ""
    
    indices_to_delete = random.sample(range(len(words)), num_deletions)
    new_words = [word for idx, word in enumerate(words) if idx not in indices_to_delete]
    
    return " ".join(new_words)

def insert_random_characters(text, insertion_ratio=0.2):
    if not text:
        return text
    
    special_chars = ['@', '#', '$', '%', '&', '*', '!', '?', '(', ')', '[', ']', '{', '}', '<', '>']
    
    # Calculate number of characters to insert based on percentage
    num_insertions = max(1, int(len(text) * insertion_ratio))
    
    # Ensure we don't try to insert more characters than possible positions
    if len(text) < num_insertions:
        return text
    
    positions = sorted(random.sample(range(len(text) + 1), num_insertions))
    chars = random.choices(special_chars, k=num_insertions)
    
    modified_text = ""
    prev_pos = 0
    for pos, char in zip(positions, chars):
        modified_text += text[prev_pos:pos] + char
        prev_pos = pos
    modified_text += text[prev_pos:]
    
    return modified_text 
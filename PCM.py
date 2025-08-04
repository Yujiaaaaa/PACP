import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

punkt_param = PunktParameters()
sentence_tokenizer = PunktSentenceTokenizer(punkt_param)

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)


embedding_model = SentenceTransformer('./all-MiniLM-L6-v2')

def compute_pcm(prompt, task_complexity=5, alpha=1, beta=1, gamma=1, delta=1, epsilon=1):
    print("Step 0: Preprocessing prompt: separating natural language and code...")
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", prompt, flags=re.DOTALL)
    code_only_text = "\n".join(code_blocks)
    natural_text = re.sub(r"```(?:python)?\s*.*?```", "", prompt, flags=re.DOTALL)

    print("Step 1: Splitting sentences and words in natural text...")
    sentences = sentence_tokenizer.tokenize(natural_text)
    words = nltk.word_tokenize(natural_text)

    print("Step 2: Computing lexical complexity (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit([natural_text])
    idfs = vectorizer.idf_
    lexical_complexity = np.mean(idfs)

    print("Step 3: Computing syntactic complexity (chunk tree depth)...")
    def sentence_depth(sentence):
        tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        return tree.height()
    syntactic_complexity = np.mean([sentence_depth(s) for s in sentences]) if sentences else 0

    print("Step 4: Computing semantic complexity (embedding + cosine distance)...")
    code_sentences = code_only_text.split('\n') if code_only_text.strip() else []
    all_sentences = sentences + code_sentences
    embeddings = embedding_model.encode(all_sentences, show_progress_bar=False)
    cosine_similarities = np.inner(embeddings, embeddings)
    cosine_distances = 1 - cosine_similarities
    i_upper = np.triu_indices_from(cosine_distances, k=1)
    semantic_complexity = np.mean(cosine_distances[i_upper]) if len(all_sentences) > 1 else 0

    print("Step 5: Computing structural complexity...")
    code_lines = code_only_text.strip().count('\n') + 1 if code_only_text.strip() else 0
    structural_complexity = (
        len(sentences) * np.log(len(words) / len(sentences) + 1)
        + 0.5 * np.log(code_lines + 1)
    ) if len(sentences) > 0 else 0

    print("Step 6: Combining all into PCM score...")
    pcm = (alpha * lexical_complexity +
           beta * syntactic_complexity +
           gamma * semantic_complexity +
           delta * structural_complexity +
           epsilon * task_complexity)

    print("All steps done.")
    return {
        'lexical_complexity': lexical_complexity,
        'syntactic_complexity': syntactic_complexity,
        'semantic_complexity': semantic_complexity,
        'structural_complexity': structural_complexity,
        'task_complexity': task_complexity,
        'PCM': pcm
    }

prompt = '''As a seasoned Python developer, you have extensive experience in writing and refining Python code, and your expertise includes various domains such as software engineering, data processing, and algorithm optimization. Your task is to either create new Python scripts that fulfill specific functional requirements or modify existing Python code to enhance its performance, readability, or functionality. These tasks involve implementing basic functions, optimizing existing code, and ensuring that the code adheres to best practices in software development. The output should be a well-structured Python script. It should include clear comments explaining the purpose of each section of the code. The script should be formatted according to PEP 8 guidelines, ensuring readability and consistency. Below is an instruction that describes a task, Write a response that appropriately completes the request. Given a Python dictionary, print out the keys and values that are duplicated.my_dict = {'a': 'apple', 'b': 'banana', 'c': 'apple'}. Example : Input: Generate a program in Python that takes in a string and returns the reverse of the string."Hello World" Output: def reverse_string(str): """ Reverses a given string """ rev_str = "" for c in str: rev_str = c + rev_str return rev_str str = "Hello World" rev_str = reverse_string(str) print(rev_str)'''


result = compute_pcm(prompt)
print("\nPrompt Complexity Breakdown:")
for k, v in result.items():
    print(f"{k}: {v:.3f}")

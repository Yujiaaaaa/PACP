import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import re 

client = OpenAI(
    api_key = "",
    base_url = ""
)


def build_sentence_prompt(sentence, left_context1=None, left_context2=None, right_context1=None, right_context2=None):
    context_str = ""
    if left_context1:
        context_str += f"Previous sentence 1: {left_context1}\n"
    if left_context2:
        context_str += f"Previous sentence 2: {left_context2}\n"
    if right_context1:
        context_str += f"Next sentence 1: {right_context1}\n"
    if right_context2:
        context_str += f"Next sentence 2: {right_context2}\n"
    
    return f"""
You are now an expert in prompt engineering for large language models. You are to categorize the TARGET sentence into five components: <Role>, <Directive>, <Additional Information>, <Output Formatting>, and <Examples>. 

The TARGET sentence is the one you need to classify. The previous and next sentences are only provided for context to help you better understand the TARGET sentence.

Guidelines:
- Only classify the TARGET sentence
- Output only the TARGET sentence wrapped in the appropriate tag(s)
- A sentence may belong to multiple categories (wrap in each relevant tag)
- Never modify or truncate the original sentence
- Do not add any explanations or extra content
- If no category fits, output the sentence unchanged
- Pay special attention to output formatting instructions, which often appear at the end of the prompt
- Make sure to capture complete sentences, including any trailing instructions

Component Definitions:
<Role>: The persona/identity the model should adopt (e.g., "You are a cybersecurity expert specializing in...")
<Directive>: Instructions specifying the task to perform, including question.(e.g., "You are given a password...", "Now complete the question...")
<Additional Information>: Background/context for the task (e.g., "The goal is to verify if the answer provides a clear...", "The task involves ensuring that the answer..." )
<Output Formatting>: Formatting constraints or output structure requirements (e.g., "Answer only returns the number of steps required", "Output the result in the format...")
<Examples>: Input-output pairs or few-shot demonstrations (e.g., "Here are some examples: Positive Example 1...")

Here are some examples of how to classify sentences:

Example 1:
TARGET: "You are a cybersecurity expert specializing in password strength analysis and optimization, and your job is to evaluate and modify passwords to meet stringent security criteria."
Output: <Role>You are a cybersecurity expert specializing in password strength analysis and optimization, and your job is to evaluate and modify passwords to meet stringent security criteria.</Role>

Example 2:
TARGET: "You are given a password and you need to generate the number of steps required to convert the given password to a strong password."
Output: <Directive>You are given a password and you need to generate the number of steps required to convert the given password to a strong password.</Directive>

Example 3:
TARGET: "A password is considered strong if (a) it has at least 6 characters and at most 20 characters; (b) it contains at least one lowercase letter and one uppercase letter, and at least one digit; (c) it does not contain three repeating characters in a row."
Output: <Additional Information>A password is considered strong if (a) it has at least 6 characters and at most 20 characters; (b) it contains at least one lowercase letter and one uppercase letter, and at least one digit; (c) it does not contain three repeating characters in a row.</Additional Information>

Example 4:
TARGET: "Answer only returns the number of steps required."
Output: <Output Formatting>Answer only returns the number of steps required.</Output Formatting>

Example 5:
TARGET: "Here are some examples: Positive Example 1 - Input: password = a Output: 5. Positive Example 2 - Input: password = aA1 Output: 3."
Output: <Examples>Here are some examples: Positive Example 1 - Input: password = a Output: 5. Positive Example 2 - Input: password = aA1 Output: 3.</Examples>

Example 6:
TARGET:"Now complete the question: Despectus tibi sum nec."
Output:<Directive>Now complete the question: Despectus tibi sum nec.

Example 7:
TARGET:"Latin Text: Omnia mutantur, nihil interit. Translation: Everything changes, nothing perishes."
Output:<Examples>Latin Text: Omnia mutantur, nihil interit. Translation: Everything changes, nothing perishes.

{context_str}
Now classify the following TARGET sentence:
{sentence}
""".strip()

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def save_to_json(data, file_path):
    json_data = data.to_dict('records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

df = read_json_file("")

required_columns = ['context']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in JSON file")

context_output = []
def traverse_and_concatenate(lst):
    results = []
    n = len(lst)
    
    if n == 0:
        return results
    if n == 1:
        results.append((lst[0], None, None, None, None))
        return results
        
    for i in range(n):
        left_context1 = lst[i-2] if i > 1 else None
        left_context2 = lst[i-1] if i > 0 else None
        
        right_context1 = lst[i+1] if i < n-1 else None
        right_context2 = lst[i+2] if i < n-2 else None
        
        results.append((lst[i], left_context1, left_context2, right_context1, right_context2))
    
    return results

df_1 = df.copy()

def process_output(output_str, is_initial=True):
    role_pattern = r'<Role>(.*?)</Role>'
    directive_pattern = r'<Directive>(.*?)</Directive>'
    additional_info_pattern = r'<Additional Information>(.*?)</Additional Information>'
    output_format_pattern = r'<Output Formatting>(.*?)</Output Formatting>'
    examples_pattern = r'<Examples>(.*?)</Examples>'
    
    role_content = ' '.join(re.findall(role_pattern, output_str))
    directive_content = ' '.join(re.findall(directive_pattern, output_str))
    additional_info_content = ' '.join(re.findall(additional_info_pattern, output_str))
    output_format_content = ' '.join(re.findall(output_format_pattern, output_str))
    examples_content = ' '.join(re.findall(examples_pattern, output_str))
    
    def clean_content(content):
        if not content:
            return ""
        content = re.sub(r'<[^>]+>', '', content)
        content = ' '.join(content.split())
        return content
    
    final_output = []
    if role_content:
        final_output.append(f"<Role>{clean_content(role_content)}</Role>")
    if directive_content:
        final_output.append(f"<Directive>{clean_content(directive_content)}</Directive>")
    if additional_info_content:
        final_output.append(f"<Additional Information>{clean_content(additional_info_content)}</Additional Information>")
    if output_format_content:
        final_output.append(f"<Output Formatting>{clean_content(output_format_content)}</Output Formatting>")
    if examples_content:
        final_output.append(f"<Examples>{clean_content(examples_content)}</Examples>")
    
    return '\n'.join(final_output)

def merge_classifications(initial_output, missing_classifications):
    role_pattern = r'<Role>(.*?)</Role>'
    directive_pattern = r'<Directive>(.*?)</Directive>'
    additional_info_pattern = r'<Additional Information>(.*?)</Additional Information>'
    output_format_pattern = r'<Output Formatting>(.*?)</Output Formatting>'
    examples_pattern = r'<Examples>(.*?)</Examples>'
    
    role_content = re.findall(role_pattern, initial_output)
    directive_content = re.findall(directive_pattern, initial_output)
    additional_info_content = re.findall(additional_info_pattern, initial_output)
    output_format_content = re.findall(output_format_pattern, initial_output)
    examples_content = re.findall(examples_pattern, initial_output)
    
    for classification in missing_classifications:
        if '<Role>' in classification:
            role_content.extend(re.findall(role_pattern, classification))
        elif '<Directive>' in classification:
            directive_content.extend(re.findall(directive_pattern, classification))
        elif '<Additional Information>' in classification:
            additional_info_content.extend(re.findall(additional_info_pattern, classification))
        elif '<Output Formatting>' in classification:
            output_format_content.extend(re.findall(output_format_pattern, classification))
        elif '<Examples>' in classification:
            examples_content.extend(re.findall(examples_pattern, classification))
    
    final_output = []
    if role_content:
        final_output.append(f"<Role>{' '.join(role_content)}</Role>")
    if directive_content:
        final_output.append(f"<Directive>{' '.join(directive_content)}</Directive>")
    if additional_info_content:
        final_output.append(f"<Additional Information>{' '.join(additional_info_content)}</Additional Information>")
    if output_format_content:
        final_output.append(f"<Output Formatting>{' '.join(output_format_content)}</Output Formatting>")
    if examples_content:
        final_output.append(f"<Examples>{' '.join(examples_content)}</Examples>")
    
    return '\n'.join(final_output)

def check_classification(original_sentences, classified_output):
    role_pattern = r'<Role>(.*?)</Role>'
    directive_pattern = r'<Directive>(.*?)</Directive>'
    additional_info_pattern = r'<Additional Information>(.*?)</Additional Information>'
    output_format_pattern = r'<Output Formatting>(.*?)</Output Formatting>'
    examples_pattern = r'<Examples>(.*?)</Examples>'
    
    classified_content = []
    classified_content.extend(re.findall(role_pattern, classified_output))
    classified_content.extend(re.findall(directive_pattern, classified_output))
    classified_content.extend(re.findall(additional_info_pattern, classified_output))
    classified_content.extend(re.findall(output_format_pattern, classified_output))
    classified_content.extend(re.findall(examples_pattern, classified_output))

    missing_sentences = []
    for sentence in original_sentences:
        sentence = sentence.strip()
        found = False
        for content in classified_content:
            if sentence.lower() in content.lower() or content.lower() in sentence.lower():
                found = True
                break
        if not found:
            missing_sentences.append(sentence)
    
    return missing_sentences

def classify_missing_sentence(sentence, left_context1=None, left_context2=None, right_context1=None, right_context2=None):
    prompt = f"""
You are now an expert in prompt engineering for large language models. You need to classify the following sentence into one of these components: <Role></Role>, <Directive></Directive>, <Additional Information></Additional Information>, <Output Formatting></Output Formatting>, or <Examples></Examples>.

Context:
{left_context1 if left_context1 else ''}
{left_context2 if left_context2 else ''}
{right_context1 if right_context1 else ''}
{right_context2 if right_context2 else ''}

Sentence to classify:
{sentence}

Please classify this sentence into the most appropriate category. Output only the sentence wrapped in the appropriate tag.
Remember:
- <Role> is for defining the model's role or identity
- <Directive> is for task instructions and input questions
- <Additional Information> is for background information and context
- <Output Formatting> is for output format requirements
- <Examples> is for example inputs and outputs
"""
    return prompt

count = 0
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing requests"):
    eff_context = []
    eff_context_output = []
    
    context_columns = [col for col in row.index if col.startswith('contextblock_')]
    for col in context_columns:
        if not pd.isnull(row[col]):
            eff_context.append(row[col])
    
    sentence_windows = traverse_and_concatenate(eff_context)
    
    for i, (target_sentence, left_context1, left_context2, right_context1, right_context2) in enumerate(sentence_windows):
        prompt = build_sentence_prompt(target_sentence, left_context1, left_context2, right_context1, right_context2)
        response = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        eff_context_output.append(answer)
    
    eff_context_output_str = ' '.join(eff_context_output)
    processed_output = process_output(eff_context_output_str, is_initial=True)
   
    missing_sentences = check_classification(eff_context, processed_output)
    
    if missing_sentences:
        missing_classifications = []
        for missing_sentence in missing_sentences:
            
            idx = eff_context.index(missing_sentence)
            
            left1 = eff_context[idx-2] if idx > 1 else None
            left2 = eff_context[idx-1] if idx > 0 else None
            right1 = eff_context[idx+1] if idx < len(eff_context)-1 else None
            right2 = eff_context[idx+2] if idx < len(eff_context)-2 else None
            
            prompt = classify_missing_sentence(missing_sentence, left1, left2, right1, right2)
            response = client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            missing_classification = response.choices[0].message.content.strip()
            missing_classifications.append(missing_classification)
        
        processed_output = merge_classifications(processed_output, missing_classifications)
    
    df_1.loc[count, "context_output"] = processed_output
    count += 1

save_to_json(df_1, "")
    




    











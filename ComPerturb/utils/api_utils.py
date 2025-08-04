import time
import logging
import tiktoken
from openai import OpenAI, APIConnectionError, APIError

def truncate_message(messages, model_name="gpt-3.5-turbo", max_tokens=4000):
    """ Ensure OpenAI API request does not exceed max token limit """
    tokenizer = tiktoken.encoding_for_model(model_name)
    
    while True:
        total_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
        
        if total_tokens <= max_tokens:
            break  
        
        # If it exceeds the limit, truncate the longest content
        longest_message = max(messages, key=lambda m: len(tokenizer.encode(m["content"])))
        longest_message["content"] = longest_message["content"][:len(longest_message["content"]) // 2]  # 截断一半

    return messages

def safe_openai_request(client, messages, max_retries=5, delay=2):
    """ Safe OpenAI API request to avoid overlong context """
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # Calculate token length and truncate
    messages = truncate_message(messages, model_name="gpt-3.5-turbo", max_tokens=4090)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="",
                messages=messages,
                temperature=0.2,
                max_tokens=1000,  
            )
            return response.choices[0].message.content.strip()

        except APIConnectionError as e:
            logging.warning(f"API connection failed, attempt {attempt + 1}/{max_retries}. Error: {e}")
            time.sleep(delay)
        except APIError as e:
            logging.error(f"API request failed, error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unknown error occurred: {e}")
            return None

    logging.error(f"API request failed, reached max retries {max_retries}.")
    return None

def init_openai_client(api_key, base_url):
    """ Initialize OpenAI client """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    ) 
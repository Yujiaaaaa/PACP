from openai import OpenAI, APIConnectionError, APIError
import pandas as pd
import logging

class SimilarityCalculator:
    def __init__(self, sim_api_key, sim_base_url):
        """OpenAI client for similarity calculation"""
        self.client = OpenAI(api_key=sim_api_key, base_url=sim_base_url)

    def compare(self, output, answer):
        """Use OpenAI to determine whether the semantics of two pieces of code/text are consistent, return 1 or 0"""
        try:
            output = output if pd.notnull(output) else ""
            answer = answer if pd.notnull(answer) else ""

            if not output.strip() or not answer.strip():
                return 0

            prompt = (
                "You are an intelligent evaluator for both general text and code generation tasks. "
                "Given the following two pieces of content (which may be natural language or code), determine whether they express the same meaning, intent, or answer. "
                "Ignore differences in wording, formatting, style, or programming language. "
                "If you believe they are equivalent, return only '1'; if not, return only '0'. Do not provide any explanation.\n\n"
                f"Content 1:\n{output}\n"
                f"Content 2:\n{answer}\n"
                "Output only '1' or '0':"
            )

            messages = [{"role": "user", "content": prompt}]
            response = self.safe_openai_request(messages)

            if response in ['1', '0']:
                return int(response)
            else:
                logging.error(f"Invalid response from model: {response}")
                return 0

        except Exception as e:
            logging.error(f"Code similarity comparison failed: {str(e)}")
            return 0

    def safe_openai_request(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model="", 
                messages=messages,
                temperature=0
            )
            return completion.choices[0].message.content.strip()
        except (APIConnectionError, APIError) as e:
            logging.error(f"OpenAI API call failed: {str(e)}")
            return '0' 
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerplexityCalculator:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_length = 4096  
    
    def calculate(self, text):
        """
        计算文本的困惑度
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
        input_ids = inputs["input_ids"]
        
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]  
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return math.exp(loss) 
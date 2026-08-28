# imports
from tqdm import tqdm
import csv
import pandas as pd
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# file specific configurations
prompt_base_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts"
zero_shot_prompt_path = Path(prompt_base_path).joinpath("zero_shot_lid_prompt.txt")
one_shot_prompt_path = Path(prompt_base_path).joinpath("one_shot_lid_prompt.txt")
prompts = [zero_shot_prompt_path, one_shot_prompt_path]

# helper functions
def get_prompt(text: str, shot: list) -> str:
    with open(prompts[shot], "r") as f:
        prompt_template = f.read()
    return prompt_template.replace("{text}", text)

class CommertialModels:
    def __init__(self, model_name: str, input_csv_file_path: str, output_csv_file_path: str, max_tokens: int = 64, temperature: float = 0.0):
        self.model_name = model_name
        self.model_path = Path.home().joinpath('models', model_name)
        self.tokenizer = MistralTokenizer.from_file(f"{self.model_path}/tokenizer.model.v3")
        self.model = Transformer.from_folder(self.model_path)
        self.input_csv_file_path = input_csv_file_path
        self.output_csv_file_path = output_csv_file_path
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_response(self, user_input: str) -> str:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_input)])
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], self.model, max_tokens=self.max_tokens, temperature=self.temperature, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result

    def inference_all(self, shot=0) -> str:
        idf = pd.read_csv(self.input_csv_file_path)
        
        # header for output CSV
        with open(self.output_csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sentence", "Response"])

        # iterate and write responses to output CSV
        for row in tqdm(idf.itertuples(index=False), desc=f"running {shot}-shot"):
            with open(self.output_csv_file_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                sentence = row.Sentences
                prompt = get_prompt(sentence, shot)
                response = self.get_response(prompt)
                writer.writerow([sentence, response])
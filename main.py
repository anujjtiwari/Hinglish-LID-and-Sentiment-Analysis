# PROGRAM GATEWAY
import argparse
from lib.helper import add_unique_id
from lib.inference import *

def parse_args(): # parsing all the arguments
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--shot_value", type=int, default=0, help="Number of shots")
    parser.add_argument("--test_data", type=str, required=True, help="Test dataset name")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--function_name", type=str, required=True, help="Function name to call for inference")
    parser.add_argument("--zero_shot_prompt_path", type=str, required=False, help="Path to zero-shot prompt file")
    parser.add_argument("--one_shot_prompt_path", type=str, required=False, help="Path to one-shot prompt file")
    return parser.parse_args()

if __name__ == "__main__": # driving code
    model_to_function = { # mapping
        "mistralai/Mistral-7B-Instruct-v0.3"    : Mistral_7B_Instruct_v0_3,
        "CohereLabs/c4ai-command-a-03-2025"     : c4ai_command_a_03_2025,
        "meta-llama/Llama-3.1-8B-Instruct"      : Llama_3_1_8B_Instruct,
        "CohereLabs/aya-expanse-8b"             : aya_expanse_8b,
        "Qwen/Qwen2.5-7B-Instruct"              : Qwen2_5_7B_Instruct,
        "meta-llama/Llama-3.3-70B-Instruct"     : Llama_3_3_70B_Instruct
    }
    args = parse_args() # config
    add_unique_id(test_data, test_data) # add unique id to test CSV file

    # final execution
    output_file = output_path + "/" + args.model_id.split("/")[-1] + "_" + args.shot_value + "_shot.csv"
    model = model_to_function[args.model_id]()(args.zero_shot_prompt_path, args.one_shot_prompt_path, args.test_data, output_file, max_seq_length=args.max_seq_length, temperature=args.temperature)
    response = model.get_response(args.shot_value, "I love watching football") # get response based on shot value
    print(f"[+] Response: {response}")


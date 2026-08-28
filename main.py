# PROGRAM GATEWAY
from lib.inference import CommertialModels

# driving code
if __name__ == "__main__":
    model_name = "mistralai-7B-instruct-v0.3"
    train_csv_file_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/comi_lingua_dataset/train.csv"
    test_csv_file_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/comi_lingua_dataset/test.csv"
    output_zero_csv_file_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/results/mistral_zero_shot.csv"
    output_one_csv_file_path = "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/results/mistral_one_shot.csv"
    max_tokens = 2048
    temperature = 0.0

    # zero shot
    model0 = CommertialModels(model_name, input_csv_file_path=test_csv_file_path, output_csv_file_path=output_zero_csv_file_path, max_tokens=max_tokens, temperature=temperature)
    model0.inference_all(shot=0)

    # one shot
    model1 = CommertialModels(model_name, input_csv_file_path=test_csv_file_path, output_csv_file_path=output_one_csv_file_path, max_tokens=max_tokens, temperature=temperature)
    model1.inference_all(shot=1)
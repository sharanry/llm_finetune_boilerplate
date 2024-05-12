# MODEL = "microsoft/Phi-3-mini-4k-instruct"
# PROMPT_FORMAT = "<|user|>\nPROMPT <|end|>\n<|assistant|>" #ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#chat-format

MODEL = "keeeeenw/MicroLlama"
PROMPT_FORMAT = "Question: PROMPT Answer: " #ref: https://github.com/keeeeenw/MicroLlama?tab=readme-ov-file#want-to-try-it-out
DATASET = ".data/vegalite_dataset"



TARGET_MODULES = ["q_proj", "v_proj"]
LORA_R = 256 # 512
LORA_ALPHA = 512 # 1024
LORA_DROPOUT = 0.05
EPOCHS = 3
LEARNING_RATE = 1e-4  


LORA_SAVE_FOLDER_NAME = ".saves/test_lora"

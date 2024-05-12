from config import MODEL, PROMPT_FORMAT
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def generate_text(prompt, model, tokenizer):
    text_generator = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer
    )

    formatted_prompt = PROMPT_FORMAT.replace("PROMPT", prompt)
    print("formatted_prompt", formatted_prompt)

    sequences = text_generator(
        formatted_prompt,
        do_sample=True,
        top_k=5,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_new_tokens=128,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# Generate
# generate_text("What are you?", model, tokenizer)

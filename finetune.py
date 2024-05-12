from config import LORA_R, LORA_ALPHA, LORA_DROPOUT, EPOCHS, LEARNING_RATE, LORA_SAVE_FOLDER_NAME, TARGET_MODULES
from dataset import dataset
from model import model, tokenizer, generate_text

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, default_data_collator

import bitsandbytes


# Define LoRA Config
lora_config = LoraConfig(
                 r = LORA_R, # the dimension of the low-rank matrices
                 lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
                 lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
                 bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=TARGET_MODULES,
                )
print(model)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# define the training arguments first.
training_args = TrainingArguments(
                    output_dir=LORA_SAVE_FOLDER_NAME,
                    overwrite_output_dir=True,
                #     fp16=True, #converts to float precision 16 using bitsandbytes
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    learning_rate=LEARNING_RATE,
                    num_train_epochs=EPOCHS,
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
tokenizer.pad_token = tokenizer.eos_token
# training the model 
trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
# only saves the incremental 🤗 PEFT weights (adapter_model.bin) that were trained, meaning it is super efficient to store, transfer, and load.
trainer.model.save_pretrained(LORA_SAVE_FOLDER_NAME)


# # save the full model and the training arguments
# trainer.save_model(LORA_SAVE_FOLDER_NAME)
# trainer.model.config.save_pretrained(LORA_SAVE_FOLDER_NAME)





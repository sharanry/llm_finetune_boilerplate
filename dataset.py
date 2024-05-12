import os
from config import DATASET, PROMPT_FORMAT
from model import tokenizer
from torch.utils.data import Dataset, random_split


# Ensure DATASET is a directory path
if os.path.isdir(DATASET):
    dataset_files = [os.path.join(DATASET, file) for file in os.listdir(DATASET) if file.endswith('.txt')]
else:
    raise ValueError(f"Provided DATASET path '{DATASET}' is not a directory.")

sections = [
 "### PLOT STYLE: ",
 "### DESCRIPTION: ",
 "### VEGA LITE JSON: ",
]
raw_dataset = []
for file_path in dataset_files:
    with open(file_path, 'r') as file:
        content = file.read()
        data_dict = {}
        for section in sections:
            start = content.find(section) + len(section)
            end = content.find("###", start)
            content = content.replace("View this example in the online editor", "")
            data_dict[section.strip(': ').strip("### ")] = content[start:end].strip() if end != -1 else content[start:].strip()
            
        raw_dataset.append(data_dict)

qa_dataset = []
for datapoint in raw_dataset:
    description = datapoint.get("DESCRIPTION", "")
    plot_style = datapoint.get("PLOT STYLE", "")
    question = f"{plot_style}\n{description}"
    answer = datapoint["VEGA LITE JSON"]
    qa_dataset.append({"question": question, "answer": answer})

# qa_dataset[0]

dataset = []
for datapoint in qa_dataset:
    inputs_ids = tokenizer.encode(PROMPT_FORMAT.replace("PROMPT", datapoint["question"]+"\n\n"), return_tensors="pt")[0]
    # print("inputs_ids:", inputs_ids)
    dataset.append({
        # "input": PROMPT_FORMAT.replace("PROMPT", datapoint["question"]+"\n\n"),
        # "target": datapoint["answer"],
        "input_ids": inputs_ids,
        "labels": tokenizer.encode(datapoint["answer"], return_tensors="pt")[0]
        })
    
# dataset[0]

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Instantiate the QADataset
qa_dataset_instance = QADataset(dataset)
len(qa_dataset_instance)



train_dataset, val_dataset = random_split(qa_dataset_instance, [160, 25])
dataset = {
    "train": train_dataset,
    "val": val_dataset
}





# Imports
from typing import Tuple
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class TextClassificationData:
    def __init__(self, dataset_name, model_name, batch_size=32, max_length=128) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_function(self, examples: dict) -> dict:
        return self.tokenizer(examples['text'], max_length=self.max_length,
                                padding='max_length',truncation=True)

    def get_dataset(self) -> Tuple[DatasetDict, list]:
        dataset = load_dataset(self.dataset_name)
        label_names = dataset['train'].features['label'].names
        return dataset, label_names

    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, list]:
        dataset, label_names = self.get_dataset()
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True,
                                         remove_columns=['text'])
        tokenized_datasets.set_format(type = 'torch',
                                  columns = ['input_ids', 'attention_mask', 'label'])
        train_loader = DataLoader(dataset=tokenized_datasets['train'],
                                batch_size=self.batch_size, shuffle=True)

        test_loader = DataLoader(dataset=tokenized_datasets['test'],
                              batch_size=self.batch_size)

        return train_loader, test_loader, label_names

if __name__ == "__main__":
    ag_news_data = TextClassificationData(dataset_name="ag_news", model_name='bert-base-uncased')
    train_loader, test_loader, label_names = ag_news_data.get_dataloader()
    print(f"Label names: {label_names}")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")

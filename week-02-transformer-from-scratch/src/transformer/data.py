# Imports
from typing import Tuple

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class TextClassificationData:
    """
    Data handling class for text classification task using Hugging Face datasets, tokenizers, and PyTorch DataLoader.
    This class loads a specified dataset and prepares it for training, validation, and testing with a PyTorch DataLoader.
    Args:
        dataset_name (str): name of the Hugging Face dataset to load.
        model_name (str): name of the Hugging Face model to use for tokenization.
        batch_size (int): batch size for DataLoader.
        max_length (int): maximum sequence length for tokenization.
        seed (int): random seed for reproducibility.
    """
    def __init__(self, dataset_name, model_name, batch_size=32, max_length=128, seed=42) -> None:
        """
        Initialize the TextClassificationData class.
            dataset_name (str: name of the Hugging Face dataset to load.
            model_name (str): name of the Hugging Face model to use for tokenization.
            batch_size (int, optional): batch size for dataloader. Defaults to 32.
            max_length (int, optional): context length (sequence length). Defaults to 128.
            seed (int, optional): random seed for reproducibility. Defaults to 42.
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_function(self, examples: dict) -> dict:
        """
        Tokenize the input text using the specified tokenizer.
        """
        return self.tokenizer(examples['text'], max_length=self.max_length,
                                padding='max_length',truncation=True)

    def get_dataset(self) -> Tuple[DatasetDict, list]:
        """
        Downloads dataset and returns dataset, label names
        """
        dataset = load_dataset(self.dataset_name)
        label_names = dataset['train'].features['label'].names
        return dataset, label_names

    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
        """
        Prepares and returns DataLoaders for training, validation, and testing, along with label names.
        """
        dataset, label_names = self.get_dataset()
        # Tokenize dataset
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True,
                                         remove_columns=['text'])
        tokenized_datasets.set_format(type = 'torch',
                                  columns = ['input_ids', 'attention_mask', 'label'])
        # Train-val split
        split = tokenized_datasets['train'].train_test_split(test_size=0.15, seed=self.seed)
        train_dataset = split['train']
        val_dataset = split['test']
        # DataLoader
        g = torch.Generator()   #for deterministic DataLoader shuffling
        g.manual_seed(self.seed)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self.batch_size, shuffle=True, generator=g, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                              batch_size=self.batch_size, generator=g, shuffle=False)
        test_loader = DataLoader(dataset=tokenized_datasets['test'],
                              batch_size=self.batch_size, generator=g, shuffle=False)

        return train_loader, val_loader, test_loader, label_names

if __name__ == "__main__":
    ag_news_data = TextClassificationData(dataset_name="ag_news", model_name='bert-base-uncased')
    train_loader, val_loader, test_loader, label_names = ag_news_data.get_dataloader()
    print(f"Label names: {label_names}")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in val loader: {len(val_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")

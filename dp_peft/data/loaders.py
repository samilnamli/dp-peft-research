from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from typing import Tuple, Optional
import torch


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_text_dataloaders(
    dataset_name: str,
    tokenizer_name: str,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if dataset_name == 'agnews':
        dataset = load_dataset('ag_news')
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        text_column = 'text'
        label_column = 'label'
    elif dataset_name == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
        text_column = 'sentence'
        label_column = 'label'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_texts = list(train_dataset[text_column])
    train_labels = list(train_dataset[label_column])
    test_texts = list(test_dataset[text_column])
    test_labels = list(test_dataset[label_column])
    
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_vision_dataloaders(
    dataset_name: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'cifar10':
        dataset = load_dataset('cifar10')
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = load_dataset('cifar100')
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def transform_train(examples):
        examples['pixel_values'] = [train_transform(img.convert('RGB')) for img in examples['img']]
        return examples
    
    def transform_test(examples):
        examples['pixel_values'] = [test_transform(img.convert('RGB')) for img in examples['img']]
        return examples
    
    train_dataset = dataset['train'].with_transform(transform_train)
    test_dataset = dataset['test'].with_transform(transform_test)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([example['label'] for example in examples])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader

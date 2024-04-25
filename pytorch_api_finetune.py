import torch
from tqdm import tqdm
from datasets import load_from_disk
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1、加载模型和预训练权重
checkpoint = r"E:\gitlocal\huggingface\google_bert\bert_base_uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2、加载数据集
dataset_path = r"F:\pythonProject\datasets\glue\mrpc"
raw_datasets = load_from_disk(dataset_path)


# 4、数据集预处理
def tokenize_function(sample):
    # 这里可以添加多种操作，不光是tokenize
    # 这个函数处理的对象，就是Dataset这种数据类型，通过features中的字段来选择要处理的数据
    return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 4.1 动态padding，collator数据
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5、pytorch的Dataloader数据准备
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

# 5.1通过这里的dataloader，每个batch的seq_len可能不同
train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)

# 6、定义 optimizer 和 learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)  # num of batches * num of epochs
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,  # scheduler是针对optimizer的lr的
    num_warmup_steps=0,
    num_training_steps=num_training_steps)

# 7、Pytorch training loops
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        # 要在GPU上训练，需要把数据集都移动到GPU上：
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# 8、Evaluation
metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():  # evaluation的时候不需要算梯度
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # 由于dataloader是每次输出一个batch，因此我们要等着把所有batch都添加进来，再进行计算
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

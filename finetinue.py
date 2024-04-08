# # 1、下载数据集并准备进行训练。首先加载 Yelp 评论数据集
# from datasets import load_dataset
#
# dataset = load_dataset("yelp_review_full")
# dataset["train"][100]  # 查看数据集格式
#
# # 2、调用分词处理器处理文本
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
#
#
# # 构建对整个数据集应用预处理的方法
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
# # 使用 Datasets map 方法一步处理数据集
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# # 创建子集进行微调以减少所需的时间
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#
# # 3、加载模型并指定预期标签的数量，分析Yelp Review 数据集有5个标签
# from transformers import AutoModelForSequenceClassification
#
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
#
# # 4、指定训练超参数，创建TrainingArguments 类
# """
# output_dir ( str ) — 将写入模型预测和检查点的输出目录。
# overwrite_output_dir ( bool ，可选，默认为 False ) — 如果是 True ，则覆盖输出目录output_dir的内容。如果 output_dir 指向检查点目录，则使用它继续训练。
# evaluation_strategy ( str 或 IntervalStrategy，可选，默认为 "no" ) — 训练期间采用的评估策略。可能的值为：
#         "no" ：训练期间不进行评估。
#         "steps" ：每隔 eval_steps 进行一次评估（并记录）。
#         "epoch" ：评估在每个时期结束时进行。
# Prediction_loss_only ( bool ，可选，默认为 False ) — 执行评估和生成预测时，仅返回损失。
# per_device_train_batch_size ( int ，可选，默认为 8) — 用于训练的每个 GPU/XPU/TPU/MPS/NPU 核心/CPU 的批量大小。
# per_device_eval_batch_size ( int ，可选，默认为 8) — 用于评估的每个 GPU/XPU/TPU/MPS/NPU 核心/CPU 的批量大小。
# gradient_accumulation_steps ( int ，可选，默认为 1) — 在执行向后/更新传递之前累积梯度的更新步骤数。
#     使用梯度累加时，一步算作向后传递的一步。因此，每个 gradient_accumulation_steps * xxx_step 个训练样例都会进行记录、评估、保存。
# eval_accumulation_steps ( int ，可选) — 在将结果移至 CPU 之前累积输出张量的预测步骤数。如果未设置，整个预测将在转移到 CPU 之前累积在 GPU/NPU/TPU 上（速度更快，但需要更多内存）。
# eval_delay ( float ，可选) — 在执行第一次评估之前要等待的epoch数或步骤数，具体取决于评估策略。
# Learning_rate ( float ，可选，默认为 5e-5) — AdamW 优化器的初始学习率。
# weight_decay ( float ，可选，默认为 0) — 应用于除 AdamW 优化器中的所有偏差和 LayerNorm 权重之外的所有层的权重衰减（如果不为零）。
# adam_beta1 ( float ，可选，默认为 0.9) — AdamW 优化器的 beta1 超参数。
# adam_beta2 ( float ，可选，默认为 0.999) — AdamW 优化器的 beta2 超参数。
# adam_epsilon ( float ，可选，默认为 1e-8) — AdamW 优化器的 epsilon 超参数。
# num_train_epochs( float ，可选，默认为 3.0) — 要执行的训练时期总数（如果不是整数，将执行停止训练之前最后一个时期的小数部分百分比）。
# max_steps ( int ，可选，默认为 -1) — 如果设置为正数，则表示要执行的训练步骤总数。覆盖 num_train_epochs 。对于有限数据集，通过数据集重复训练（如果所有数据都已耗尽），直到达到 max_steps 。
# lr_scheduler_type ( str 或 SchedulerType，可选，默认为 "linear" ) — 要使用的调度程序类型。有关所有可能的值，请参阅 SchedulerType 的文档。
# lr_scheduler_kwargs (‘dict’，可选，默认为 {}) — lr_scheduler 的额外参数。有关可能的值，请参阅每个调度程序的文档。
# Warmup_ratio ( float ，可选，默认为 0.0) — 用于从 0 到 learning_rate 的线性预热的总训练步骤的比率。
# Warmup_steps ( int ，可选，默认为 0) — 用于从 0 到 learning_rate 的线性预热的步数。覆盖 warmup_ratio 的任何效果。
# save_strategy ( str 或 IntervalStrategy，可选，默认为 "steps" ) — 训练期间采用的检查点保存策略。可能的值为：
#         "no" ：训练期间不进行保存。
#         "epoch" ：保存在每个纪元结束时完成。
#         "steps" ：每隔 save_steps 进行一次保存。
# save_steps（ int 或 float ，可选，默认为 500） - 如果 save_strategy="steps" ，则两个检查点保存之前的更新步骤数。应该是 [0,1) 范围内的整数或浮点数。如果小于 1，将被解释为总训练步数的比率。
# save_total_limit ( int , 可选) — 如果传递一个值，将限制检查点的总数。删除 output_dir 中较旧的检查点。启用 load_best_model_at_end 后，除了最近的检查点之外，将始终保留根据 metric_for_best_model 的“最佳”检查点。例如，对于 save_total_limit=5 和 load_best_model_at_end ，最后四个检查点将始终与最佳模型一起保留。当 save_total_limit=1 和 load_best_model_at_end 时，可能会保存两个检查点：最后一个和最好的一个（如果它们不同）。
# """
# from transformers import TrainingArguments
#
# # 如果想在微调期间监控评估指标，在训练参数中指定 evaluation_strategy 参数，以在每个周期结束时报告评估指标
# training_args = TrainingArguments(output_dir="test_trainer",
#                                   evaluation_strategy="epoch",
#                                   num_train_epochs=2,
#                                   per_device_train_batch_size=8,
#                                   per_device_eval_batch_size=8)
#
# # 5、加载 Evaluate 评价指标
# """
# Trainer 在训练期间不会自动评估模型性能。需要向 Trainer 传递一个函数来计算和报告指标。
# Evaluate 库提供了一个简单的 accuracy 函数，使用 evaluate.load 函数加载
# """
# import numpy as np
# import evaluate
#
# metric = evaluate.load("accuracy")
#
#
# # metric 调用 compute 以计算预测的准确性
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     # 递给 compute 之前，将logits值转换为预测
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# # 6、使用模型、训练参数、训练和测试数据集以及评估函数创建一个 Trainer 对象：
# """
# model：这是你要训练、评估或预测的模型。
# 可以是Hugging Face库中提供的预训练模型（PreTrainedModel），也可以是你自己定义的PyTorch模型（torch.nn.Module）。
# 如果你使用自定义的PyTorch模型，确保它们与Hugging Face库中的模型以相同的方式工作，以便与Trainer类兼容。
#
# args：这是用于调整训练的参数。
# 如果不提供，将使用默认的一组参数。
# 你可以通过创建TrainingArguments的实例来调整训练参数，如设置训练的最大步数、学习率等。
#
# data_collator：这是用于形成批次数据的函数。
# 如果不提供，Trainer将根据是否提供了分词器（tokenizer）来选择默认的数据整合器。
# 数据整合器的作用是将训练或评估数据集的元素整合成批次。
#
# train_dataset：这是用于训练的数据集。
# 可以是PyTorch的Dataset对象或IterableDataset对象。
#
# eval_dataset：这是用于评估的数据集。
# 可以是PyTorch的Dataset对象或字典，其中键是用于标识数据集的名称。
#
# tokenizer：这是用于数据预处理的分词器。
# 如果提供，将用于自动对输入进行填充，并在模型中保存，以便重新运行中断的训练或重用微调后的模型。
#
# model_init：这是用于实例化模型的函数。
# 如果提供了这个函数，每次调用train()方法时都会从此函数返回的模型实例开始训练。
# 这个函数可以没有参数，也可以接受一个参数，例如用于超参数优化的trial对象。
#
# compute_metrics：这是用于评估时计算指标的函数。
# 它接收一个EvalPrediction对象作为参数，并返回一个包含指标名称和值的字典。
#
# callbacks：这是用于自定义训练循环的回调函数列表。
# 可以通过添加回调函数来扩展或修改Trainer的默认行为。
#
# optimizers：这是优化器和学习率调度器的元组。
# 默认情况下，Trainer将使用AdamW优化器和一个由参数args控制的学习率调度器。
#
# preprocess_logits_for_metrics：这是用于在缓存每个评估步骤时预处理对数的函数。
# 它接收模型的输出对数和标签作为输入，并返回预处理后的对数。
# 这个函数可以用来在计算指标之前对模型输出进行后处理。
# """
# from transformers import Trainer
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
#
# # 7、然后通过调用 train() 微调模型：
# trainer.train()

def transform_ssh_to_scp(ssh_command, local_path):
    # 分割ssh命令，提取目标路径
    parts = ssh_command.split()
    remote_path = parts[-1]

    # 构建scp命令
    scp_command = f"scp -rP {parts[2]} {parts[3]}:{remote_path} {local_path}"

    return scp_command

ssh_command = "ssh -p 14835 root@connect.westb.seetacloud.com /实例中/某个文件或文件夹路径"
local_path = "<本地文件/文件夹>路径"

scp_command = transform_ssh_to_scp(ssh_command, local_path)
print(scp_command)


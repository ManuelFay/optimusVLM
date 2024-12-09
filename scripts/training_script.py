from transformers import AutoProcessor
import torch
from datasets import load_dataset, VerificationMode
from transformers import TrainingArguments, Trainer


from optimus_vlm.models.optimus import *
from optimus_vlm.models.optimusVLM.modeling_optimusvlm import OptimusVLMForMLM

processor = AutoProcessor.from_pretrained("manu/optimus-vlm")
model2 = OptimusVLMForMLM.from_pretrained("manu/optimus-vlm-init",
                                          device_map="cuda",
                                          torch_dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2")

model2.lm_head.weight =  model2.model.text_model.embed_tokens.weight
# freeze image encoder
for param in model2.model.vision_model.parameters():
    param.requires_grad = False


# ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
ds = load_dataset('nz/arxiv-ocr-v0.2', data_files=["data/train-00000-of-00446.parquet",
                                                   "data/train-00001-of-00446.parquet",
                                                   "data/train-00002-of-00446.parquet",
                                                   "data/train-00003-of-00446.parquet",
                                                   "data/train-00004-of-00446.parquet"], verification_mode=VerificationMode.NO_CHECKS)

split_ds = ds["train"].train_test_split(test_size=0.1)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")]


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # question = example["question"]
        # answer = example["multiple_choice_answer"]
        text = example["text"][:500]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR the image."},
                    {"type": "image"},
                    # {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    # {"type": "text", "text": answer}
                    {"type": "text", "text": text}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        # print(text)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100

    # print(texts)
    if True:
        mlm_probability = 0.5
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = None
        if special_tokens_mask is None:
            special_tokens_mask = [
                processor.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[:, :1545] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        inputs = batch["input_ids"]
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(processor.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        batch["input_ids"] = inputs

    batch["labels"] = labels
    # breakpoint()

    return batch


model_name = "manu-optimus-vlm-trained"

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="adamw_hf", # for 8-bit, keep this, else adamw_hf
    bf16=True, #underlying precision for 8bit
    output_dir=f"./{model_name}-vqav2",
    hub_model_id=f"{model_name}-vqav2",
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model2,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

trainer.train()


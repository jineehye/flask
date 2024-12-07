import os
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train_blenderbot():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    # 상대 경로를 절대 경로로 변환
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리
    file_path = os.path.join(script_dir, "conversation_logs", "training_data.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found at {file_path}")

    # Load dataset for fine-tuning
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # MLM=False since this is seq2seq fine-tuning
    )

    training_args = TrainingArguments(
    output_dir="blenderbot-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=50,
    report_to="none"  # W&B 비활성화
)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Training the model...")
    trainer.train()

    print("Saving the fine-tuned model...")
    model.save_pretrained("blenderbot-finetuned")
    tokenizer.save_pretrained("blenderbot-finetuned")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_blenderbot()
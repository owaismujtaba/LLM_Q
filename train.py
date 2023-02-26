
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelWithLMHead
import config
from data_utils import data_loader
import pdb

def trainer():
    
    print(" *********************** Training Mode *************************")
    
    training_args = TrainingArguments(
                        output_dir = config.TRAINED_MODEL_PATH,
                        num_train_epochs=config.EPOCHS,
                        per_device_train_batch_size=config.BATCH_SIZE,
                        per_device_eval_batch_size=config.BATCH_SIZE,
                        warmup_steps=config.STEPS,
                        logging_dir=config.LOGS,
                        logging_steps=config.LOGGING_STEPS
                    )

    
    
    dataset = data_loader()
    #pdb.set_trace()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    
    
    trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False),
                    train_dataset=dataset,
                )
    
    
    trainer.train()
    
    model.save_pretrained(config.TRAINED_MODEL_PATH+'modelv1')
    print(" *********************** Training Done *************************")
    

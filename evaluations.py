from transformers import AutoTokenizer, AutoModelWithLMHead
import config

    
    
def evaluate_model():
    
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelWithLMHead.from_pretrained(config.TRAINED_MODEL_PATH, from_tf=True)

    model.eval()
    
    
    while True:
        
        prompt = input('Enter a prompt (type "exit" to quit): ')
    
        # Check if user wants to exit
        if prompt.lower() == 'exit':
            break

        # Generate text based on the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = tokenizer.eos_token_id
        output = model.generate(input_ids, max_length=50,attention_mask=attention_mask, pad_token_id=pad_token_id, do_sample=True, top_p=0.95, top_k=50)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)
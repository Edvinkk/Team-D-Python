import re
import torch
from google.colab import drive, files
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from transformers import pipeline

#Mount Google Drive to save the model
drive.mount('/content/drive', force_remount=True)

#Load and clean the text data
uploaded = files.upload()  #cleanedData.txt'
with open('cleanedData.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

#clean text further as it was still not perfect
def clean_text(text):
    #Replace multiple spaces with a single space and trim
    text = re.sub(r'\s+', ' ', text).strip()
    #Remove specific patterns from scraped data
    text = re.sub(r'-- -- --.*$', '', text)
    #Remove Wikipedia references like [1]
    text = re.sub(r'\[\d+\]', '', text)
    #Remove Wikipedia section headers like ==History==
    text = re.sub(r'==.*?==', '', text)
    #Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    #Remove non-ASCII characters more thoroughly
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    #Remove extra punctuation or special characters
    text = re.sub(r'[^a-zA-Z0-9\s.,;?!-]', ' ', text)
    return text

cleaned_text = clean_text(text_data)
paragraphs = [p for p in cleaned_text.split('. ') if len(p) > 10 and len(p.split()) < 100 and p.strip()]

#Add synthetic examples to reinforce associations
synthetic_examples = [
    ##MDMA examples
    "MDMA is commonly known as ecstasy. It is a popular drug in party settings.",
    "MDMA is also called molly. It is often used at music festivals.",
    "MDMA is a stimulant drug, not a drink like purple drank.",
    "MDMA is not related to lean or purple drank. It is known as ecstasy.",
    "MDMA does not contain codeine or promethazine. It is a stimulant drug.",
    "MDMA is not related to cocaine. It is a stimulant drug called ecstasy.",
    #Cocaine examples
    "Cocaine is often referred to as coke. It is also known as crack in some forms.",
    "Cocaine is a stimulant drug often called coke in street slang.",
    "Cocaine is not a drink like lean. It is a stimulant drug called coke.",
    "Cocaine is called coke, not cock. It is a stimulant drug.",
    "Cocaine is not related to MDMA. It is known as coke or crack.",
    #Lean examples
    "Purple drank is a drink made with codeine. Purple drank, also known as lean, contains codeine and promethazine.",
    "Lean is not made with coke; it is made with codeine.",
    "Lean is another name for purple drank, a drink made with codeine.",
    "Lean is not related to MDMA or cocaine. It contains codeine and promethazine.",
    "Lean is a drink, not a stimulant drug like MDMA or cocaine.",
    #Dialogue-style examples for drug questions
    "User: What is MDMA? Bot: MDMA is commonly known as ecstasy. It is a popular drug in party settings.",
    "User: What is MDMA also called? Bot: MDMA is also called molly. It is often used at music festivals.",
    "User: Is MDMA a drink? Bot: No, MDMA is not a drink. It is a stimulant drug known as ecstasy.",
    "User: Does MDMA contain codeine? Bot: No, MDMA does not contain codeine. It is a stimulant drug.",
    "User: What is cocaine? Bot: Cocaine is often referred to as coke. It is also known as crack in some forms.",
    "User: What is cocaine called? Bot: Cocaine is called coke, not cock. It is a stimulant drug.",
    "User: Is cocaine a drink? Bot: No, cocaine is not a drink. It is a stimulant drug called coke.",
    "User: What is lean made with? Bot: Lean, also known as purple drank, is a drink made with codeine and promethazine.",
    "User: Is lean related to MDMA? Bot: No, lean is not related to MDMA. Lean is a drink made with codeine.",
    #Dialogue-style examples for greetings
    "User: Hi Bot: Hello! How can I help you today?",
    "User: Hello Bot: Hi there! What would you like to know?",
    "User: Hey Bot: Hey! How can I assist you?",
] * 40  #Keep 40 repetitions to reinforce associations
paragraphs.extend(synthetic_examples)

#reate a Hugging Face Dataset
data_dict = {"text": paragraphs}
dataset = Dataset.from_dict(data_dict)

#Load the tokenizer and tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token  #Set padding token to EOS token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

#Create a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

#Load the DistilGPT-2 model
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.config.pad_token_id = tokenizer.eos_token_id   #Ensure the model uses the same padding token

#Define training arguments
training_args = TrainingArguments(
    output_dir='./distilgpt2_results',
    num_train_epochs=7,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.05,
    logging_dir='./distilgpt2_logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    fp16=True,
)

#Initialize the Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]  #Adjust early stopping
)

trainer.train()

#Save the fine-tuned model and tokenizer to Google Drive
model.save_pretrained('/content/drive/MyDrive/fine_tuned_distilgpt2')
tokenizer.save_pretrained('/content/drive/MyDrive/fine_tuned_distilgpt2')


generator = pipeline("text-generation", model="/content/drive/MyDrive/fine_tuned_distilgpt2", tokenizer=tokenizer)
print(generator("What is MDMA?", max_length=50))

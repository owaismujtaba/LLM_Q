import os
import PyPDF2
import re
import config
import string
import re
from nltk.corpus import stopwords
import nltk
from transformers import TextDataset, GPT2Tokenizer
import pdb
def extract_write_text():
    
    
    print(" *************** Extracting and writing Data ***************")
    text_list = []

    for filename in os.listdir(config.DATA):
        #pdb.set_trace()
        if filename.endswith('.pdf'):
            with open(os.path.join(config.DATA, filename), 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_list.append(text)
    
    

    with open(config.PROCESSED_DATA +'output.txt', 'w') as f:
        for item in text_list:
            f.write("%s\n" % item)

                    
    print(" *************** Extracting Data Done ***************")
                    
    return text_list


def clean_data():
    
    print(" *************** Cleaning Extracted Data  ***************")

    #nltk.download('stopwords')
    stop_words = stopwords.words('english')
    #pdb.set_trace()
    with open(config.PROCESSED_DATA +'output.txt', 'r') as f:
        text = f.read()
    f.close()
    text = text.lower()
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])
    text =  text.encode('ascii', 'ignore').decode()
    text = re.sub("@\S+", " ", text)
    text = re.sub("https*\S+", " ", text)
    text = re.sub("#\S+", " ", text)
    text = re.sub("\'\w+", '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    #text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub('\s{2,}', " ", text)
    #text = text.split(" ")
    file = open(config.PROCESSED_DATA +'output_cleaned.txt', 'w')
    file.write(text)
    f.close()
    print(" *************** Cleaning Done  ***************")
    #pdb.set_trace()
    return text


def data_loader():
    
    
    file = config.PROCESSED_FILE
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')      
    dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=file,
                block_size=128
                )
    
    
    
    return dataset



import os
from transformers import AutoTokenizer
from datasets import Dataset

#Importing from Hugging Face the tokenizer used for the XLM-RoBERTa model
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def process_file_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    #Divide the text into paragraphs using the newline character.
    #This is helpful because transformers have a maximum number of tokens they
    #can read at once, so dividing the text into paragraphs avoids cutting off pieces of text
    paragraphs = raw_text.split('\n')
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for para in paragraphs:
        if not para.strip(): #skip empty rows
            continue
            
        #Divide the text in chunks using <EOS> as a separator
        chunks = para.split("<EOS>")
        clean_text = ""
        eos_positions = set()
        
        #For loop used to save the position of the last characters before <EOS>
        for i, chunk in enumerate(chunks):
            clean_text += chunk
            if i < len(chunks) - 1:
                eos_positions.add(len(clean_text) - 1)
        
        #Tokenization of the clean paragraph
        tokenized = tokenizer(
            clean_text,
            truncation=True,
            max_length=512, #If a paragraph contains more than 128 tokens, it will be cut off
            return_offsets_mapping=True 
            #returns a map that, for each token, indicates the corresponding position in the clean text in the
            #form a couple (start, end)
            #start: index of the character in the original text where this token begins.
            #end: index of the character in the original text where this token ends.
            #es "Ciao prof."
            #Tokens generated: [ "<s>", "Ciao", " prof", ".", "</s>" ]
            #Offset mapping: [ (0,0), (0, 4), (4, 9), (9, 10), (0,0) ]
            #The beginning and the end of a sequence (which is not necessarily a sentence!) are marked with
            #the special characters <s> and </s> respectively and in the offset mapping they're both marked with
            #(0,0), so start and end for this characters are both 0
        )
        
        labels = []
        for start, end in tokenized["offset_mapping"]:
            if start == end == 0:
                labels.append(-100) #labeled with -100 so that the model will ignore them during training
                continue
            
            #Checking if this token contains an <EOS>
            is_eos = False
            for pos in range(start, end + 1):
                if pos in eos_positions:
                    is_eos = True
                    break
            
            if is_eos:
                labels.append(1) #end of sentence
            else:
                labels.append(0) #not end of a sentence

        #Paragraph checked, now update the "global" lists with the info of this spefici paragraph        
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        all_labels.append(labels)
        
    #creating a dataset in the form of a Dataset object from the Hugging Face library
    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    })
    
    return dataset
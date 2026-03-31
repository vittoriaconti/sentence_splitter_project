import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "../models/final_model"
BASE_MODEL = "xlm-roberta-base"

print("Loading the trained model and the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

#Executes the code on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model in execution on: {device.type.upper()}")

def get_safe_chunks(text, max_chars=1400):
    chunks = []
    while len(text) > max_chars:
        split_point = -1 #will memorize the index of the cut and act as a boolean
        for punct in ['. ', '? ', '! ', '.\n', '?\n', '!\n', '." ', '."\n']:
            pos = text.rfind(punct, 0, max_chars) #returns the last occurence of the punctuation mark
            if pos > split_point:
                split_point = pos
        
        #Emergency case: if there's no ideal cut point (in correspondence to punctuation marks) it
        #will cut where there's a space. If there are no spaces, it will cut at the end of thw chunk
        if split_point == -1:
            pos = text.rfind(' ', 0, max_chars)
            cut_idx = pos if pos != -1 else max_chars
        else:
            #It will cut after the punctuation mark
            cut_idx = split_point + 1
            
        chunks.append(text[:cut_idx])
        text = text[cut_idx:] #Removes the part just processed and saved in the chunks list from the original text
        
    if text:
        chunks.append(text)
    return chunks


def split_text_into_sentences(text):
    if not text.strip():
        return []
        
    all_final_sentences = []
    
    #Divide the text into chunks
    text_blocks = get_safe_chunks(text, max_chars=1400)
    
    #Process each chunk
    for chunk in text_blocks:
        if not chunk.strip():
            continue
            
        inputs = tokenizer(
            chunk, 
            return_tensors="pt", #Tells Hugging Face to return data in PyTorch format
            truncation=True, 
            max_length=512, 
            return_offsets_mapping=True
        )
        
        #Removes the offset_mapping from the model input and convert it to a NumPy array
        #(array of 2-dimensional arrays)
        offsets = inputs.pop("offset_mapping")[0].numpy()
        #Moves data to the device where the model is executed on
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        #Disables the gradient calculation to save memory and fasten the execution
        with torch.no_grad():
            outputs = model(**inputs)
            #extract logits, the raw numbers that represent
            #how much the model thinks the token belongs to class 0 or class 1
            #logits are 3-dimensional tensors with this structure: (Batch_size, Number_of_tokens, Hidden_size)
            #xml-RoBERTa-base returns (1, 512, 768) because the text is processed one batch at a time, each
            #chunk has a maximum of 512 tokens and for each token RoBERTa returns a vector of 768 numbers (hidden state)
            #Then AutoModelForTokenClassification converts the hidden state into a 2-dimensional vector, so the
            #final tensor is (1, 512, 2)
            logits = outputs.logits
            #For each token, selects the class with the highest score, moves the token
            #to the CPU and convert it to a NumPy array
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        raw_sentences = []
        last_split_idx = 0
        in_eos_block = False
        
        for i, pred in enumerate(predictions):
            offset = offsets[i]
            #Ignore special tokens <s> and </s> 
            if offset[0] == offset[1] == 0:
                continue
                
            if pred == 1:
                in_eos_block = True
            elif pred == 0 and in_eos_block: #the sentence is just ended and a new one is starting -> need to cut
                #offsets[i-1] indicates the previous token, the token that marks the end of the previous sentence
                #offsets[i-1][1] takes the index of the last character of the previous token
                end_char = offsets[i-1][1]
                if end_char > last_split_idx: #to make sure that the new point where the sentence is cut is after the last point of cutting
                    sentence = chunk[last_split_idx:end_char].strip() #actual cut
                    if sentence:
                        raw_sentences.append(sentence)
                    last_split_idx = end_char
                in_eos_block = False

        #Check if there are remaining sentences and eventually add them in the list      
        if last_split_idx < len(chunk):
            remaining = chunk[last_split_idx:].strip()
            if remaining:
                raw_sentences.append(remaining)
                
        all_final_sentences.extend(raw_sentences)
        
    #Check if there are isolated punctuation marks
    final_sentences = []
    isolated_chars = [".", "!", "?", ";", ":", '"', "'", "»", "”", "’", ")", "]", "}"]
    for s in all_final_sentences:
        #If the sentencence is just a punctuation mark and there is a previous sentence it can be attached to
        if s in isolated_chars and final_sentences:
            final_sentences[-1] += s
        else:
            final_sentences.append(s)
            
    return final_sentences
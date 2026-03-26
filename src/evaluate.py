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

def split_text_into_sentences(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", #Tells Hugging Face to return data in PyTorch format
        truncation=True, 
        max_length=512, 
        return_offsets_mapping=True
    )
    
    #Removes the offset_mapping from the model input and convert it to a NumPy array
    offsets = inputs.pop("offset_mapping")[0].numpy()
    
    #Moves data to the device where the model is executed on
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    #Tells PyTorch to not calculate the gradients 
    with torch.no_grad():
        #pass the tokenized text to the model. The model returns two raw numbers that represent
        #how much the model thinks the token belongs to class 0 or class 1
        outputs = model(**inputs)
    
    #For each token, the max between the raw numbers is taken (chooses the class
    #with higher "score"). Then sends the predictions to the CPU so that they can be
    #read with NumPy.
    #NB. The argmax is not the activation function of the classification layer because it's not
    #differentiable. The activation function is the softmax, implicitly applied in the CrossEntropyLoss
    #loss function used as default by Hugging Face. argmax is just a decision function
    #predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    # 1. Calcola le probabilità effettive (da 0 a 1) usando la softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # 2. Imposta una soglia (threshold). 
    # 0.5 è uguale all'argmax. Alzandola (es. 0.6 o 0.7) riduci i tagli sbagliati (alzi la Precision).
    THRESHOLD = 0.6 
    
    # 3. Se la probabilità della classe 1 (B-EOS) supera la soglia, assegna 1, altrimenti 0
    predictions = (probs[:, 1] > THRESHOLD).astype(int)
    
    raw_sentences = [] #will contain all the sentences
    last_split_idx = 0
    in_eos_block = False
    
    for i, pred in enumerate(predictions):
        #Ignore special tokens <s> and </s>
        if offsets[i][0] == 0 and offsets[i][1] == 0:
            continue
            
        if pred == 1:
            in_eos_block = True
        elif pred == 0 and in_eos_block: #the sentence is just ended and a new one is starting -> need to cut
            #offsets[i-1] indicates the previous token, the token that marks the end of the previous sentence
            #offsets[i-1][1] takes the index of the last character of the previous token
            end_char = offsets[i-1][1]
            if end_char > last_split_idx: #to make sure that the new point where the sentence is cut is after the last point of cutting
                sentence = text[last_split_idx:end_char].strip() #actual cut
                if sentence:
                    raw_sentences.append(sentence)
                last_split_idx = end_char
            in_eos_block = False
            
    #Check if there are remaining sentences and eventually add them in the list
    if last_split_idx < len(text):
        remaining = text[last_split_idx:].strip()
        if remaining:
            raw_sentences.append(remaining)
            
    #Check if there are isolated punctuation marks
    final_sentences = []
    for s in raw_sentences:
        #If the sentencence is just a punctuation mark and there is a previous sentence it can be attached to
        if s in [".", "!", "?", ";", ":"] and final_sentences:
            final_sentences[-1] += s
        else:
            final_sentences.append(s)
            
    return final_sentences
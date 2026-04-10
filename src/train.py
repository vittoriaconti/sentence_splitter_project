import glob
from datasets import concatenate_datasets #Hugging Face function that merges more datasets
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from dataset import MODEL_NAME, process_file_to_dataset

def main():
    print("\nLooking for training and validation files...")
    
    #glob.glob looks for all the files with a specfic pattern
    train_files = glob.glob("../data/raw/*/*train.sent_split")
    dev_files = glob.glob("../data/raw/*/*dev.sent_split")
    
    if not train_files:
        print("Error: no training file found!")
        return

    print(f"{len(train_files)} training files found")
    print(f"{len(dev_files)} validation files found")

    #----- Setting some parameters for the training object -----

    #Process each training and validation file with the function from the file dataset.py
    #Basically convert each file into a suitable dataset and put them in a list
    train_datasets_list = [process_file_to_dataset(f) for f in train_files]
    dev_datasets_list = [process_file_to_dataset(f) for f in dev_files]

    #Merge all the training datasets in a unique huge dataset compatible with Hugging Face
    train_dataset = concatenate_datasets(train_datasets_list)
    #Merge all the validation datasets in a unique huge dataset compatible with Hugging Face
    dev_dataset = concatenate_datasets(dev_datasets_list)
    
    print(f"Dataset successfully merged! Total train samples: {len(train_dataset)}, Total validation samples: {len(dev_dataset)}")

    print("Loading the model and the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #Takes XML-RoBERTa and adds a layer for the the classification of samples
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        #Mapping between the numbers and the names of the classes
        #"O" corresponds to 0 (O = outside, not end of sentence)
        #"B-EOS" corresponds to 1 (B-EOS = beginning end of sentence
        #NB. base RoBERTa returns the hidden states of the tokes, which are
        #vectors of dimension 768. With AutoModelForTokenClassification the
        #vector of 768 elements is mapped in a vector of 2 elements (logits)
        #(acts as a normal linear layer) and then with softmax function the
        #logits are converted into probabilities
        id2label={0: "O", 1: "B-EOS"},
        label2id={"O": 0, "B-EOS": 1}
    )

    training_args = TrainingArguments(
        output_dir="../models/checkpoints",
        eval_strategy="epoch", 
        learning_rate=3e-5,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.02,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="../logs",
        logging_steps=50, 
        push_to_hub=False, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("\nTraining is ready to start!\n")
    trainer.train()

    print("\nTraining completed! Saving the final model...")
    trainer.save_model("../models/final_model")
    tokenizer.save_pretrained("../models/final_model")
    print("Model successfully saved in '../models/final_model'.")

if __name__ == "__main__":
    main()
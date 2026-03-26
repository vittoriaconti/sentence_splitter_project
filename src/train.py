import glob
from datasets import concatenate_datasets
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from dataset import MODEL_NAME, process_file_to_dataset

def main():
    print("\nLooking for training and validation files...")
    
    #glob.glob looks for all the files with that structure of the path in the subdirectory
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
    #Collects samples from the dataset in batches so that they can be analyzed by the GPU, also dealing with padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    #Takes XML-RoBERTa and adds a layer for the the classification of samples
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, #only 2 possible classes (0 and 1)
        #Mapping between the numbers and the names of the classes
        #"O" corresponds to 0 (O = outside, not end of sentence)
        #"B-EOS" corresponds to 1 (B-EOS = beginning end of sentence
        #NB. The model doesn't return 0 or 1, returns two raw numbers that represent
        #how much the model thinks the token belongs to class 0 or class 1. es = [4.5, -1.2].
        #The token is classified taking the max between those 2 raw numbers
        id2label={0: "O", 1: "B-EOS"},
        label2id={"O": 0, "B-EOS": 1}
    )

    training_args = TrainingArguments(
        output_dir="../models/sentence_splitter_english",
        eval_strategy="epoch", #the model evaluates itself on the validation set after each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8, #Process 8 sentences at once -> could be changed basing on the GPU capacities
        per_device_eval_batch_size=8,
        num_train_epochs=4, #the model has 3 epochs, so it will evaluate itself 3 times
        weight_decay=0.01, #regularization term that decreses weigths to avoid overfitting
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="../logs",
        logging_steps=50, #saves stats every 50 steps in the log subdirectory
        push_to_hub=False, #doesn't upload the model on the Hugging Face cloud
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    print("\nTraining is ready to start!\n")
    trainer.train()
    
    trainer.save_model("../models/sentence_splitter_english_final")
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()
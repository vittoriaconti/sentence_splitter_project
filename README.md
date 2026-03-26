## How to Get the Model and Run the Project

Due to GitHub's file size limits, the pre-trained model weights are not included in this repository. However, you can easily generate the model from scratch by following these steps:

1. **Train the Model:**  
Run the training script by executing the following command in your terminal:
```bash
python train.py
```
Hugging Face will automatically create a `models/` directory (if it doesn’t exist) where it will save the final model and the intermediate training checkpoints.

2. **Clean Up (Optional but recommended):**  
During training, several intermediate checkpoint folders are generated inside the `models/` directory. Once the training is successfully completed, you can safely delete these checkpoint folders to save disk space. You only need to keep the `final_model` folder.

3. **Evaluate Performances:**  
After the model has been trained and saved, you can test its performance against the baseline (NLTK) by running:
```bash
python score.py
```
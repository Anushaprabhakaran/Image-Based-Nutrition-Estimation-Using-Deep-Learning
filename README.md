# Image-Based-Nutrition-Estimation-Using-Deep-Learning
Image-based nutritional estimation using a ResNet-50 regression model trained on the Nutrition5k dataset with transfer learning and fine-tuning.

## Project Overview
This project implements a deep learning regression pipeline using a pretrained ResNet-50 model to predict four nutritional metrics — calories, fat, carbohydrates, and protein — directly from RGB food images using the Nutrition5k dataset.

**Author:** Anusha Prabhakaran
**Institution:** Purdue University, Masters in Engineering Management
**Course:** ECE 570, Spring 2026

---

## Repository Structure

- Checkpoint_1.ipynb — Baseline model, 2000 images, 10 epochs
- Checkpoint_2.ipynb — Expanded dataset, 3489 images, 25 epochs
- Final_Model.ipynb — Final model with all layers fine-tuned, learning rate scheduler, 35 epochs
- README.md — This file

---

## Dependencies

The code runs on Google Colab with the following libraries:
- Python 3.x
- PyTorch >= 2.0
- torchvision >= 0.15
- pandas
- numpy
- Pillow (PIL)
- scikit-learn
- Google Colab (drive mount)

All dependencies are pre-installed in Google Colab. No manual installation is required if running on Colab.

---

## Dataset

The project uses the Nutrition5k dataset by Thames et al. (2021). The dataset is publicly available but requires Google Cloud access.

### How to download:
1. Authenticate with Google Cloud in Colab using: from google.colab import auth and auth.authenticate_user()
2. Images are downloaded directly from the Google Cloud bucket at: gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/
3. Metadata CSV files (dish_metadata_cafe1.csv and dish_metadata_cafe2.csv) must be downloaded from the official Nutrition5k GitHub repository at: https://github.com/google-research-datasets/Nutrition5k

The download code is included in each notebook. 3489 RGB images are downloaded, resized to 224x224, and saved to Google Drive automatically.

---

## How to Run

Step 1 — Open Final_Model.ipynb in Google Colab

Step 2 — Set runtime to T4 GPU: Runtime > Change runtime type > T4 GPU

Step 3 — Mount Google Drive when prompted by the notebook

Step 4 — Upload dish_metadata_cafe1.csv and dish_metadata_cafe2.csv to Colab when prompted. These are available from the Nutrition5k GitHub repository.

Step 5 — Run all cells. The notebook will:
- Download and resize all 3489 images to Google Drive
- Create the filtered labels CSV
- Train the ResNet-50 model for 35 epochs
- Save checkpoints after every epoch to Google Drive
- Evaluate and print MAE per nutrient
- Predict nutritional values for held-out demo images

Note: If the session disconnects, simply re-run the notebook. It will automatically resume from the last saved checkpoint.

Expected runtime: Approximately 60-90 minutes on T4 GPU.

---

## Expected Results

After training, the model should produce approximately:

| Metric | Value |
|--------|-------|
| Test MSE | 3865 |
| Calories MAE | 83.58 kcal |
| Fat MAE | 5.53 g |
| Carbs MAE | 10.43 g |
| Protein MAE | 8.33 g |

Minor variations are expected due to random train/test splits and GPU non-determinism.

---

## Code Authorship

Written by the author (Anusha Prabhakaran):
- NutritionDataset class for dataset loading and preprocessing
- train_transform and test_transform for data augmentation pipelines
- train_model function including training loop with scheduler
- evaluate_model function for per-nutrient MAE evaluation
- predict_demo_images function for held-out prediction table
- predict_one function for single image prediction with comparison
- Checkpoint resume logic
- Image download and resize pipeline
- CSV filtering and metadata processing

Adapted from PyTorch documentation and standard practices:
- ResNet-50 model loading using torchvision.models
- DataLoader setup with num_workers and pin_memory
- ReduceLROnPlateau scheduler initialization

LLM Assistance: Claude (Anthropic) was used for debugging, resolving Google Colab environment issues, and guidance on training strategies. All code was executed, tested, and verified by the author.

---

## Checkpoints

Model checkpoints are saved to Google Drive after every epoch at:
/content/drive/MyDrive/nutrition_checkpoints_final/

The notebook automatically detects and resumes from the latest checkpoint if training is interrupted.

---

## References

Thames et al. (2021). Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food. CVPR 2021.

He et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

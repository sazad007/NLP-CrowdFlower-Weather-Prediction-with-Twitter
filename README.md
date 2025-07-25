
# 🌦️ NLP: Partly Sunny with a Chance of Hashtags

This repository implements a weather-condition prediction model based on the Kaggle competition **“CrowdFlower: Weather Prediction with Twitter”**, also known as *Partly Sunny with a Chance of Hashtags*.

## 🧠 Project Overview

- **Objective**: Given tweets about the weather, build a model to classify the weather condition (e.g. sunny, rainy, snow).
- **Approach**: Preprocess tweets → engineer features using NLP → train and evaluate machine learning models to predict weather-related categories.

---

## 📁 Dataset Description

**Source**: Kaggle competition [CrowdFlower: Weather Twitter](https://www.kaggle.com/competitions/crowdflower-weather-twitter)

- Format: `train.csv` for training and `test.csv` for scoring
- Total rows: ~120k annotated tweets
- Key columns include:

| Column Name      | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `tweet`          | Raw tweet text                                                              |
| `location`       | User‑provided location (may be blank)                                       |
| `state`          | State from where the tweet was created                                   |
| `s1-s5, w1-w4, k1-k15` | 24 numerical columns holding confidence scores (0.0 to 1.0) for each label|

### Label Structure

The 24 labels cover three axes of annotation:

- **Time reference**: past, present, future
- **Sentiment**: positive, negative, neutral
- **Weather condition**: sunny, cloudy, rainy, snow, etc.

Each row provides a probability of belonging to each of these categories, allowing either multi-label modeling or taking the maximum-confidence label for classification.

---

## 📝 Notebook: `Partly_Sunny_with_a_Chance_of_Hashtags.ipynb`

This notebook is built using **TensorFlow/Keras** and focuses on using **deep learning** for multi-label classification of weather-related tweet categories. It follows this pipeline:

1. **Data Loading**
   - Reads the dataset using `pandas` and loaded word2vec model from Google Drive.

2. **Data Preprocessing**
   - Extracts relevant label columns (24 class probabilities).
   - Cleans tweets using regular expressions (`re`) to remove special characters, numbers, URLs, and punctuation.
   - Applies **NLTK** for stopword removal.

3. **Text Vectorization**
   - Tokenizes tweets using `word2vec`.
  
4. **Model Building**
   - Constructs a simple **feedforward neural network** using `tensorflow.keras`:
     - Input layer
     - Dense layers with `ReLU` activations
     - Dropout layers for regularization
     - 3 Output layer with 24 units and `softmax`, `sigmoid` activation

5. **Loss Function & Optimization**
   - Uses categorical_crossentropy and binary_crossentropy as the loss function to compare predicted vs. actual class distributions.

6. **Training**
   - Splits data into train/test using `train_test_split`
   - Trains the model using `.fit()` with validation data

7. **Evaluation**
   - Evaluates the trained model using `.evaluate()` on the test set

---
## 📈 Kaggle Submission Results

After training and evaluating the model, predictions were generated and submitted to the **CrowdFlower: Weather Twitter** competition on Kaggle.

- ✅ **Final RMSE Score**: `0.16365`
- 🧪 **Public RMSE Score (evaluated)**: `0.16344`
- 📏 **Evaluation Metric**: Root Mean Squared Error (RMSE)
- ⏳ **Note**: The submission was made after the competition deadline, so it was **not reflected on the public leaderboard**.

Despite the late submission, these results provide a useful benchmark for model performance against the original competition standard.

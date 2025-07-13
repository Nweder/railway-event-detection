# Railway Event Detection with Acceleration Data

This project analyzes transient events (e.g., defects or discontinuities) in railway switches using acceleration data from trackside sensors. The task was completed as part of an assignment focused on preprocessing, normalization, and machine learning classification using SVM.

## Dataset
The dataset contains extracted features from acceleration signals for three trials:

- Trail1_extracted_features_acceleration_m2ai0.csv  
- Trail2_extracted_features_acceleration_m2ai0.csv  
- Trail3_extracted_features_acceleration_m2ai0.csv

Each file includes both statistical and frequency features, along with a column `event` indicating transient occurrences.

## Goals
- Clean and merge the data
- Normalize features
- Encode event types (normal = 0, event = 1)
- Train an SVM classifier
- Compare performance using:
  - 80/20 train-test split
  - 5-fold cross-validation

## Model
- Classifier: Support Vector Machine (SVC)
- Accuracy (80/20 split): ~96.7%
- Average Accuracy (5-fold CV): ~97.95%

## Completed Tasks
-Grade 3: Preprocessing and normalization
-Grade 4: SVM training and cross-validation

## 👤 Author
**Mohamad Nweder**  
Course: AI and Digitalization  
Luleå University of Technology  

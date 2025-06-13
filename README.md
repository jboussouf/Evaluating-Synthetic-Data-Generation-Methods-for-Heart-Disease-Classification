# Evaluating Synthetic Data Generation Methods for Heart Disease Classification

## 1. Introduction
This project investigates the impact of various synthetic data generation techniques on the performance of machine-learning classifiers for heart disease prediction. Starting from a real dataset of **1,025** patient records, we generated additional data using seven different methods and re-trained the same set of classifiers to compare their performance when evaluated on synthetic data.

## 2. Dataset
- **Original data**: 1,025 rows, each with clinical and demographic features and a binary heart disease label.
- **Train/Test split**: 75% train (≈768 samples), 25% test (≈257 samples).  
- **Synthetic data**: Generated to 75% of the original training set size (≈576 samples).

## 3. Synthetic Data Generation Methods
1. **Bootstrapping**  
   Resample with replacement from the original training data to create bootstrapped datasets.  
2. **Data Swapping/Shuffling**  
   Independently shuffle each feature column, preserving marginals but breaking joint distributions.  
3. **SMOTE** (Synthetic Minority Over-sampling Technique)  
   Interpolates new minority-class examples in feature space.  
4. **GaussianCopulaSynthesizer**  
   Fits a Gaussian copula to the multivariate distribution and samples from it.  
5. **CTGAN** (Conditional Tabular GAN)  
   Uses conditional vector embeddings to generate realistic mixed-type tabular data.  
6. **TVAESynthesizer**  
   Employs a tabular variational autoencoder optimized for mixed data types.  
7. **CopulaGANSynthesizer**  
   Combines copula fitting with GAN training for improved fidelity.

## 4. Experimental Setup
- **Classifiers evaluated**:  
  - Decision Tree  
  - Random Forest  
  - Extra Trees  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Quadratic Discriminant Analysis (QDA)  
  - Multi-Layer Perceptron (MLP)  
- **Procedure**:  
  1. Train each classifier on the original training set.  
  2. Evaluate on the held-out real test set → _Baseline real performance_.  
  3. Generate synthetic training data (≈576 samples).  
  4. Retrain each classifier on synthetic data only.  
  5. Evaluate on a held-out **synthetic** test set (75% of synthetic data) → _Synthetic-only performance_.

## 5. Results

| Model                 | Real-Train → Real-Test Accuracy | Synth-Train → Synthetic-Test Accuracy |
|-----------------------|---------------------------------|---------------------------------------|
| Decision Tree         | 98.83 %                         | 90.00 %                               |
| Random Forest         | 98.83 %                         | 96.66 %                               |
| Extra Trees           | 98.83 %                         | 95.33 %                               |
| Logistic Regression   | 82.49 %                         | 95.33 %                               |
| SVM                   | 70.03 %                         | 94.66 %                               |
| QDA                   | 85.99 %                         | 78.66 %                               |
| MLP                   | 82.87 %                         | 96.00 %                               |

## 6. Discussion
- **Baseline vs. synthetic**: Classifiers trained on **real data** achieved near-perfect accuracy on real test data, whereas training on **synthetic data** and testing on synthetic test sets yielded lower but still high scores—suggesting synthetic samples capture many patterns but may overfit to their own distribution.  
- **Method comparison**:  
  - **Bootstrapping** and **SMOTE** produced the most stable synthetic-synthetic performance (90 % and 95.33 %, respectively).  
  - **GAN/VAE-based methods** (CTGAN, TVAE, CopulaGAN) generally matched Bootstrapping but required more tuning to avoid mode collapse or oversmoothing.  
  - **QDA** struggled most on synthetic data (78.66 %), indicating difficulties modeling complex feature distributions from generated samples.  
- **Implications**: High synthetic-synthetic accuracy does not guarantee real-world generalization. Overfitting to synthetic artifacts can inflate performance when evaluated on synthetic holdouts.

## 7. Conclusion
Synthetic data can help augment limited healthcare datasets, but:
- **Bootstrapping** and **SMOTE** remain the most reliable generators under default settings.
- **Deep generative models** (CTGAN, TVAESynthesizer, GaussianCopula) require careful hyperparameter tuning and possibly hybrid approaches to close the gap with real-data performance.

---

## 8. Datasets
[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

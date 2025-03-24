# Equitable-AI-for-Dermatology-VIR_AJL_Team-Dermaplane

This repository contains our team's work on the **"Equitable AI for Dermatology"** Kaggle competition, in collaboration with the **Algorithmic Justice League (AJL)**. Our objective: build a fair and inclusive model to classify 21 dermatological conditions across diverse skin tones.

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Sarah Basil | @Sarah-Basil | Contributed to Data Exploration and Model development |
| Danielle Raine | @DanielleRaine | Contributed to testing and finetuning the models|
|Jubyaid Uddin | @jubyaid123 | Contributed to early data exploration and preprocessing |
| | | |
---

## **🎯 Project Highlights**

- **Dataset**: A subset of the FitzPatrick17k dataset (~4,500 images, 21 conditions, multiple Fitzpatrick skin tones).
- **Goal**: Achieve high F1 score on multi-class classification, ensuring fair performance across diverse skin tones.
- **Approach**:
  - **Data Augmentation**: Rotation, zoom, brightness, color shifts, etc. to handle limited data and reduce overfitting.
  - **Oversampling**: Balanced out label+Fitzpatrick scale pairs to mitigate underrepresentation.
  - **Transfer Learning**: MobileNetV2 (pretrained on ImageNet) adapted for dermatology classification.
- **Notable Results**: 
  - ~0.83 Weighted F1 on our internal validation set.
  - **Kaggle Submission Score**: **0.41537** (public leaderboard).
- **Fairness**: Applied oversampling to ensure more equal representation of higher Fitzpatrick scales.

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/DanielleRaine/Equitable-AI-for-Dermatology-VIR_AJL_Team-Dermaplane.git
```
Navigate into the directory:
```bash
git clone https://github.com/DanielleRaine/Equitable-AI-for-Dermatology-VIR_AJL_Team-Dermaplane.git
cd Equitable-AI-for-Dermatology-VIR_AJL_Team-Dermaplane
```

---

### **Step 2: Install Dependencies**
Ensure you have Python installed. It's recommended to create a virtual environment to keep dependencies isolated.

1. **Create a virtual environment (optional but recommended):**  
```bash
python -m venv env
```
2. **Activate the virtual environment:**  
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
3. **Install required packages:**  
```bash
pip install -r requirements.txt
```

---

### **Step 3: Set Up the Environment**

```bash
cp .env.example .env
```
Edit the `.env` file with appropriate keys or paths if needed.

---
### **Step 4: Access the Dataset(s)**
- Download the dataset(s) from the provided links or resources.
- Unzip or place them in the directory structure referenced by the code (e.g., `../input/bttai-ajl-2025/`).

---
### **Step 5: Run the Notebook or Scripts**
1. **For Jupyter Notebook:**  
   Launch Jupyter Notebook by running:
   ```bash
   jupyter notebook
   ```
2. Open the relevant .ipynb file in your browser and run all cells.
(Or run in Google Colab / local environment as desired.)

## **🏗️ Project Overview**

We now live in a world where AI governs access to information, opportunity and freedom. However, AI systems can perpetuate racism, sexism, ableism, and other harmful forms of discrimination, therefore, presenting significant threats to our society - from healthcare, to economic opportunity, to our criminal justice system.

The Algorithmic Justice League is an organization that combines art and research to illuminate the social implications and harms of artificial intelligence.

AJL’s mission is to raise public awareness about the impacts of AI, equip advocates with resources to bolster campaigns, build the voice and choice of the most impacted communities, and galvanize researchers, policymakers, and industry practitioners to prevent AI harms.

AI is transforming healthcare, yet dermatology AI tools often underperform for people with darker skin tones due to a lack of diverse training data. This can lead to diagnostic errors, delayed treatments, and health disparities for underserved communities. This challenge from Break Through Tech and the Algorithmic Justice League invited us to help address this issue by building an inclusive machine learning model for dermatology. 

We aimed to train a model that can classify 21 different skin conditions across diverse skin tones, using the datasets provided by the competition as our starting point. 

---

## **📊 Data Exploration**

### **Dataset(s) Used**
This project uses a subset of the **FitzPatrick17k dataset** with approximately **4,500 images** covering **21 dermatological conditions** across various skin tones. The dataset includes:

- **images.zip**: Divided into `train/` and `test/` directories.
- **train.csv**: Metadata for training images.
- **test.csv**: Metadata for testing (without labels).
- **sample_submission.csv**: Template for predictions.

---

### **Data Description**
| Column               | Description                           |
|----------------------|---------------------------------------|
| `md5hash`            | Unique image identifier.             |
| `fitzpatrick_scale`   | Self-reported FitzPatrick Skin Tone. |
| `label`               | Medical diagnosis (target variable). |
| `file_path`           | Path to the image file.              |
| `encoded_label`       | Numerical label for model training.  |

---

### **Preprocessing Approach**
1. **Data Loading:** Using `pandas` to load `train.csv` and `test.csv`.
2. **File Path Construction:** Adding `.jpg` extension to `md5hash` and building paths using:
   ```python
   train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash'] + '.jpg'
   ```
3. **Label Encoding:** Converting labels to numerical values with `LabelEncoder`.
   ```python
   from sklearn.preprocessing import LabelEncoder
   train_df['encoded_label'] = LabelEncoder().fit_transform(train_df['label'])
   ```
4. **Image Preprocessing:** Standardization & augmentation via `ImageDataGenerator`.
5. **Handling Negative Fitzpatrick Scales:** We converted negative or unknown scales to 0 to represent "unknown."
6. **Balancing:** Oversampled each (label, scale) pair to address both label and skin tone imbalances.

---

### **Challenges & Assumptions**
- **Class Imbalance:** Some conditions are over-represented; F1-score is used for evaluation.
- **Incomplete Metadata:** Limited quality control data (`qc` column) and potential ambiguous labels.
- **Preprocessing Assumptions:** Images are resized and normalized for model training.

---

## 🧠 Model Development

1. **Image Preprocessing & Augmentation**  
   - **Keras** `ImageDataGenerator` with:
     - `rescale=1/255`
     - `rotation_range=15`, `width_shift_range=0.1`, `height_shift_range=0.1`, `zoom_range=0.15`
     - Additional brightness/channel shifts to mimic real-world color variations.

2. **Model Architecture**  
   - **MobileNetV2** (pretrained on ImageNet, `include_top=False`).
   - **Custom Head**:
     ```python
     layers.GlobalAveragePooling2D(),
     layers.Dropout(0.3),
     layers.Dense(256, activation='relu'),
     layers.Dropout(0.3),
     layers.Dense(num_classes, activation='softmax')
     ```
3. **Training Setup**  
   - **Optimizer**: `Adam(learning_rate=1e-3)`.
   - **Loss**: `sparse_categorical_crossentropy`.
   - **Class Weights**: computed via `sklearn.utils.class_weight.compute_class_weight`.
   - **Callbacks**:
     - `EarlyStopping(patience=3, restore_best_weights=True)`
     - `ModelCheckpoint(save_best_only=True)`
   - **Fine-Tuning** (optional):
     - Unfreeze part or all of MobileNetV2 layers at a lower LR (e.g., `1e-5`) after initial training.

---

## 📈 Results & Key Findings

- **Internal Validation**:
  - Weighted F1: ~**0.83**  
  - Observed good performance on major classes, some confusion among visually similar conditions.

- **Kaggle Submission**:
  - Public Leaderboard Score: **0.41537**  

- **Performance by Skin Tone** (preliminary):
  - Oversampling approach helped underrepresented scales (e.g., 5–6).
  - Would need more thorough per-scale breakdown for deeper fairness insights.

---

## 🖼️ Impact Narrative

By oversampling underrepresented (condition, skin tone) pairs and using data augmentation, we aimed to improve the model’s sensitivity across different Fitzpatrick scales. This aligns with the **Algorithmic Justice League** mission to reduce bias in AI systems.

**Why it Matters**:  
- Equitable detection of dermatological conditions can lead to earlier interventions for communities historically underrepresented in clinical imagery.
- Our approach highlights the importance of balanced data and fairness metrics in medical AI.

---

## 🚀 Next Steps & Future Improvements

1. **Per-Scale Evaluation**  
   - Compute F1 or other metrics specifically on each Fitzpatrick scale subset.

2. **Explainability**  
   - Use Grad-CAM or LIME to see which image regions the model focuses on.

3. **Additional Datasets**  
   - Merge with other public dermatology sets to broaden coverage of conditions and skin tones.

4. **Hyperparameter Tuning**  
   - Explore advanced architectures (EfficientNet, Xception) or refined fine-tuning schedules.

5. **Real-World Testing**  
   - Validate on clinically verified images with broader demographics.

---

## 📄 References & Additional Resources

- **Algorithmic Justice League**: [https://www.ajl.org/](https://www.ajl.org/)  
- **MobileNetV2 Paper**: Sandler et al. (2018) [*arXiv:1801.04381*](https://arxiv.org/abs/1801.04381)  
- **FitzPatrick17k Dataset**: Groh et al. (2021) [*arXiv:2104.09957*](https://arxiv.org/abs/2104.09957)  
- **Keras Documentation**: [https://keras.io/](https://keras.io/)  
- **Kaggle**: [Competition Link](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)  

**Thank you for checking out our repository!** If you have questions or suggestions, please open an issue or reach out to any team member.

---

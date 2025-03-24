# Equitable-AI-for-Dermatology-VIR_AJL_Team-Dermaplane

This repository contains our team's work on the **"Equitable AI for Dermatology"** Kaggle competition, in collaboration with the **Algorithmic Justice League (AJL)**. Our objective: build a fair and inclusive model to classify 21 dermatological conditions across diverse skin tones.

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Alice Doe | @AliceDoe | Built CNN model, performed data augmentation |
| Mel Ramakrishnan | @MelRam | Led EDA, visualized dataset distributions, handled missing data |
| Charlie Nguyen | @CharlieN | Implemented explainability tools |
/\ examples /\
| Sarah Basil | | |
| Danielle Raine | @DanielleRaine | |
|Jubyaid Uddin | @jubyaid123 | |
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


---
### **Step 5: Run the Notebook or Scripts**
1. **For Jupyter Notebook:**  
   Launch Jupyter Notebook by running:
   ```bash
   jupyter notebook
   ```
   Then, navigate to the desired `.ipynb` file in your browser.

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

---

### **Challenges & Assumptions**
- **Class Imbalance:** Some conditions are over-represented; F1-score is used for evaluation.
- **Incomplete Metadata:** Limited quality control data (`qc` column) and potential ambiguous labels.
- **Preprocessing Assumptions:** Images are resized and normalized for model training.

---
**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## **🧠 Model Development**

**Describe (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

To address model fairness, we leveraged data augmentation techniques to balance the number of entries of skin conditions affecting various skin tones. Then, we validated the model performance on a validation data set to assess the impact on classification accuracy across different skin tones.

We want AI to serve everyone equally, whether they have different skin colors, gender identities, sexual orientations, physical abilities, or anything else that makes each person unique. Our work will aid the pursuit of this goal.

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---


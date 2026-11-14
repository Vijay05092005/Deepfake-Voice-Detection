# Deepfake Voice Detection System
### Intelligent Audio Authenticity Verification Using Deep Learning and Prolog

## 1. Project Overview

The Deepfake Voice Detection System is an intelligent application designed to identify audio recordings containing genuine human speech or synthetic/manipulated speech.

The system combines a **Convolutional Neural Network (CNN)** with a **Prolog-based knowledge system**.

The CNN analyses audio features and predicts whether a recording is bona fide or spoof. Prolog applies rule-based reasoning to interpret the prediction and provide an assessment and recommendation.

## 2. Objectives

- Detect synthetic or manipulated speech recordings.
- Extract meaningful audio features using Log-Mel Spectrograms.
- Develop a CNN-based audio classification model.
- Integrate deep learning predictions with Prolog reasoning.
- Provide an interactive interface for uploading and analysing audio recordings.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Architecture | Convolutional Neural Network (CNN) |
| Audio Processing | Librosa |
| Feature Extraction | Log-Mel Spectrogram |
| Knowledge Representation | Prolog |
| Python-Prolog Integration | PySwip |
| Frontend | Streamlit |
| Development Environment | Google Colab and VS Code |

## 4. System Architecture

The system follows a hybrid AI architecture.

**Processing workflow:**

1. User uploads an audio recording.
2. The audio preprocessing module loads and processes the recording.
3. Log-Mel Spectrogram features are extracted.
4. The trained CNN analyses the extracted features.
5. The CNN predicts bona fide or spoof probabilities.
6. Prolog applies knowledge-based rules to the prediction.
7. The system generates an assessment and recommendation.
8. Results are displayed through the Streamlit interface.

## 5. Dataset

**Dataset:** ASVspoof 2019 – Logical Access (LA).

The dataset contains genuine and spoofed speech recordings.

| Dataset Split | Number of Recordings |
|---|---:|
| Training | 25,380 |
| Development | 24,844 |
| Evaluation | 71,237 |
| **Total** | **121,461** |

The dataset was divided into training, development, and evaluation sets.

The original dataset is not included in this GitHub repository.

## 6. Deep Learning Model

The project uses a three-block 2D Convolutional Neural Network.

**Input features:**
- Feature type: Log-Mel Spectrogram
- Mel frequency bins: 128
- Time frames: 501
- Input shape: 128 × 501

**Classification labels:**

- Class 0: Bona fide (genuine human speech)
- Class 1: Spoof (synthetic or manipulated speech)

The trained model is stored in:

`models/saved_models/best_deepfake_cnn.pth`

The model configuration and decision threshold are stored in:

`models/saved_models/final_model_config.json`

## 7. Model Evaluation

The trained CNN was evaluated using the held-out evaluation dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 92.31% |
| Balanced Accuracy | 83.25% |
| Macro F1-Score | 80.75% |
| ROC-AUC | 95.54% |
| Average Precision | 99.48% |

The evaluation dataset contains more spoof recordings than bona fide recordings. Therefore, accuracy should be interpreted alongside balanced accuracy and macro F1-score.

The model demonstrates strong spoof detection performance, although some genuine recordings are incorrectly classified as spoof.

## 8. Prolog Knowledge-Based Reasoning

Prolog provides an additional reasoning layer after CNN classification.

It evaluates:
- CNN classification result.
- Model confidence level.
- Audio assessment.
- Recommendation for further verification.

The system produces assessments such as:
- Suspicious
- Likely genuine
- Requires verification
- Uncertain

The knowledge base is implemented in:

`prolog/deepfake_rules.pl`

## 9. Project Structure

```text
Deepfake-Voice-Detection/
│
├── frontend/
│   └── app.py
│
├── integration/
│   ├── audio_preprocessing.py
│   ├── cnn_model.py
│   ├── cnn_prolog_integration.py
│   ├── load_cnn.py
│   ├── predict_audio.py
│   ├── test_audio_preprocessing.py
│   └── test_prolog.py
│
├── models/
│   └── saved_models/
│       ├── best_deepfake_cnn.pth
│       └── final_model_config.json
│
├── prolog/
│   └── deepfake_rules.pl
│
├── investigate_audio.py
├── .gitignore
└── README.md
```

## 10. Installation and Execution

### Prerequisites

- Python
- SWI-Prolog
- Git
- Required Python libraries

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Deepfake-Voice-Detection.git
```

### Step 2: Navigate to the project directory

```bash
cd Deepfake-Voice-Detection
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install dependencies

Install the required Python libraries used by the project, including PyTorch, Librosa, Streamlit, and PySwip.

Ensure SWI-Prolog is installed and configured correctly.

### Step 5: Run the application

```bash
python -m streamlit run frontend/app.py
```

The Streamlit application will open in your browser.

Upload a WAV or FLAC audio recording and click **Analyse Audio** to view the results.

## 11. Limitations

- The model may incorrectly classify some genuine recordings as spoof.
- Performance can vary depending on audio quality and recording characteristics.
- The model's softmax probabilities are not calibrated confidence estimates.
- The system provides an automated assessment and is not a substitute for professional forensic verification.

## 12. Future Improvements

- Improve detection of genuine speech recordings.
- Explore advanced deep learning architectures.
- Expand evaluation using additional datasets.
- Improve model generalisation to different audio recording conditions.
- Enhance the frontend with additional visualisations and analysis features.

## 13. Conclusion

This project demonstrates a hybrid artificial intelligence approach combining deep learning and symbolic reasoning for deepfake voice detection.

The CNN identifies patterns associated with genuine and synthetic speech, while Prolog provides rule-based interpretation and recommendations.

The integrated system offers an interactive approach to audio authenticity assessment.

---

**Project:** Deepfake Voice Detection System  
**Technologies:** Python | PyTorch | CNN | Librosa | Prolog | Streamlit

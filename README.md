# 🌍 Land Type Classifier

An AI-powered land type classification system that identifies land cover categories from satellite images using a fine-tuned ResNet18 deep learning model trained on the EuroSAT dataset.

---

## 🚀 Live Demo

Hugging Face Space:

https://huggingface.co/spaces/Mohammad4422/Land-Classifier

---

## Features

- Upload satellite images
- Predict land cover type
- Confidence score for predictions
- Interactive Gradio interface
- Fast inference using PyTorch
- Trained on the EuroSAT dataset

---

## Land Classes

- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

---

## Model

- Architecture: ResNet18
- Framework: PyTorch
- Dataset: EuroSAT
- Trained Model: `resnet18_eurosat.pth`

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Gradio
- NumPy
- Pillow
- Matplotlib

---

## Project Structure

```
Land-Type-Classifier
│
├── app.py
├── requirements.txt
├── resnet18_eurosat.pth
├── README.md
└── data
    └── notebooks
```

---

## Installation

```bash
git clone https://github.com/mhesham-eng/Land-Type-Classifier.git

cd Land-Type-Classifier

pip install -r requirements.txt

python app.py
```

---

## Demo

Launch the application locally:

```bash
python app.py
```

Or use the online demo on Hugging Face.

---

## Team

- Mohammad Hesham 
- Raneem Ehab
- Aya Hemida
- Hussien elsayed
- ز

---

## License

MIT License

# 🛒 Amazon AI Recommendation System

> An end-to-end AI-powered recommendation system delivering personalized product suggestions using **Collaborative Filtering (SVD)** and **NLP-based Content Filtering (TF-IDF)**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Surprise](https://img.shields.io/badge/Surprise-SVD-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

👉 **[Open App](https://itscharan05-amazon-ai-recommendation-system-appapp-rw4m4s.streamlit.app/)** 

---

## ✨ Features

| Feature | Description |
|---|---|
| 👤 Personalized Recommendations | SVD Matrix Factorization based on user ratings and behavior |
| 🧠 AI Product Similarity | TF-IDF NLP on review text to find semantically similar products |
| 🎨 Modern Dashboard UI | Dark theme, glassmorphism cards, responsive Streamlit layout |
| ⚡ Fast Performance | `@st.cache_data` and `@st.cache_resource` for instant repeat queries |

---

## 🧠 ML Models

| Model | Library | Purpose |
|---|---|---|
| SVD (Matrix Factorization) | Surprise | Personalized user recommendations |
| TF-IDF Vectorization | Scikit-Learn | Text feature extraction from reviews |
| Cosine Similarity | Scikit-Learn | Similar product detection |

---

## 🏗️ Architecture

```
User Ratings                    Review Text
     │                               │
     ▼                               ▼
Rating Matrix               TF-IDF Vectorization
     │                               │
     ▼                               ▼
SVD Collaborative              Cosine Similarity
    Filtering                        │
     │                               │
     ▼                               ▼
Personalized              Similar Product
Recommendations           Recommendations
          │                    │
          └────────┬───────────┘
                   ▼
          Streamlit Dashboard
                   │
                   ▼
            Caching Layer
     (@st.cache_data / @st.cache_resource)
```

---

## 📂 Project Structure

```
amazon-recommendation-system/
│
├── app/
│   └── app.py                  # Streamlit dashboard
│
├── data/
│   ├── processed/
│   │   └── clean_data.csv      # Cleaned dataset
│   └── raw/
│       └── amazon-product-review.csv
│
├── notebooks/
│   └── processing.ipynb        # Data preprocessing & EDA
│
├── requirements.txt
├── packages.txt
├── README.md
└── .python-version
```

---

## 📊 Dataset

**Source:** [Amazon Product Reviews — Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)

The dataset includes:
- User reviews and ratings
- Product names and metadata
- Review text for NLP processing

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/itsCharan05/amazon-ai-recommendation-system.git
cd amazon-ai-recommendation-system
```

### 2. Create and activate a Conda environment
```bash
conda create -n surprise_env python=3.10
conda activate surprise_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app/app.py
```

---

## 📈 Model Performance

| Model | Metric |
|---|---|
| SVD (Collaborative Filtering) | RMSE on test split |
| TF-IDF + Cosine Similarity | Cosine distance score |

---

## 🌐 Deployment Options

- [Streamlit Community Cloud](https://streamlit.io/cloud) ← recommended
- [Render](https://render.com)
- [Railway](https://railway.app)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- AWS / Azure

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python 3.10 | Core development |
| Streamlit | Frontend dashboard |
| Pandas + NumPy | Data processing |
| Scikit-Learn | TF-IDF vectorization & cosine similarity |
| Scipy | SVD collaborative filtering |

---

## 🗺️ Roadmap

- [ ] Product image support
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Real-time recommendations
- [ ] User authentication
- [ ] Recommendation REST API
- [ ] Deep learning model (Neural CF)
- [ ] Docker deployment
- [ ] Cloud database integration

---

## 💡 Key Learnings

- Recommendation system design (collaborative vs content-based)
- Matrix factorization with SVD
- NLP-based similarity search with TF-IDF
- Streamlit app development and deployment
- Model evaluation and data preprocessing pipelines

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add some feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using Python & Streamlit</p>

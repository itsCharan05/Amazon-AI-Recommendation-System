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
git clone https://github.com/YOUR_USERNAME/amazon-ai-recommendation-system.git
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

This app can be deployed on any of the following platforms:

- [Streamlit Community Cloud](https://streamlit.io/cloud) ← recommended for quick sharing
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
| Surprise | SVD collaborative filtering |

---

## 🗺️ Roadmap

- [ ] Product image support
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Real-time recommendations
- [ ] User authentication
- [ ] Recommendation REST API
- [ ] Deep learning recommendation model (e.g. Neural CF)
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

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using Python & Streamlit</p>

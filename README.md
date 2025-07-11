# 📘 Extractive Text Summarization Methods

This repository presents a collection of traditional and neural extractive summarization techniques including TF-IDF, KMeans with Sentence-BERT, BERTSum, and TextRank with graph-based approaches.

Extractive summarization selects key sentences directly from the source text based on statistical and semantic features without generating new sentences.

---

## 🚀 Notebooks & Approaches

| Notebook | Method | Technique |
|----------|--------|-----------|
| `01_tf-idf_cosine.ipynb` | TF-IDF + Cosine Similarity | Sentence scoring based on vector space model |
| `02_kmeans_sentencebert.ipynb` | Sentence-BERT + KMeans | Clustering-based summarization |
| `03_bertsum_cnn_dailymail.ipynb` | BERTSum | Supervised sentence classification |
| `04_pagerank_similarity_graph.ipynb` | TextRank | Graph-based unsupervised ranking |

---

## 📂 Folder Structure

```

extractive-text-summarization-methods/
├── Notebooks/
│   ├── 01\_tf-idf\_cosine.ipynb
│   ├── 02\_kmeans\_sentencebert.ipynb
│   ├── 03\_bertsum\_cnn\_dailymail.ipynb
│   └── 04\_pagerank\_similarity\_graph.ipynb
├── LICENSE
├── README.md
└── requirements.txt

````

---

## 💻 Run Locally

```bash
git clone https://github.com/Koushim/extractive-text-summarization-methods.git
cd extractive-text-summarization-methods
pip install -r requirements.txt
````

---

## 📈 Highlights

* ✅ Traditional TF-IDF + Cosine approach
* ✅ Semantic embeddings with Sentence-BERT
* ✅ Supervised extractive model with BERTSum
* ✅ Unsupervised TextRank (graph + PageRank)

---

## ✍️ Author

**Koushik Reddy**
🔗 [Hugging Face](https://huggingface.co/Koushim) 

---

## 📌 License

This project is open source and available under the [Apache License](LICENSE).


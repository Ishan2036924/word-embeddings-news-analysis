# Word Embeddings for News Article Analysis

A comprehensive exploration of Word2Vec and GloVe embeddings applied to news article analysis, including training custom embeddings, comparative analysis, visualizations, and document clustering.


## Project Overview

This project demonstrates practical applications of word embeddings in Natural Language Processing:

- **Trained custom Word2Vec models** on ~60,000 news articles (~1M tokens)
- **Compared custom embeddings with pre-trained GloVe** to understand trade-offs
- **Created visualizations** revealing semantic relationships in embedding space
- **Built a document clustering application** that automatically groups articles by topic

## Key Findings

### 1. Domain-Specific vs General Embeddings

| Word Pair | Custom Word2Vec | GloVe | Winner |
|-----------|-----------------|-------|--------|
| apple-iphone | 0.79 | 0.63 | Custom |
| climate-change | 0.81 | 0.63 | Custom |
| security-hack | 0.28 | 0.03 | Custom |
| facebook-twitter | 0.49 | 0.84 | GloVe |

**Insight**: Custom embeddings excel at domain-specific collocations, while GloVe captures broader semantic relationships.

### 2. Visualizations

#### Semantic Direction Analysis


Words projected onto the "startup → company" axis reveal a meaningful gradient from emerging ventures to established entities.

#### Similarity Heatmap

Pairwise similarities reveal clusters: tech companies, political figures, and security-related terms.

### 3. Document Clustering

Successfully clustered 58,000+ documents into 6 topics using averaged word embeddings:
- Legal/Judicial News
- Domestic Policy
- International News
- Tech/Science/Business
- Election Campaign Coverage



## Technologies Used

- **Python 3.8+**
- **Gensim** - Word2Vec training
- **NLTK** - Text preprocessing
- **Scikit-learn** - Clustering, PCA, t-SNE
- **Matplotlib/Seaborn** - Visualizations
- **Pandas/NumPy** - Data manipulation

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/word-embeddings-news-analysis.git
cd word-embeddings-news-analysis
pip install -r requirements.txt
```

## Dataset

This project uses the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle. Download and place in the project directory before running.

## Usage

Open the Jupyter notebook and run cells sequentially:

```bash
jupyter notebook Word_Embedding_Project.ipynb
```

## Project Structure

```
├── Word_Embedding_Project.ipynb  # Main analysis notebook
├── requirements.txt              # Python dependencies
├── images/                       # Visualization outputs
└── README.md                     # This file
```

## Results Summary

| Metric | Value |
|--------|-------|
| Documents Processed | 58,708 |
| Vocabulary Size | 15,905 words |
| Models Trained | 4 (CBOW/Skip-gram × 100d/200d) |
| Clustering Accuracy | 6 distinct topic clusters |

## Lessons Learned

1. **More data isn't always better** - Domain focus matters more than raw size for specialized tasks
2. **Preprocessing decisions matter** - Stopword removal significantly impacts embedding quality
3. **Visualizations reveal insights** - t-SNE showed clusters invisible in raw similarity scores
4. **Trade-offs exist** - No single embedding model is best for all use cases

## Future Improvements

- [ ] Train on larger, more balanced corpus
- [ ] Compare with contextual embeddings (BERT)
- [ ] Build interactive visualization dashboard
- [ ] Add semantic search functionality

## License

MIT License

## Acknowledgments

- Stanford NLP for GloVe embeddings
- HuffPost for the News Category Dataset
- Gensim library developers

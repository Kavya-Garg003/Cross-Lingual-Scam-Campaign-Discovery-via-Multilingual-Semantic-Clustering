# Multilingual Scam Campaign Discovery

> Identifying coordinated scam operations across 5 languages using semantic embeddings and unsupervised clustering.

---

## What this project does

Scammers don't operate randomly. They run **organized campaigns** — the same scam idea sent across thousands of messages, in multiple languages, with slightly different wording. A lottery scam in Tamil and the same lottery scam in Hindi look completely different on the surface but come from the same operation.

This system finds those connections automatically.

Instead of classifying individual messages as spam or not spam, this pipeline groups semantically similar messages across languages into **campaign clusters**, then fingerprints each cluster with shared infrastructure — URLs, phone numbers, money mentions — to build actionable intelligence about coordinated scam networks.

**Core result:** 25,411 messages across 5 languages → 123 distinct campaigns discovered, with a silhouette score of 0.7259 and an intra/inter cluster cosine similarity gap of 0.60.

---

## Key findings

- **123 scam campaigns** identified from 25,411 messages (EN, HI, TA, KN, FR)
- **Silhouette score 0.7259** vs 0.3744 for TF-IDF + DBSCAN baseline — 94% improvement
- **Intra-cluster similarity 0.96** vs inter-cluster 0.36 — strong campaign separation
- **Tamil–Kannada pairs** show highest cross-lingual similarity (0.738), consistent with shared Dravidian morphological structure
- **MiniLM outperforms LaBSE** (0.7259 vs 0.6766) for campaign-level clustering — semantic grouping matters more than retrieval alignment for this task
- **Full pipeline runs in 130 seconds** on 25k messages with a T4 GPU
- **287 inter-campaign edges** discovered via shared phone numbers and URLs, revealing coordinated multi-campaign operations

---

## Pipeline overview

```
Raw scam messages (5 languages)
        ↓
Multilingual encoder (paraphrase-multilingual-MiniLM-L12-v2)
        ↓
384-dimensional semantic embeddings
        ↓
FAISS cosine similarity index
        ↓
UMAP dimensionality reduction (384D → 50D for clustering, 2D for visualization)
        ↓
HDBSCAN clustering (min_cluster_size=30, min_samples=5)
        ↓
├── Entity extraction (regex + spaCy NER)
├── Campaign fingerprinting (URLs, phones, keywords)
└── Campaign relationship graph (shared-entity edges)
```

---

## Dataset

| Language | Code | Messages |
|---|---|---|
| Tamil | `ta` | 5,082 |
| Hindi | `hi` | 5,082 |
| English | `en` | 5,082 |
| Kannada | `kn` | 5,082 |
| French | `fr` | 5,083 |
| **Total** | | **25,411** |

The dataset is balanced across all 5 languages. Messages cover scam categories including lottery prizes, tax refunds, bank account verification, KYC updates, OTP phishing, and delivery parcel scams.

---

## Results

### Baseline comparison

| Method | Clusters | Noise % | Silhouette |
|---|---|---|---|
| **Ours (MiniLM + HDBSCAN)** | **123** | **8.4%** | **0.7259** |
| LaBSE + HDBSCAN | 138 | 8.9% | 0.6766 |
| English-only SBERT + HDBSCAN | 116 | 5.1% | N/A |
| TF-IDF + DBSCAN | 191 | 7.4% | 0.3744 |
| BERTopic | 493 | 16.8% | N/A |

### Cross-lingual similarity (same-campaign pairs)

| Language pair | Cosine similarity | Pairs |
|---|---|---|
| TA–KN | 0.7380 | 4 |
| EN–TA | 0.5793 | 5 |
| HI–KN | 0.5547 | 4 |
| HI–TA | 0.5404 | 7 |
| EN–FR | 0.5315 | 8 |
| EN–KN | 0.4667 | 4 |
| EN–HI | 0.3735 | 6 |
| Inter-cluster baseline | 0.3580 | — |

All pairs except EN–HI are clearly above the inter-cluster baseline, confirming cross-lingual campaign grouping. The EN–HI gap is a finding in itself — Hindi campaigns in this dataset appear to target different sub-topics than English ones.

### Scalability (T4 GPU)

| Messages | Total time | Encoding | UMAP | Campaigns found |
|---|---|---|---|---|
| 1,000 | 13.6s | 1.5s | 12.0s | 37 |
| 5,000 | 38.9s | 7.0s | 31.0s | 72 |
| 10,000 | 81.9s | 14.7s | 65.6s | 158 |
| 25,000 | 130.0s | 36.9s | 80.6s | 259 |

UMAP dominates runtime at scale (62% of total at 25k). Encoding is fast on GPU (28% of total). Campaign discovery scales sub-linearly — 25× more data produces only 7× more campaigns, suggesting real consolidation of scam operations.

---

## Setup

### Requirements

```
Python 3.9+
Google Colab with T4 GPU (recommended) or local GPU
```

### Install dependencies

```bash
pip install sentence-transformers faiss-cpu scikit-learn umap-learn hdbscan \
            bertopic spacy psutil deep-translator
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

### Run on your own data

Your dataset needs two columns: `text` and `language` (ISO 639-1 codes like `en`, `hi`, `ta`).

```python
import pandas as pd
df = pd.read_csv('your_dataset.csv')
# columns: text, language
```

Then open `scam_campaign_pipeline.ipynb` and run all cells in order. The notebook is self-contained and annotated at every step.

### Recommended: save embeddings after Cell 6

The encoding step takes 1–2 minutes on a T4. Save embeddings to Drive so you don't re-run if the session disconnects:

```python
import numpy as np
np.save('/content/drive/MyDrive/embeddings.npy', embedding_matrix)

# Reload next session instead of re-encoding:
# embedding_matrix = np.load('/content/drive/MyDrive/embeddings.npy')
```

---

## Outputs

After running the full pipeline you get three output files:

**`scam_messages_clustered.csv`** — original dataset with two new columns: `cluster` (raw HDBSCAN assignment, -1 = noise) and `cluster_final` (after nearest-centroid fallback for noise points).

**`campaign_fingerprints.json`** — one entry per cluster containing: message count, language distribution, top URLs, top phone numbers, money mentions, crypto wallet addresses, named organizations, keywords (translated to English), and sample messages.

**`human_eval_pairs.csv`** — 50 message pairs (25 same-campaign, 25 different-campaign, shuffled) generated for human annotation. Fill in the `annotator_1` and `annotator_2` columns and run Cell 30 to compute Cohen's Kappa inter-annotator agreement.

---

## Notebook structure

| Cell range | What it does |
|---|---|
| 1–6 | Imports, dataset loading, language distribution visualization |
| 7–12 | Multilingual encoding, normalization, cross-lingual sanity check |
| 13–15 | FAISS index, similarity search function |
| 16–21 | UMAP (50D + 2D), HDBSCAN grid search, final clustering |
| 22–25 | Cluster visualization, language mix analysis |
| 26–31 | Entity extraction, campaign fingerprinting, entity deduplication |
| 32–33 | Silhouette score, intra vs inter-cluster similarity |
| 34–36 | Cross-lingual evaluation, ablation study (multilingual vs English-only) |
| 37–40 | Baseline comparisons (TF-IDF+DBSCAN, BERTopic), summary table |
| 41–44 | Zero-shot scam type labeling, campaign relationship graph |
| 45–47 | Scalability benchmark, LaBSE comparison |
| 48–50 | Human evaluation sheet generation, Cohen's Kappa computation |
| 51 | Save outputs |

---

## Model choices

**Encoder: `paraphrase-multilingual-MiniLM-L12-v2`**

Chosen over LaBSE and English-only models after ablation. LaBSE is optimized for cross-lingual retrieval (finding exact translations) whereas MiniLM is optimized for semantic similarity (finding messages that mean the same thing). For campaign discovery, semantic grouping matters more than translation alignment — a Hindi lottery scam and an English lottery scam don't need to be exact translations of each other, they just need to represent the same scam type. MiniLM achieves 0.7259 silhouette vs LaBSE's 0.6766.

**Clustering: HDBSCAN over K-Means or DBSCAN**

K-Means requires a fixed number of clusters K — inappropriate when you don't know how many campaigns exist. DBSCAN has a single `eps` parameter that is sensitive and requires manual tuning. HDBSCAN finds variable-density clusters automatically, produces confidence scores per assignment, and outputs noise points (genuine outliers) rather than forcing everything into a cluster. The grid search over `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon` finds the best configuration for the data.

**Dimensionality reduction: UMAP before HDBSCAN**

Raw 384D embeddings are too high-dimensional for HDBSCAN to work well (the curse of dimensionality). UMAP to 50D preserves local structure while removing noise dimensions, producing significantly tighter clusters. The 2D UMAP is generated separately (different `min_dist`) purely for visualization and is never used for clustering.

---

## Limitations

- The EN–HI cross-lingual similarity (0.374) barely exceeds the inter-cluster baseline (0.358), suggesting the model struggles to align Hindi and English scam messages in this dataset. This may reflect genuine topical divergence between Hindi and English scam sub-types rather than a model limitation.
- spaCy NER is applied only to English and partially to Hindi/French via `xx_ent_wiki_sm`. Tamil and Kannada NER is not supported by any off-the-shelf spaCy model and falls back to regex-only extraction.
- The campaign graph edges depend on entity quality. Synthetic or placeholder phone numbers (e.g. `1234567890`) in the dataset create spurious cross-cluster links and should be filtered before graph analysis.
- Zero-shot scam type labeling using BART requires translating non-English samples to English first. Classification quality degrades when translation fails or produces awkward output.

---

## Potential extensions

- **Temporal analysis** — if message timestamps are available, track how campaigns evolve over time and detect new campaign emergence
- **LaBSE fine-tuning** — fine-tune LaBSE on a small set of labeled same-campaign pairs to improve cross-lingual alignment specifically for scam language
- **Active learning** — use cluster confidence scores from HDBSCAN to select the most uncertain messages for human review, reducing annotation cost
- **Real-time detection** — deploy the FAISS index as a live service: new incoming messages are embedded and assigned to the nearest existing campaign or flagged as a potential new one
- **Graph community detection** — apply Louvain or Leiden community detection on the campaign relationship graph to identify scam operator groups above the campaign level

---

## Citation

If you use this work, please cite:

```bibtex
@misc{scam_campaign_discovery_2025,
  title   = {Multilingual Scam Campaign Discovery via Semantic Clustering},
  author  = {[Kavya Garg]},
  year    = {2025},
  url     = {https://github.com/[Kavya-Garg003}
}
```

---

## License

MIT License. See `LICENSE` for details.

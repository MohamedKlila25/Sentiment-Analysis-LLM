# Sentiment Analysis with LLM Embeddings 🎭

**Groupe 10** - Projet Académique de Classification de Sentiments sur Twitter

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Dataset](#-dataset)
- [Architecture du Projet](#-architecture-du-projet)
- [Approches et Modèles](#-approches-et-modèles)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure des Fichiers](#-structure-des-fichiers)
- [Contributeurs](#-contributeurs)
- [Références](#-références)

## 🎯 Vue d'ensemble

Ce projet académique explore différentes approches pour la **classification de sentiments** sur des tweets en trois catégories : **positif**, **négatif** et **neutre**. Nous comparons des méthodes traditionnelles de machine learning, des réseaux de neurones MLP, et des modèles de langage pré-entraînés (LLMs) avec fine-tuning.

### Objectifs principaux

1. **Comparer** les performances de différentes approches de classification
2. **Évaluer** l'apport des embeddings BERT vs vectorisation classique (TF-IDF)
3. **Explorer** le fine-tuning léger (LoRA) de modèles Transformers
4. **Implémenter** une approche d'ensemble (voting) pour améliorer les performances

## 📊 Dataset

- **Source**: [Sentiment Analysis Dataset - Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- **Type**: Tweets avec métadonnées
- **Classes**: 3 catégories de sentiment
  - Négatif (0)
  - Neutre (1)
  - Positif (2)
- **Taille**: ~27,000 tweets d'entraînement
- **Features supplémentaires**: Time of Tweet, Age of User, données démographiques

## 🏗️ Architecture du Projet


```
 Prétraitement et exploration des données
    ├── Nettoyage et tokenization
    ├── Vectorisation TF-IDF
    └── Feature engineering
    
 Modèles classiques de ML
    ├── LinearSVC
    ├── Logistic Regression
    ├── Multinomial Naive Bayes
    └── Random Forest
    
Réseaux de neurones (MLP)
    ├── MLP sur TF-IDF
    └── MLP sur embeddings BERT
    
 Analyse comparative

 LLM pré-entraîné (DistilBERT)

 Analyse BERT et embeddings

 Fine-tuning avec LoRA
    ├── RoBERTa-base + LoRA
    ├── BERT-base + LoRA
    ├── DistilBERT + LoRA
    └── Ensemble Voting
```

##  Approches et Modèles

### 1. Machine Learning Classique

Vectorisation TF-IDF + modèles traditionnels :

| Modèle | Train F1 | Val F1 | Test F1 |
|--------|----------|--------|---------|
| LinearSVC | 0.725 | 0.713 | 0.714 |
| Logistic Regression | 0.730 | 0.709 | 0.711 |
| Multinomial NB | 0.669 | 0.660 | 0.661 |
| Random Forest | 0.972 | 0.697 | 0.696 |

**Meilleur modèle classique**: Logistic Regression


#### MLP sur TF-IDF
- **Architecture**: Réseau dense à plusieurs couches
- **Performance**: F1-score ~0.73 (similaire aux modèles classiques)

#### MLP sur Embeddings BERT
- **Simple Classifier** (1 couche linéaire): 65.17% accuracy
- **MLP Classifier** (3 couches): 67.23% accuracy
- **Amélioration**: Les embeddings BERT capturent mieux la sémantique

### 3. Fine-tuning avec LoRA

Utilisation de **LoRA (Low-Rank Adaptation)** pour un fine-tuning efficace :

| Modèle | Params entraînables | Train Loss | Val F1 | Test F1 | Test Accuracy |
|--------|---------------------|------------|--------|---------|---------------|
| **RoBERTa-base** | 1.48M (1.10%) | 0.4269 | 0.7913 | **0.7948** | 79.49% |
| **BERT-base** | 887K (0.80%) | 0.4436 | 0.7943 | **0.7947** | 79.46% |
| **DistilBERT** | 740K (1.09%) | 0.4859 | 0.7921 | **0.7860** | 78.58% |

**Configuration LoRA**:
- `r=8` (rank)
- `lora_alpha=16`
- `lora_dropout=0.1`
- Appliqué sur: `query` et `value` layers

### 4. Ensemble Voting 

Combinaison pondérée des 3 modèles LLM :

```python
Weights: RoBERTa (0.50) + BERT (0.35) + DistilBERT (0.15)
```

**Résultats Ensemble**:
- **F1-Score**: 0.8069 ( 1.21% vs meilleur modèle individuel)
- **Accuracy**: 80.67%

##  Résultats

### Comparaison Finale

```
┌─────────────────────────────┬───────────┬───────────┬───────────┐
│ Modèle                      │ Train F1  │  Val F1   │  Test F1  │
├─────────────────────────────┼───────────┼───────────┼───────────┤
│ LinearSVC (TF-IDF)          │   0.725   │   0.713   │   0.714   │
│ Logistic Regression         │   0.730   │   0.709   │   0.711   │
│ Random Forest               │   0.972   │   0.697   │   0.696   │
│ MLP (TF-IDF)                │   0.733   │   0.717   │   0.719   │
│ MLP (BERT embeddings)       │    -      │    -      │   0.672   │
│ RoBERTa + LoRA              │   0.821   │   0.791   │   0.795   │
│ BERT + LoRA                 │   0.820   │   0.794   │   0.795   │
│ DistilBERT + LoRA           │   0.811   │   0.792   │   0.786   │
│ ENSEMBLE (3 LLMs)           │    -      │    -      │   0.807   │
└─────────────────────────────┴───────────┴───────────┴───────────┘
```

### Observations Clés

1. **LLMs >> ML Classique**: Gain de ~8-9% en F1-score
2. **LoRA efficace**: Seulement 0.8-1.1% des paramètres entraînés
3. **Ensemble bénéfique**: +1.2% vs meilleur modèle seul
4. **RoBERTa légèrement meilleur**: Mais écart faible avec BERT

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA (recommandé pour GPU)
- 8GB+ RAM

### Étapes d'installation

```bash
# Cloner le repository
git clone https://github.com/votregroupe/sentiment-analysis-llm.git
cd sentiment-analysis-llm

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Requirements principaux

```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.12.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
nltk>=3.8
tqdm>=4.65.0
kagglehub
```

##  Utilisation

### 1. Télécharger le dataset

```python
import kagglehub
path = kagglehub.dataset_download('abhi8923shriv/sentiment-analysis-dataset')
print(f'Dataset téléchargé dans: {path}')
```

### 2. Exécuter le notebook

```bash
jupyter notebook Sentiment_Analysis_LLM_Embeddings_Groupe10.ipynb
```

### 3. Pipeline complet

Le notebook est organisé séquentiellement. Exécutez les cellules dans l'ordre pour :

1.  Prétraiter les données
2.  Entraîner les modèles classiques
3.  Tester les MLPs
4.  Fine-tuner les LLMs avec LoRA
5.  Créer l'ensemble et évaluer

### 4. Prédiction sur un nouveau texte

```python
# Exemple avec RoBERTa fine-tuné
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prédire
text = "I love this product, it works perfectly!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(predictions, dim=-1).item()

# Afficher le résultat
sentiments = {0: "Négatif", 1: "Neutre", 2: "Positif"}
print(f"Sentiment: {sentiments[sentiment]}")
print(f"Confiance: {predictions[0][sentiment]:.2%}")
```

##  Méthodologie

### Prétraitement du Texte

```python
def preprocess_text(text):
    # Nettoyage
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Mentions/Hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Ponctuation
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)
```

### Stratégie d'entraînement LoRA

```python
from peft import LoraConfig, get_peft_model

# Configuration LoRA
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,                # Scaling factor
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# Appliquer LoRA au modèle
model = get_peft_model(base_model, lora_config)

# Training
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
batch_size = 32
```

### Ensemble Voting

```python
def ensemble_prediction(text, models, tokenizers, weights):
    predictions = []
    
    for model, tokenizer in zip(models, tokenizers):
        inputs = tokenizer(text, return_tensors='pt', 
                          padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(probs)
    
    # Moyenne pondérée
    ensemble_probs = sum(w * p for w, p in zip(weights, predictions))
    return np.argmax(ensemble_probs)
```

##  Apprentissages Clés

### Pourquoi BERT embeddings > TF-IDF ?

1. **Représentation contextuelle**: BERT capture le sens en fonction du contexte
2. **Similarité sémantique**: Mots similaires ont des embeddings proches
3. **Pré-entraînement massif**: Connaissances générales du langage

### Avantages de LoRA

- **Efficacité mémoire**: <1% des paramètres à entraîner
- **Rapidité**: Fine-tuning 3-5x plus rapide
- **Performances**: Résultats comparables au fine-tuning complet
- **Flexibilité**: Plusieurs adaptateurs pour différentes tâches

##  Analyses Complémentaires

### Distribution des Classes

```
Classe Négative: ~28%
Classe Neutre:   ~40%
Classe Positive: ~32%
```

→ Dataset légèrement déséquilibré vers la classe neutre

### Matrice de Confusion (Ensemble)

```
              Prédiction
           Neg  Neu  Pos
Réel Neg   450   78   22
     Neu    65  512   73
     Pos    18   89  443
```

**Observations**:
- Meilleure précision sur les sentiments négatifs et positifs
- Classe neutre plus difficile (confusions avec pos/neg)

##  Contributeurs

- **Mahdi Abid**
- **Mohamed Amine Chaghal**
- **Mohamed Klila**

##  Références

### Papers

- Devlin et al. (2018) - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Liu et al. (2019) - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- Sanh et al. (2019) - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- Hu et al. (2021) - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Documentation

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Library](https://huggingface.co/docs/peft/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Datasets

- [Sentiment Analysis Dataset - Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)

##  Remerciements

- **Professeurs et encadrants** pour leurs conseils
- **Hugging Face** pour les modèles pré-entraînés et la bibliothèque Transformers
- **Kaggle** pour le dataset
- **Communauté PyTorch** pour les ressources éducatives

  
⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !



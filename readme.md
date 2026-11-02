# Projet 4 : Multi-Agent Social Choice & Preference Aggregation Engine

Ce projet implémente un système multi-agents modélisant une élection et le vote stratégique, développé dans le cadre du cours Systèmes Multi-Agents (SMA). Le système évalue plusieurs modes de scrutin (Pluralité, Deux tours, Vote Unique Transférable / STV, et Vote par Approbation) et compare le vote sincère au vote stratégique (tactique).

## 🛠 Prérequis et Dépendances

Ce projet utilise Python 3. Les bibliothèques externes requises pour exécuter les simulations, l'interface graphique et les benchmarks sont les suivantes :

* `numpy` (Calculs matriciels et lois de probabilité)

* `pandas` (Manipulation de données pour l'audit)

* `matplotlib` (Génération des graphiques de benchmark)

* `plotly` (Visualisations interactives pour l'interface web)

* `streamlit` (Framework pour l'interface graphique web)

* `requests` (Pour communiquer avec l'API locale d'Ollama)

### Services Externes (Agents LLM)

Pour exécuter les agents basés sur un LLM (Modèle de Langage), ce projet utilise **Ollama** en local, ce qui évite les coûts d'API et les limites de taux.

1. Installez Ollama depuis [ollama.com](https://ollama.com/).

2. Lancez le service Ollama en arrière-plan.

3. Téléchargez un modèle léger (par défaut, le code cherche `llama3.2` ou `mistral`) en ouvrant un terminal et en tapant :

   ```
   ollama pull llama3.2
   ```

*(Note : Si Ollama n'est pas détecté, le système basculera automatiquement sur un moteur cognitif d'utilité espérée de secours).*

## ⚙️ Installation

1. Clonez ce dépôt sur votre machine locale.

2. Créez un environnement virtuel (recommandé) :

   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances via le fichier requirements :

   ```
   pip install -r requirements.txt
   ```

## 🚀 Utilisation

Le projet propose trois modes d'exécution distincts.

### 1. Interface Graphique Interactive (Streamlit)

L'interface permet de générer des populations (par modèle spatial ou factions), de visualiser les sondages, d'exécuter des scrutins de manière unitaire et de comparer l'efficacité des modes de vote.

```
streamlit run app.py
```

*(Cela ouvrira automatiquement une fenêtre dans votre navigateur web).*

### 2. Benchmarks (Évaluations à grande échelle)

Ce script lance des simulations de Monte-Carlo pour tracer les courbes de perte d'utilité sociale en fonction du pourcentage d'électeurs stratégiques, et compare les agents statiques aux agents LLM. Les graphiques sont sauvegardés dans un dossier `outputs_benchmarks/`.

```
python benchmark.py
```

### 3. Mode Terminal (CLI Interactive)

Pour tester rapidement une élection mixte dans votre console sans interface graphique, exécutez le script principal :

```
python main.py
```

## 📁 Structure du Projet

* `app.py` : Interface web principale (Streamlit).

* `benchmark.py` : Suite d'évaluations et génération de graphiques.

* `main.py` : Interface terminale interactive.

* `agent.py` : Définition de l'électeur et de ses stratégies de vote.

* `population.py` : Génération de l'électorat (modèle spatial, lois de Dirichlet, factions).

* `voting_systems.py` : Algorithmes des 4 modes de scrutin et moteur d'audit.

* `llm_agents.py` : Intégration et prompt engineering pour la prise de décision des agents IA via Ollama.
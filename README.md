

# Face Wellbeing Model

## 🥷 | Members



| Name | Github |
|------|--------|
|Juan Felipe Raysz Muñoz|[@Sephir0ath](https://github.com/Sephir0ath)|
|Gabriela Antonia Muñoz Castillo|[@Nodercif](https://github.com/Nodercif)|
|Francisca Isidora Núñez Larenas|[@sshiro0](https://github.com/sshiro0)|
|Javier Alejandro Campos Contreras|[@4lehh](https://github.com/4lehh)|
|Oliver Isaías Peñailillo Sanzana|[@pyrrss](https://github.com/pyrrss)|

<div align="center">

## **Technologies used**

<a href="https://skillicons.dev">
  <img src="https://skillicons.dev/icons?i=git,github,vscode&perline=5" />
</a>

## **Programming languages and dependences**

<a href="https://skillicons.dev">
  <img src="https://skillicons.dev/icons?i=python,bash,sklearn,tensorflow&perline=5" />
</a>

</div>

## 🪢 | **Quick Start**

> [!IMPORTANT]
> Requires: Python 3.12

```sh
# Install dependences
pip install -r requirements.txt

# Execute code
python3 main.py
```

## 🤓 | **Parameter execution** 
> [!TIP]
> Use this mode if you don't want to use a default menu. It's not necessary to use this mode.

- **Flag** `--question`: Select one (1, 2, 3, 4 or 5). 
- **Flag** `--feature`: Add a specific feature (gaze, all features, etc).
- **Flag** `--label`: Select a specific label (depression or anxiety).
- **Flag** `--temporality`: Use temporality data or not (True or False).
- **Flag** `--model`: Select a specific machine learning model (RandomForestClassifier, GradientBoostingClassifier, LSTM, etc).
- **Flag** `--mode`: Select one if you want to apply Selection Feature, Dimensionality Reduction or Over Sampling (PCA, t-SNE, selection feature, SMOTE, pass).

```sh
# Example 
python3 main.py --question 1 --feature "All features" --label "anxiety" --temporality "False" --mode "PCA"
```

## ⏺️ | Structure

>[!IMPORTANT]
>The project structure should always be like this.

```sh
├───data
│   ├───Person1
│   │   ├───audio_features
│   │   ├───facial_features
│   │   ├───facial_features_1s_avg
│   │   ├───facial_features_avg
│   │   ├───facial_features_stats
│   │   └───transcripts_embeddings
│   ├───Person2
│   ├───...
│   └───labels.csv
├───src
│   ├───__init__.py
│   ├───data_extractor.py
│   ├───dimensionality_reduction.py
│   ├───feature_selection.py
│   ├───graph.py
│   ├───make_table.py
│   ├───manu.py
│   ├───models.py
│   ├───smote.py
│   └───structure.py
├───main.py
├───requirements.py
└───README.md
```
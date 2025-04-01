import streamlit as st
import plotly.express as px
from graphviz import Digraph
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_icon="💠",
)

# Dictionnaire des recommandations
recommandations = {
    "Automatisation": {
        "Texte": [("BERT", 70), ("Outils NLP", 30)],
        "Nombres": [("Régression logistique", 60), ("Arbres de décision", 40)],
        "Images": [("CNN", 80), ("Transfer Learning", 20)],
        "Séries temporelles": [("Moyenne mobile", 50), ("ARIMA", 50)]
    },
    "Précision": {
        "Texte": [("Transformers fine-tunés", 70), ("SVM avec features textuelles", 30)],
        "Nombres": [("XGBoost", 60), ("Deep Learning", 40)],
        "Images": [("ResNet", 70), ("EfficientNet", 30)],
        "Séries temporelles": [("LSTM", 60), ("Prophet", 40)]
    },
    "Découverte d'insights": {
        "Texte": [("LDA", 60), ("Word Embeddings", 40)],
        "Nombres": [("PCA", 50), ("t-SNE", 50)],
        "Images": [("Autoencodeurs", 70), ("GAN", 30)],
        "Séries temporelles": [("Détection d'anomalies", 60), ("Clustering temporel", 40)]
    },
    "Personnalisation": {
        "Texte": [("Recommandation basée sur le contenu", 70), ("Collaborative Filtering avec texte", 30)],
        "Nombres": [("Systèmes de recommandation avec embeddings", 60), ("K-Means", 40)],
        "Images": [("CNN pour features visuelles", 70), ("Similarité cosinus", 30)],
        "Séries temporelles": [("Modèles de prévision personnalisés", 50), ("Recommandation séquentielle", 50)]
    }
}

# Dictionnaire des descriptions
descriptions = {
    "BERT": "Modèle pré-entraîné pour les tâches NLP comme la classification.",
    "Outils NLP": "Outils pour tokenization, stemming, etc.",
    "Régression logistique": "Modèle linéaire pour classification rapide.",
    "Arbres de décision": "Modèle basé sur des règles.",
    "CNN": "Réseaux convolutionnels pour les images.",
    "Transfer Learning": "Réutilisation de modèles pré-entraînés.",
    "Moyenne mobile": "Lissage simple des séries temporelles.",
    "ARIMA": "Prévisions pour séries temporelles.",
    "Transformers fine-tunés": "Modèles ajustés pour précision.",
    "SVM avec features textuelles": "Classification précise sur texte.",
    "XGBoost": "Gradient boosting précis pour données tabulaires.",
    "Deep Learning": "Réseaux profonds pour relations complexes.",
    "ResNet": "Réseau résiduel pour images.",
    "EfficientNet": "Optimisé pour précision et efficacité.",
    "LSTM": "Réseaux récurrents pour séries temporelles.",
    "Prophet": "Prévisions faciles pour séries temporelles.",
    "LDA": "Topic modeling pour textes.",
    "Word Embeddings": "Représentations vectorielles des mots.",
    "PCA": "Réduction de dimension pour exploration.",
    "t-SNE": "Visualisation des données complexes.",
    "Autoencodeurs": "Apprentissage non supervisé pour insights.",
    "GAN": "Génération de données synthétiques.",
    "Détection d'anomalies": "Identification des événements inhabituels.",
    "Clustering temporel": "Regroupement des séries temporelles.",
    "Recommandation basée sur le contenu": "Suggestions basées sur le contenu.",
    "Collaborative Filtering avec texte": "Recommandations via préférences similaires.",
    "Systèmes de recommandation avec embeddings": "Personnalisation via embeddings.",
    "K-Means": "Segmentation des données.",
    "CNN pour features visuelles": "Extraction de caractéristiques visuelles.",
    "Similarité cosinus": "Mesure de similarité pour recommandations.",
    "Modèles de prévision personnalisés": "Prévisions adaptées à l'utilisateur.",
    "Recommandation séquentielle": "Suggestions basées sur des séquences."
}

# Interface utilisateur
st.title("Outil de recommandation de modèles d'IA")
st.write("""Sélectionnez vos besoins pour obtenir des recommandations de modèles d'IA. Cet outil, 
développé avec Streamlit, a un objectif résolument pédagogique : il vise à illustrer de manière concrète 
et interactive les propos avancés dans l’article sur l’impact de l’intelligence artificielle dans l’analyse de données.""")

with st.form("formulaire"):
    objectif = st.selectbox("Objectif principal", ["Automatisation", "Précision", "Découverte d'insights", "Personnalisation"])
    type_donnees = st.selectbox("Type de données", ["Texte", "Nombres", "Images", "Séries temporelles"])
    soumettre = st.form_submit_button("Obtenir des recommandations")

# Traitement de la soumission
if soumettre:
    modeles_scores = recommandations[objectif][type_donnees]
    modeles, scores = zip(*modeles_scores)
    
    # Création du graphique
    df = pd.DataFrame({"Modèle": modeles, "Pertinence": scores})
    fig = px.bar(df, x="Modèle", y="Pertinence", title="Modèles recommandés")
    st.plotly_chart(fig)
    fig = px.treemap(df, path=["Modèle"], values="Pertinence", title="Pertinence des modèles")
    st.plotly_chart(fig)
    # Visualisation de l’arbre de décision avec Graphviz
    dot = Digraph(comment="Arbre de décision pour le choix des modèles d’IA")
    dot.node("A", "Objectif principal")
    dot.node("B", f"{objectif}")
    dot.node("C", f"Type de données : {type_donnees}")
    dot.node("D", f"Recommandations : {', '.join([f'{m} ({s})' for m, s in modeles_scores])}")
    
    dot.edges(["AB", "BC", "CD"])
    
    # Afficher l’arbre dans Streamlit
    st.subheader("Arbre de décision")
    st.graphviz_chart(dot)
    
    # Sankey Diagram
    fig2 = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        node=dict(
            pad=50,
            thickness=5,
            line=dict(color = "black", width = 0.05),
            label=["Objectif", objectif, type_donnees] + list(modeles),
            color="blue"
        ),
        link=dict(
            source=[0, 1, 2, 2],  # Indices des nœuds source
            target=[1, 2, 3, 4],  # Indices des nœuds cible (modèles)
            value=[100, 100, scores[0], scores[1]],
            color="#F9F9F9"  # Flux proportionnels aux scores
        )
    )])
    fig2.update_layout(title_text="Flux de décision pour les modèles d’IA", font_size=20)
    st.plotly_chart(fig2)
    
    # Liste des modèles avec descriptions
    st.write("### Modèles recommandés :")
    for modele in modeles:
        with st.expander(modele):
            st.write(descriptions[modele])
    
    st.write("**Note :** Les scores (0-100) reflètent la pertinence relative des modèles pour votre cas.")

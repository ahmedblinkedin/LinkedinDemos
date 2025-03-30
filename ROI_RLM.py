import streamlit as st
import pandas as pd
import statsmodels.api as sm

# Titre et description
st.title("Calcul du ROI d'un projet data")
st.write("""
Cet outil utilise une régression linéaire multiple pour estimer l'impact de votre projet data sur les bénéfices nets et calculer le ROI.  
Vous pouvez :  
- **Saisir vos données manuellement** (préremplies avec un exemple sur 3 mois).  
- **Uploader un fichier CSV** pour analyser vos propres données.
""")
# Choix entre saisie manuelle ou upload CSV
option = st.radio("Choisissez une méthode pour entrer vos données :", 
                  ("Saisie manuelle", "Uploader un fichier CSV"))

if option == "Saisie manuelle":
    # Nombre de mois
    num_months = st.number_input("Nombre de mois", min_value=2, value=3)

    # Données de l'exemple préremplies (liste de dictionnaires pour chaque observation)
    default_data = [
        {"Benefices_Nets": 50000, "Investissement_Projet_Data": 10000, "Budget_Marketing": 20000, "Prix_Concurrents": 50, "Saison": 1},
        {"Benefices_Nets": 60000, "Investissement_Projet_Data": 12000, "Budget_Marketing": 25000, "Prix_Concurrents": 45, "Saison": 0},
        {"Benefices_Nets": 45000, "Investissement_Projet_Data": 8000, "Budget_Marketing": 18000, "Prix_Concurrents": 55, "Saison": 1}
    ]

    # Créer une liste pour stocker les données
    data = []
    for i in range(num_months):
        st.subheader(f"Mois {i+1}")
        # Utiliser les valeurs par défaut si i est dans l'exemple, sinon 0
        defaults = default_data[i] if i < len(default_data) else {
            "Benefices_Nets": 0.0, "Investissement_Projet_Data": 0.0, "Budget_Marketing": 0.0, 
            "Prix_Concurrents": 0.0, "Saison": 0
        }
        benefices = st.number_input(f"Bénéfices nets pour le mois {i+1}", value=float(defaults["Benefices_Nets"]))
        investissement = st.number_input(f"Investissement projet data pour le mois {i+1}", value=float(defaults["Investissement_Projet_Data"]))
        budget_marketing = st.number_input(f"Budget marketing pour le mois {i+1}", value=float(defaults["Budget_Marketing"]))
        prix_concurrents = st.number_input(f"Prix moyen des concurrents pour le mois {i+1}", value=float(defaults["Prix_Concurrents"]))
        saison = st.selectbox(f"Saison pour le mois {i+1}", options=[0, 1], index=defaults["Saison"])  # 0 = hors saison, 1 = saison
        data.append([benefices, investissement, budget_marketing, prix_concurrents, saison])

    # Convertir en DataFrame
    columns = ['Benefices_Nets', 'Investissement_Projet_Data', 'Budget_Marketing', 'Prix_Concurrents', 'Saison']
    df = pd.DataFrame(data, columns=columns)
# --- Partie 2 : Chargement d’un fichier CSV ---
else:
    st.subheader("Uploader un fichier CSV")
    st.write("""
    **Schéma attendu pour le fichier CSV** :  
    Votre fichier doit contenir au moins 2 lignes (observations) et les colonnes suivantes :  
    - `Benefices_Nets` : Les bénéfices nets (en euros ou toute unité monétaire).  
    - `Investissement_Projet_Data` : L’investissement dans le projet data (en euros).  
    - `Budget_Marketing` : Le budget marketing (en euros).  
    - `Prix_Concurrents` : Le prix moyen des concurrents (en euros ou unité pertinente).  
    - `Saison` : Une variable binaire (0 = hors saison, 1 = saison).
    Benefices_Nets,Investissement_Projet_Data,Budget_Marketing,Prix_Concurrents,Saison
    50000,10000,20000,50,1
    60000,12000,25000,45,0
    45000,8000,18000,55,1
    """)
    # Widget pour uploader le fichier CSV
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Charger le fichier CSV dans un DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['Benefices_Nets', 'Investissement_Projet_Data', 'Budget_Marketing', 'Prix_Concurrents', 'Saison']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Erreur : Les colonnes suivantes manquent dans votre fichier CSV : {missing_columns}")
        elif len(df) < 2:
            st.error("Erreur : Le fichier CSV doit contenir au moins 2 observations pour effectuer une régression.")
        else:
            st.write("Aperçu des données chargées :")
            st.dataframe(df)  # Afficher un aperçu du fichier chargé
    else:
        df = None
        st.info("Veuillez uploader un fichier CSV pour continuer.")

# --- Calcul du ROI ---
if st.button("Calculer le ROI") and df is not None:
    # Préparer les données pour la régression
    Y = df['Benefices_Nets']  # Variable dépendante
    X = df[['Investissement_Projet_Data', 'Budget_Marketing', 'Prix_Concurrents', 'Saison']]  # Variables explicatives
    X = sm.add_constant(X)  # Ajouter une constante pour l'intercept

    # Ajuster le modèle de régression
    try:
        model = sm.OLS(Y, X).fit()

        # Extraire le coefficient de l’investissement
        beta_1 = model.params['Investissement_Projet_Data']

        # Calculer les bénéfices attribuables au projet data
        investissement_total = df['Investissement_Projet_Data'].sum()
        benefices_attribuables = beta_1 * investissement_total

        # Calculer le ROI
        ROI = (benefices_attribuables / investissement_total) * 100 if investissement_total != 0 else 0

        # Afficher les résultats
        st.write(f"**Coefficient de l'investissement (β1)** : {beta_1:.4f}")
        st.write(f"**Bénéfices attribuables au projet** : {benefices_attribuables:.2f} €")
        st.write(f"**ROI du projet data** : {ROI:.2f}%")

        # Optionnel : afficher le résumé complet du modèle
        st.write("**Résumé du modèle** :")
        st.text(model.summary().as_text())
    except Exception as e:
        st.error(f"Erreur lors du calcul : {str(e)}")

else:
    if df is None and option == "Uploader un fichier CSV":
        st.warning("Veuillez uploader un fichier CSV avant de calculer.")

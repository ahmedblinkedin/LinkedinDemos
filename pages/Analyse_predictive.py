import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc as pm
import arviz as az
import seaborn as sns




# Fonction pour générer des données synthétiques
def generate_data(case, n_samples=100, noise_level=0.5):
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    if case == "linéaire":
        y = 2 * X + 3 + np.random.normal(0, noise_level, n_samples)
    elif case == "non_linéaire":
        y = 0.5 * X**2 - 0.1 * X**3 + np.random.normal(0, noise_level, n_samples)
    elif case == "hétéroscédastique":
        y = 2 * X + 3 + np.random.normal(0, noise_level * X, n_samples)
    elif case == "valeurs_extrêmes":
        y = 2 * X + 3 + np.random.normal(0, noise_level, n_samples)
        y[-10:] += np.random.normal(0, 10, 10)  # Ajout de valeurs extrêmes
    
    return X.reshape(-1, 1), y


# Sidebar pour les paramètres
st.sidebar.header("Paramètres des Régressions")
case = st.sidebar.selectbox("Type de données", 
                           ["linéaire", "non_linéaire", "hétéroscédastique", "valeurs_extrêmes"],
                           help="Choisissez le type de relation entre X et y")
n_samples = st.sidebar.slider("Nombre d'échantillons", 50, 500, 100)
noise_level = st.sidebar.slider("Niveau de bruit", 0.1, 2.0, 0.5)
n_iter = st.sidebar.slider("Nombre d'itérations MCMC", 500, 5000, 2000)

# Génération des données
X, y = generate_data(case, n_samples, noise_level)
df = pd.DataFrame({'X': X.flatten(), 'y': y})


# Titre de l'application
st.title("Visualisation de la Régression Linéaire : Quand ça marche et quand ça ne marche pas")

# Fonction pour générer et tracer les données
def plot_regression(X, y, title, subplot_pos):
    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calcul du R²
    r2 = r2_score(y, y_pred)
    
    # Visualisation
    plt.subplot(1, 2, subplot_pos)
    plt.scatter(X, y, color='blue', label='Données réelles')
    plt.plot(X, y_pred, color='red', label='Régression linéaire')
    plt.title(f"{title}\nR² = {r2:.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    

# Générer les données
np.random.seed(42)

# 1. Cas où la régression linéaire fonctionne bien (relation linéaire)
X_linear = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_linear = 2 * X_linear + 1 + np.random.normal(0, 1, (n_samples, 1))  # y = 2x + 1 + bruit

# 2. Cas où la régression linéaire fonctionne mal (relation non-linéaire)
X_nonlinear = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_nonlinear = np.sin(X_nonlinear) + np.random.normal(0, 0.1, (n_samples, 1))  # y = sin(x) + bruit

# Interface Streamlit
st.write("### Régression linéaire sur une relation linéaire")
st.write("Ici, les données suivent une tendance linéaire (y ≈ 2x + 1 + bruit). La régression linéaire est adaptée.")
plt.figure(figsize=(12, 5))
plot_regression(X_linear, y_linear, "Régression sur données linéaires", 1)


# Afficher le graphique
plt.tight_layout()
st.pyplot(plt)

# Explications supplémentaires
st.write("""
### Interprétation :
- **R² (coefficient de détermination)** : Mesure à quel point le modèle explique la variance des données. 
  - Proche de 1 : bon ajustement (cas linéaire).
  - Proche de 0 ou négatif : mauvais ajustement (cas non-linéaire).
- La régression linéaire suppose une relation linéaire entre X et Y. Si cette hypothèse n'est pas respectée, le modèle sera inefficace.
""")

# Titre de l'application
st.title("Visualisation de la Régression Logistique : Quand ça marche et quand ça ne marche pas")

# Fonction pour générer et tracer les données
def plot_logistic_regression(X, y, title, subplot_pos):
    # Entraînement du modèle
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # Création d'une grille pour visualiser la frontière de décision
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Visualisation
    plt.subplot(1, 2, subplot_pos)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', label='Données')
    plt.title(f"{title}\nAccuracy = {accuracy:.3f}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

# Générer les données
np.random.seed(42)

# 1. Cas où la régression logistique fonctionne bien (données linéairement séparables)
X_linear = np.vstack([
    np.random.normal(2, 1, (n_samples // 2, 2)),  # Classe 0
    np.random.normal(6, 1, (n_samples // 2, 2))   # Classe 1
])
y_linear = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# 2. Cas où la régression logistique fonctionne mal (données non linéairement séparables)
X_nonlinear = np.vstack([
    np.random.normal(0, 1, (n_samples // 2, 2)),  # Classe 0 au centre
    np.random.normal(0, 3, (n_samples // 2, 2))   # Classe 1 autour
])
y_nonlinear = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Interface Streamlit
st.write("### Exemple 1 : Régression logistique sur des données linéairement séparables")
st.write("Ici, les deux classes sont bien séparées par une frontière linéaire. La régression logistique est adaptée.")
plt.figure(figsize=(12, 5))
plot_logistic_regression(X_linear, y_linear, "Données linéairement séparables", 1)

st.write("### Exemple 2 : Régression logistique sur des données non linéairement séparables")
st.write("Ici, les classes sont mélangées de manière non linéaire (une classe entoure l'autre). La régression logistique échoue.")
plot_logistic_regression(X_nonlinear, y_nonlinear, "Données non linéairement séparables", 2)

# Afficher le graphique
plt.tight_layout()
st.pyplot(plt)

# Explications supplémentaires
st.write("""
### Interprétation :
- **Accuracy** : Proportion de prédictions correctes. 
  - Proche de 1 : bon ajustement (cas linéairement séparable).
  - Plus faible : mauvais ajustement (cas non linéairement séparable).
- La régression logistique suppose que les classes peuvent être séparées par une frontière linéaire dans l'espace des caractéristiques. Si cette hypothèse n'est pas respectée, le modèle sera inefficace.
- Pour les données non linéairement séparables, des modèles comme les SVM avec noyau ou les réseaux de neurones sont plus adaptés.
""")

# Titre de l'application
st.title("Visualisation de la Régression Polynomiale : Quand ça marche et quand ça ne marche pas")

# Fonction pour générer et tracer les données
def plot_polynomial_regression(X, y, degree, title, subplot_pos):
    # Transformation polynomiale
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédiction
    X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    y_pred = model.predict(X_poly)
    
    # Calcul du R²
    r2 = r2_score(y, y_pred)
    
    # Visualisation
    plt.subplot(1, 2, subplot_pos)
    plt.scatter(X, y, color='blue', label='Données réelles')
    plt.plot(X_smooth, y_pred_smooth, color='red', label=f'Régression polynomiale (deg={degree})')
    plt.title(f"{title}\nR² = {r2:.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

# Générer les données
np.random.seed(42)

# 1. Cas où la régression polynomiale fonctionne bien (relation quadratique)
X_poly = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_poly = 2 * X_poly**2 - 3 * X_poly + 1 + np.random.normal(0, 1, (n_samples, 1))  # y = 2x² - 3x + 1 + bruit

# 2. Cas où la régression polynomiale fonctionne mal (relation bruitée/non polynomiale)
X_noisy = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_noisy = np.sin(X_noisy) + np.random.normal(0, 2, (n_samples, 1))  # y = sin(x) + bruit fort

# Interface Streamlit
st.write("### Exemple 1 : Régression polynomiale sur une relation quadratique")
st.write("Ici, les données suivent une tendance quadratique (y ≈ 2x² - 3x + 1 + bruit). Une régression polynomiale de degré 2 est adaptée.")
plt.figure(figsize=(12, 5))
plot_polynomial_regression(X_poly, y_poly, 2, "Régression sur données quadratiques", 1)

st.write("### Exemple 2 : Régression polynomiale sur une relation bruitée/non polynomiale")
st.write("Ici, les données suivent une tendance sinusoïdale avec beaucoup de bruit (y ≈ sin(x) + bruit). Une régression polynomiale échoue à bien capturer la relation.")
plot_polynomial_regression(X_noisy, y_noisy, 2, "Régression sur données bruitées", 2)

# Afficher le graphique
plt.tight_layout()
st.pyplot(plt)

# Explications supplémentaires
st.write("""
### Interprétation :
- **R² (coefficient de détermination)** : Mesure à quel point le modèle explique la variance des données.
  - Proche de 1 : bon ajustement (cas quadratique).
  - Plus faible : mauvais ajustement (cas bruité/non polynomial).
- La régression polynomiale est efficace quand la relation entre X et Y peut être approximée par un polynôme (ex. quadratique, cubique).
- Elle échoue si la relation est fortement non polynomiale (ex. sinusoïdale) ou si le bruit domine le signal.
- Pour le second cas, des modèles comme les splines ou les séries de Fourier seraient plus adaptés.
""")

#st.set_page_config(page_title="Régression Bayésienne", layout="wide")
st.title("Visualisation des Modèles de Régression Bayésienne")





# Affichage des données
st.subheader("Visualisation des Données")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x='X', y='y', ax=ax)
ax.set_title(f"Données {case} (bruit={noise_level})")
st.pyplot(fig)

# Régression bayésienne avec pymc3
st.subheader("Modélisation Bayésienne avec PyMC3")

with st.expander("Détails du modèle PyMC3"):
    st.write("""
    Nous utilisons un modèle hiérarchique bayésien avec:
    - Prior normal pour les coefficients
    - Prior half-normal pour l'écart-type
    - Échantillonnage MCMC avec NUTS
    """)

try:
    with pm.Model() as bayesian_model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Relation linéaire
        mu = alpha + beta * X.flatten()
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sampling
        trace = pm.sample(n_iter, tune=1000, chains=2, return_inferencedata=True)
    
    # Affichage des résultats
    st.write(az.summary(trace, hdi_prob=0.95))
    
    try:
        # Create figure with proper layout
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle('MCMC Trace and Density Plots', y=1.02)
        
        # Define reference lines (example values - adjust based on your model)
        ref_lines = [
            ('alpha', {}, 3.0),  # Format: (var_name, {}, reference_value)
            ('beta', {}, 2.0),
            ('sigma', {}, 1.0)
        ]
        
        # Plot traces with reference lines
        az.plot_trace(
            trace,
            var_names=['alpha', 'beta', 'sigma'],
            lines=ref_lines,
            compact=True,
            combined=False,
            axes=axes,
            legend=True,
            plot_kwargs={'alpha': 0.8}  # Make lines slightly transparent
        )
        
        # Customize plot appearance
        for i, param in enumerate(['alpha', 'beta', 'sigma']):
            axes[i, 0].set_title(f'Trace for {param}')
            axes[i, 1].set_title(f'Density for {param}')
            axes[i, 0].set_ylabel('Sample value')
            axes[i, 1].set_ylabel('Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Trace plotting failed: {str(e)}")
        st.write("Attempting simplified plotting...")
        
        try:
            # Fallback to simple plotting without reference lines
            fig = az.plot_trace(
                trace,
                var_names=['alpha', 'beta', 'sigma'],
                compact=True,
                legend=True
            )
            st.pyplot(fig)
        except Exception as simple_error:
            st.error(f"Simplified plotting also failed: {str(simple_error)}")
            st.write("Showing numerical summary instead:")
            st.write(az.summary(trace, hdi_prob=0.95))
        
        except Exception as e:
            st.warning(f"Visualisation compacte échouée: {str(e)}")
        
        # Méthode alternative paramètre par paramètre
        for param in ['alpha', 'beta', 'sigma']:
            st.markdown(f"**{param}**")
            try:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                az.plot_trace(trace, var_names=[param], axes=axes)
                st.pyplot(fig)
            except:
                st.write(f"Impossible d'afficher les traces pour {param}")
        
        # Visualisation de la prédiction - VERSION CORRECTE
        st.subheader("Prédictions du Modèle")
    
    # Prédictions postérieures
    with bayesian_model:
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['y_obs'],
            random_seed=42,
            progressbar=True
        )
    
    # Extraction des prédictions
    if isinstance(ppc, dict) and 'y_obs' in ppc:
        # Format ancienne version PyMC3
        y_pred = ppc['y_obs'].mean(axis=0)
        y_hdi = az.hdi(ppc['y_obs'], hdi_prob=0.95)
    else:
        # Format nouvelle version PyMC
        ppc_samples = ppc.posterior_predictive['y_obs'].values
        ppc_samples = ppc_samples.reshape(-1, ppc_samples.shape[-1])
        y_pred = ppc_samples.mean(axis=0)
        y_hdi = az.hdi(ppc_samples, hdi_prob=0.95)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=X.flatten(), y=y, ax=ax, label='Données')
    sns.lineplot(x=X.flatten(), y=y_pred, ax=ax, color='red', label='Prédiction moyenne')
    ax.fill_between(X.flatten(), y_hdi[:, 0], y_hdi[:, 1], color='red', alpha=0.3, label='Intervalle crédible 95%')
    ax.set_title("Prédictions avec Intervalles Crédibles")
    ax.legend()
    st.pyplot(fig)
    
    # Calcul du RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    st.metric("RMSE", value=f"{rmse:.2f}")
    
    
except Exception as e:
    st.error(f"Erreur dans l'ajustement du modèle: {str(e)}")
    st.write("Le modèle bayésien rencontre des difficultés avec ce type de données.")


# Comparaison avec BayesianRidge de scikit-learn
st.subheader("Comparaison avec BayesianRidge (scikit-learn)")

with st.expander("À propos de BayesianRidge"):
    st.write("""
    BayesianRidge est une implémentation simplifiée de la régression bayésienne qui:
    - Utilise des priors conjugués (normaux-gamma)
    - Estime les paramètres par maximisation de la vraisemblance marginale
    - Est plus rapide mais moins flexible que les approches MCMC
    """)

try:
    # Ajustement du modèl
    br = BayesianRidge(compute_score=True)
    br.fit(X, y)
    y_pred_br = br.predict(X)
    
    # Affichage des coefficients
    coefs = pd.DataFrame({
        'Paramètre': ['alpha', 'beta', 'sigma'],
        'Valeur': [br.intercept_, br.coef_[0], np.sqrt(1/br.alpha_)]
    })
    st.write(coefs)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=X.flatten(), y=y, ax=ax, label='Données')
    sns.lineplot(x=X.flatten(), y=y_pred_br, ax=ax, color='green', label='BayesianRidge')
    ax.set_title("Prédictions avec BayesianRidge")
    ax.legend()
    st.pyplot(fig)
    
    # Calcul du RMSE
    rmse_br = np.sqrt(mean_squared_error(y, y_pred_br))
    st.metric("RMSE (BayesianRidge)", value=f"{rmse_br:.2f}")
    
except Exception as e:
    st.error(f"Erreur avec BayesianRidge: {str(e)}")

# Interprétation des résultats
st.subheader("Interprétation")
if case == "linéaire":
    st.success("""
    🎯 **Le modèle bayésien fonctionne bien** pour des données linéaires avec bruit homoscédastique.
    - Les intervalles crédibles couvrent bien les données
    - Les paramètres sont bien estimés
    - Les diagnostics MCMC montrent une bonne convergence
    """)
elif case == "non_linéaire":
    st.warning("""
    ⚠️ **Le modèle linéaire bayésien est mal adapté** aux relations non-linéaires.
    - La relation sous-jacente est quadratique/cubique
    - Un modèle polynomial bayésien serait plus approprié
    - Les intervalles crédibles ne capturent pas bien la variabilité
    """)
elif case == "hétéroscédastique":
    st.warning("""
    ⚠️ **Le modèle standard suppose une variance constante** (homoscédasticité).
    - Pour des données hétéroscédastiques, envisagez:
        - Une modélisation explicite de la variance
        - Une transformation des données
        - Une famille de distribution différente
    """)
elif case == "valeurs_extrêmes":
    st.warning("""
    ⚠️ **Les valeurs extrêmes influencent fortement** le modèle bayésien gaussien.
    - Envisagez une distribution à queues plus épaisses (Student-t)
    - Ou un modèle robuste aux outliers
    """)

# Conclusion
st.markdown("""
## Quand utiliser la régression bayésienne?

✅ **Fonctionne bien quand:**
- Petits échantillons (utilisation efficace de l'information)
- Incertitude à quantifier (intervalles crédibles)
- Information a priori disponible
- Relations linéaires ou légèrement non-linéaires

❌ **Moins adapté quand:**
- Très grandes bases de données (temps de calcul)
- Relations hautement non-linéaires (sans transformations)
- Bruit complexe (hétéroscédasticité, outliers)
- Sans information a priori pertinente
""")


st.title("Visualisation des Modèles de Moyennes Mobiles (MA)")
st.markdown("""
Ce démonstrateur montre quand les modèles de Moyennes Mobiles (MA) fonctionnent bien 
et quand ils échouent dans l'analyse prédictive.
""")

# Sidebar controls
st.sidebar.header("Paramètres Moyennes Mobiles")
ma_order = st.sidebar.selectbox("Ordre du modèle MA (q)", [1, 2, 3, 4, 5])
ma_window_size = st.sidebar.slider("Taille de la fenêtre pour MA simple", 3, 50, 10)
ma_n_samples = st.sidebar.slider("Nombre d'échantillons", 50, 500, 200)
ma_noise_level = st.sidebar.slider("Niveau de bruit ", 0.1, 2.0, 0.5)

# Generate MA time series
def generate_ma_process(order, ma_n_samples, ma_noise_level):
    # Coefficients MA - nous utilisons des valeurs qui créent un processus stationnaire
    ma_params = np.array([0.5] * order)
    ar_params = np.array([])
    
    # Générer le processus MA
    ma_process = ArmaProcess.from_coeffs(ar_params, ma_params)
    ma_series = ma_process.generate_sample(nsample=ma_n_samples, scale=ma_noise_level)
    
    # Ajouter une tendance pour voir quand MA échoue
    trend = np.linspace(0, 5, ma_n_samples)
    ma_series_with_trend = ma_series + trend
    
    return ma_series, ma_series_with_trend

# Generate data
ma_series, ma_series_with_trend = generate_ma_process(ma_order, ma_n_samples, ma_noise_level)

# Create DataFrame
dates = pd.date_range(start="2020-01-01", periods=ma_n_samples, freq="D")
df = pd.DataFrame({
    "Date": dates,
    "MA_Process": ma_series,
    "MA_Process_With_Trend": ma_series_with_trend,
    "White_Noise": np.random.normal(0, ma_noise_level, ma_n_samples)
})

# Add simple moving average
df['MA_Simple'] = df['MA_Process'].rolling(window=ma_window_size).mean()
df['MA_Simple_Trend'] = df['MA_Process_With_Trend'].rolling(window=ma_window_size).mean()
df['MA_Simple_Noise'] = df['White_Noise'].rolling(window=ma_window_size).mean()

# Plotting function
def plot_series(series_df, cols, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cols:
        ax.plot(series_df['Date'], series_df[col], label=col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Tabs for different scenarios
tab1, tab2, tab3 = st.tabs([
    "Cas idéal pour MA", 
    "MA avec tendance (problème)", 
    "Bruit blanc (MA inutile)"
])

with tab1:
    st.header("Cas idéal pour les Moyennes Mobiles")
    st.markdown("""
    **Quand MA fonctionne bien**: 
    - Lorsque la série temporelle est stationnaire (pas de tendance, variance constante)
    - Quand les chocs passés ont un effet décroissant sur les valeurs futures
    - Pour lisser le bruit et révéler les tendances sous-jacentes
    """)
    plot_series(df, ['MA_Process', 'MA_Simple'], 
               "Processus MA pur avec Moyenne Mobile (fonctionne bien)")
    
    st.markdown("""
    **Analyse**:
    - La moyenne mobile suit bien le processus MA sous-jacent
    - Elle permet de lisser les fluctuations tout en capturant la dynamique
    """)

with tab2:
    st.header("Quand les Moyennes Mobiles échouent (série avec tendance)")
    st.markdown("""
    **Problèmes**:
    - Les MA simples retardent la détection des tendances
    - Elles sous-estiment systématiquement les valeurs récentes en présence de tendance
    - Nécessitent des ajustements (différenciation) pour fonctionner correctement
    """)
    plot_series(df, ['MA_Process_With_Trend', 'MA_Simple_Trend'], 
               "Processus MA avec tendance (MA simple échoue)")
    
    st.markdown("""
    **Solution possible**:
    - Utiliser un modèle intégrant la différenciation (ARIMA)
    - Ou enlever d'abord la tendance avant d'appliquer MA
    """)

with tab3:
    st.header("Bruit blanc - où MA n'apporte rien")
    st.markdown("""
    **Pourquoi MA échoue**:
    - Le bruit blanc n'a aucune structure temporelle
    - La moyenne mobile ne fait que rajouter du délai sans améliorer la prédiction
    - La meilleure prédiction est simplement la moyenne globale
    """)
    plot_series(df, ['White_Noise', 'MA_Simple_Noise'], 
               "Bruit blanc (MA inutile)")
    
    st.markdown("""
    **Analyse**:
    - La moyenne mobile ne fait que lisser le bruit sans révéler de pattern
    - La variance est réduite mais au prix d'un décalage temporel
    - Aucun avantage prédictif dans ce cas
    """)

# Add theoretical explanation
st.sidebar.markdown("""
## Théorie des Modèles MA
Un modèle de Moyenne Mobile (MA) exprime la variable actuelle comme une combinaison linéaire des termes d'erreur passés.

Pour un modèle MA(q):
$$ X_t = \\mu + \\epsilon_t + \\theta_1\\epsilon_{t-1} + ... + \\theta_q\\epsilon_{t-q} $$

Où:
- $\\mu$ est la moyenne
- $\\epsilon_t$ est le bruit blanc
- $\\theta_i$ sont les paramètres du modèle
""")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot


st.title("Visualisation des Modèles ARIMA et SARIMA")
st.markdown("""
Cette application montre les performances des modèles ARIMA et SARIMA dans différents scénarios temporels.
""")

# Sidebar controls
st.sidebar.header("Paramètres ARIMA/SARIMA")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Paramètres Non Saisonniers")
    p = st.slider("Ordre AR (p)", 0, 3, 1)
    d = st.slider("Ordre d'intégration (d)", 0, 2, 1)
    q = st.slider("Ordre MA (q)", 0, 3, 1)

with col2:
    st.subheader("Paramètres Saisonniers")
    P = st.slider("Ordre AR saisonnier (P)", 0, 2, 0)
    D = st.slider("Ordre d'intégration saisonnier (D)", 0, 1, 0)
    Q = st.slider("Ordre MA saisonnier (Q)", 0, 2, 0)
    m = st.slider("Période saisonnière (m)", 1, 24, 12)

n_samples = st.sidebar.slider("Nombre d'échantillons", 100, 1000, 200)
noise_level = st.sidebar.slider("Niveau de bruit ", 0.1, 1.0, 0.3)
seasonal_amplitude = st.sidebar.slider("Amplitude saisonnière", 0.5, 5.0, 2.0)

# Generate synthetic time series
def generate_time_series_data(n_samples, noise_level, seasonal_amplitude):
    # Base stochastic component
    ar_params = np.array([0.7]*p + [0.0]*(3-p))[:p] if p > 0 else np.array([])
    ma_params = np.array([0.5]*q + [0.0]*(3-q))[:q] if q > 0 else np.array([])
    
    process = ArmaProcess.from_coeffs(ar_params, ma_params)
    stationary = process.generate_sample(n_samples, scale=noise_level)
    
    # Non-stationary series
    non_stationary = np.cumsum(stationary) if d > 0 else stationary.copy()
    
    # Series with deterministic trend
    trend = np.linspace(0, 10, n_samples)
    with_trend = non_stationary + trend
    
    # Random walk
    random_walk = np.cumsum(np.random.normal(0, noise_level, n_samples))
    
    # Seasonal series
    seasonal = stationary + seasonal_amplitude*np.sin(np.linspace(0, 20*np.pi, n_samples))
    
    # Seasonal with trend
    seasonal_trend = seasonal + trend
    
    return {
        'stationary': stationary,
        'non_stationary': non_stationary,
        'with_trend': with_trend,
        'random_walk': random_walk,
        'seasonal': seasonal,
        'seasonal_trend': seasonal_trend
    }

# Generate data
data = generate_time_series_data(n_samples, noise_level, seasonal_amplitude)
dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
df = pd.DataFrame(data, index=dates)

# Train-test split
train_size = int(n_samples * 0.7)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Model fitting functions
def fit_arima(series, order):
    try:
        model = ARIMA(series, order=order)
        results = model.fit()
        forecast = results.get_forecast(steps=len(test))
        return forecast.predicted_mean, forecast.conf_int(), results
    except Exception as e:
        st.error(f"Erreur ARIMA: {str(e)}")
        return None, None, None

def fit_sarima(series, order, seasonal_order):
    try:
        model = SARIMAX(series, 
                       order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(test))
        return forecast.predicted_mean, forecast.conf_int(), results
    except Exception as e:
        st.error(f"Erreur SARIMA: {str(e)}")
        return None, None, None

# Plotting functions
def plot_series(actual, predicted, conf_int, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual.index, actual, label='Réel', color='blue')
    if predicted is not None:
        ax.plot(predicted.index, predicted, label='Prédiction', color='red')
        if conf_int is not None:
            ax.fill_between(conf_int.index, 
                           conf_int.iloc[:, 0], 
                           conf_int.iloc[:, 1], 
                           color='pink', alpha=0.3)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_diagnostics(results):
    if results is None:
        return
        
    fig = plt.figure(figsize=(12, 8))
    layout = (2, 2)
    ax1 = plt.subplot2grid(layout, (0, 0))
    ax2 = plt.subplot2grid(layout, (0, 1))
    ax3 = plt.subplot2grid(layout, (1, 0))
    ax4 = plt.subplot2grid(layout, (1, 1))
    
    residuals = pd.Series(results.resid, index=train.index[:len(results.resid)])
    residuals.plot(ax=ax1, title='Résidus')
    ax1.axhline(0, color='red', linestyle='--')
    
    plot_acf(residuals, ax=ax2, title='ACF des Résidus')
    
    residuals.plot(kind='kde', ax=ax3, title='Densité des Résidus')
    
    from statsmodels.graphics.gofplots import qqplot
    qqplot(residuals, line='s', ax=ax4)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_seasonal_decomposition(series, period):
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        result = seasonal_decompose(series, model='additive', period=period)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        result.observed.plot(ax=ax1, title='Observé')
        result.trend.plot(ax=ax2, title='Tendance')
        result.seasonal.plot(ax=ax3, title='Saisonnalité')
        result.resid.plot(ax=ax4, title='Résidus')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur dans la décomposition: {str(e)}")

# Tabs for different scenarios
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Stationnaire", 
    "Non-Stationnaire", 
    "Avec Tendance", 
    "Marche Aléatoire", 
    "Saisonnier",
    "Saisonnier+Tendance"
])

with tab1:
    st.header("Cas Stationnaire - ARIMA fonctionne bien")
    pred, conf_int, results = fit_arima(train['stationary'], (p, d, q))
    plot_series(df['stationary'], pred, conf_int, "Série Stationnaire - ARIMA")
    
    if results:
        st.write(results.summary())
        mse = mean_squared_error(test['stationary'], pred)
        st.metric("MSE (Test)", f"{mse:.4f}")
        plot_diagnostics(results)

with tab2:
    st.header("Série Non-Stationnaire - ARIMA avec intégration")
    pred, conf_int, results = fit_arima(train['non_stationary'], (p, d, q))
    plot_series(df['non_stationary'], pred, conf_int, "Série Non-Stationnaire - ARIMA")
    
    if results:
        mse = mean_squared_error(test['non_stationary'], pred)
        st.metric("MSE (Test)", f"{mse:.4f}")

with tab3:
    st.header("Série avec Tendance - ARIMA standard vs Différenciation")
    pred_arima, conf_int_arima, _ = fit_arima(train['with_trend'], (p, d, q))
    pred_sarima, conf_int_sarima, _ = fit_sarima(
        train['with_trend'], 
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0)  # No seasonal component
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['with_trend'], label='Réel', color='blue')
    if pred_arima is not None:
        ax.plot(pred_arima.index, pred_arima, label='ARIMA', color='red')
    if pred_sarima is not None:
        ax.plot(pred_sarima.index, pred_sarima, label='ARIMA avec Diff', color='green')
    ax.set_title("Comparaison ARIMA standard vs Différenciation")
    ax.legend()
    st.pyplot(fig)
    
    st.warning("Pour les tendances fortes, la différenciation est cruciale (d=1 ou 2)")

with tab4:
    st.header("Marche Aléatoire - ARIMA(0,1,0)")
    pred, conf_int, results = fit_arima(train['random_walk'], (0, 1, 0))
    plot_series(df['random_walk'], pred, conf_int, "Marche Aléatoire - ARIMA(0,1,0)")
    
    st.subheader("Analyse des Lags")
    fig, ax = plt.subplots(figsize=(8, 8))
    lag_plot(df['random_walk'], lag=1, ax=ax)
    st.pyplot(fig)

with tab5:
    st.header("Données Saisonnières - SARIMA requis")
    st.subheader("Décomposition Saisonnière")
    plot_seasonal_decomposition(df['seasonal'], period=m)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ARIMA Standard")
        pred_arima, conf_int_arima, _ = fit_arima(train['seasonal'], (p, d, q))
        plot_series(df['seasonal'], pred_arima, conf_int_arima, "ARIMA Standard")
        if pred_arima is not None:
            mse = mean_squared_error(test['seasonal'], pred_arima)
            st.metric("MSE ARIMA", f"{mse:.4f}")
    
    with col2:
        st.subheader("SARIMA")
        pred_sarima, conf_int_sarima, results_sarima = fit_sarima(
            train['seasonal'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, m)
        )
        plot_series(df['seasonal'], pred_sarima, conf_int_sarima, f"SARIMA({P},{D},{Q},{m})")
        if pred_sarima is not None:
            mse = mean_squared_error(test['seasonal'], pred_sarima)
            st.metric("MSE SARIMA", f"{mse:.4f}")
            if results_sarima:
                st.write(results_sarima.summary())

with tab6:
    st.header("Saisonnalité + Tendance - SARIMA optimal")
    st.subheader("Décomposition Saisonnière")
    plot_seasonal_decomposition(df['seasonal_trend'], period=m)
    
    st.subheader("Comparaison des Modèles")
    pred_arima, _, _ = fit_arima(train['seasonal_trend'], (p, d, q))
    pred_sarima, _, results_sarima = fit_sarima(
        train['seasonal_trend'],
        order=(p, d, q),
        seasonal_order=(P, D, Q, m)
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['seasonal_trend'], label='Réel', color='blue')
    if pred_arima is not None:
        ax.plot(pred_arima.index, pred_arima, label='ARIMA', color='red')
    if pred_sarima is not None:
        ax.plot(pred_sarima.index, pred_sarima, label='SARIMA', color='green')
    ax.set_title("Comparaison ARIMA vs SARIMA")
    ax.legend()
    st.pyplot(fig)
    
    if pred_arima is not None and pred_sarima is not None:
        col1, col2 = st.columns(2)
        with col1:
            mse = mean_squared_error(test['seasonal_trend'], pred_arima)
            st.metric("MSE ARIMA", f"{mse:.4f}")
        with col2:
            mse = mean_squared_error(test['seasonal_trend'], pred_sarima)
            st.metric("MSE SARIMA", f"{mse:.4f}")
    
    st.success("SARIMA capture à la fois la saisonnalité et la tendance!")

# ACF/PACF analysis
st.sidebar.subheader("Analyse des Corrélations")
selected_series = st.sidebar.selectbox("Série pour ACF/PACF", list(df.columns))

if selected_series:
    st.subheader(f"Analyse ACF/PACF pour {selected_series}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df[selected_series], ax=ax1, lags=40)
    plot_pacf(df[selected_series], ax=ax2, lags=40, method='ywm')
    st.pyplot(fig)

# Model explanation
st.sidebar.markdown("""
## Formulation ARIMA/SARIMA

**ARIMA(p,d,q)**:
$$ (1 - \sum_{i=1}^p \phi_i L^i) (1 - L)^d X_t = (1 + \sum_{i=1}^q \theta_i L^i) \epsilon_t $$

**SARIMA(p,d,q)(P,D,Q)[m]**:
$$ \Phi_P(L^m) \phi_p(L) (1 - L^m)^D (1 - L)^d X_t = \Theta_Q(L^m) \theta_q(L) \epsilon_t $$

Où:
- $L$: opérateur retard
- $m$: période saisonnière
- $\phi$: paramètres AR
- $\theta$: paramètres MA
""")







st.title("Visualisation des Arbres de Décision en Analyse Prédictive")
st.markdown("""
Cette application montre quand les arbres de décision fonctionnent bien et quand ils échouent.
""")

# Sidebar controls
st.sidebar.header("Paramètres du Modèle d'arbres de décision")
problem_type = st.sidebar.radio("Type de problème", ["Classification", "Régression"])
max_depth = st.sidebar.slider("Profondeur max de l'arbre", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min samples pour split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min samples par feuille", 1, 10, 1)

# Generate synthetic data based on problem type
def generate_data(problem_type):
    if problem_type == "Classification":
        # Créer 3 types de données de classification
        X_linear, y_linear = make_classification(
            n_samples=500, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, flip_y=0.1, class_sep=1.0, random_state=42
        )
        
        X_nonlinear, y_nonlinear = make_classification(
            n_samples=500, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=42
        )
        
        X_xor, y_xor = make_classification(
            n_samples=500, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, flip_y=0.1, class_sep=1.0, random_state=42
        )
        # Modifier pour créer un problème XOR
        X_xor = np.random.randn(500, 2)
        y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
        
        return {
            "Linéairement Séparable": (X_linear, y_linear),
            "Non Linéaire": (X_nonlinear, y_nonlinear),
            "Relation XOR": (X_xor, y_xor)
        }
    else:
        # Créer 3 types de données de régression
        X_linear, y_linear = make_regression(
            n_samples=500, n_features=1, noise=20, random_state=42
        )
        
        X_nonlinear, y_nonlinear = make_regression(
            n_samples=500, n_features=1, noise=10, random_state=42
        )
        y_nonlinear = y_nonlinear + 100 * np.sin(X_nonlinear[:, 0] * 2)
        
        X_multi, y_multi = make_regression(
            n_samples=500, n_features=2, n_informative=2, noise=30, random_state=42
        )
        
        return {
            "Relation Linéaire": (X_linear, y_linear),
            "Relation Non Linéaire": (X_nonlinear, y_nonlinear),
            "Multiples Variables": (X_multi, y_multi)
        }

# Generate data
data = generate_data(problem_type)

# Train model and plot results
def train_and_plot(X, y, scenario_name, problem_type):
    st.subheader(f"Scénario: {scenario_name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    if problem_type == "Classification":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    else:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    if problem_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.text(classification_report(y_test, y_pred))
        
        # Plot decision boundary
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = ListedColormap(['#FFAAAA', '#AAFFAA'])
        
        if X.shape[1] == 2:
            DecisionBoundaryDisplay.from_estimator(
                model, X, cmap=cmap, alpha=0.8, ax=ax, response_method="predict"
            )
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
            ax.set_title(f"Frontière de décision (Accuracy={accuracy:.2f})")
        else:
            ax.scatter(X[:, 0], y, c=y, edgecolor='k', s=20)
            ax.set_title("Données de classification")
        
        st.pyplot(fig)
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        
        # Plot predictions vs actual
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if X.shape[1] == 1:
            # Trier pour une visualisation propre
            sorted_idx = np.argsort(X_test[:, 0])
            ax.plot(X_test[sorted_idx, 0], y_test[sorted_idx], 'o', label='Vraies valeurs')
            ax.plot(X_test[sorted_idx, 0], y_pred[sorted_idx], '-r', label='Prédictions')
            ax.set_xlabel("Feature")
            ax.set_ylabel("Target")
        else:
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
            ax.set_xlabel("Vraies valeurs")
            ax.set_ylabel("Prédictions")
        
        ax.set_title("Prédictions vs Vraies valeurs")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Plot the tree
    st.subheader("Visualisation de l'arbre de décision")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model, 
        filled=True, 
        feature_names=[f"Feature {i}" for i in range(X.shape[1])],
        class_names=[str(c) for c in np.unique(y)] if problem_type == "Classification" else None,
        ax=ax
    )
    st.pyplot(fig)

# Main app
tab1, tab2 = st.tabs(["Cas où ça marche bien", "Cas où ça marche mal"])

with tab1:
    st.header("Cas où les arbres de décision fonctionnent bien")
    if problem_type == "Classification":
        train_and_plot(*data["Linéairement Séparable"], "Données Linéairement Séparables", problem_type)
    else:
        train_and_plot(*data["Relation Linéaire"], "Relation Linéaire Simple", problem_type)

with tab2:
    st.header("Cas où les arbres de décision ont des difficultés")
    if problem_type == "Classification":
        train_and_plot(*data["Relation XOR"], "Relation XOR (Non Linéaire Complexe)", problem_type)
    else:
        train_and_plot(*data["Relation Non Linéaire"], "Relation Non Linéaire Complexe", problem_type)

# Explanation
st.sidebar.markdown("""
## Quand utiliser les arbres de décision?

**Fonctionnent bien quand:**
- Relations non linéaires entre features et target
- Données avec interactions complexes
- Features à échelles différentes
- Données avec valeurs manquantes
- Features catégorielles et numériques mélangées

**Fonctionnent mal quand:**
- Relations linéaires simples (modèles linéaires plus efficaces)
- Extrapolation hors du domaine d'entraînement
- Données avec beaucoup de bruit
- Problèmes où la relation est XOR-like
- Besoin de modèles petits et interprétables (arbres profonds deviennent complexes)
""")
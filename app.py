import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Prédiction des besoins en eau - Maroc", layout="wide")
st.title("Modèle de prédiction des besoins en eau au Maroc (2000-2023)")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'Année': range(2000, 2024),
        'Population totale (millions)': [28.7, 29.1, 29.5, 29.9, 30.3, 30.7, 31.0, 31.4, 31.8, 32.2, 
                                        32.6, 33.1, 33.5, 33.9, 34.3, 34.8, 35.2, 35.6, 36.1, 36.5, 
                                        36.9, 37.3, 37.6, 37.9],
        'Population urbaine (millions)': [16.3, 16.7, 17.0, 17.3, 17.6, 17.9, 18.2, 18.5, 18.7, 19.0,
                                         19.4, 19.7, 20.0, 20.3, 20.6, 20.9, 21.2, 21.5, 21.9, 22.2,
                                         22.5, 22.8, 23.1, 23.4],
        'PIB (USD, milliards)': [40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100,
                                104, 108, 112, 116, 120, 123, 126, 129],
        'Eau agricole (millions m³/an)': [13000, 14040, 15163.2, 16376.26, 17686.36, 19101.26, 20629.37,
                                         22279.72, 24062.09, 25987.06, 28066.02, 30311.31, 32736.21,
                                         35355.11, 38183.52, 41238.2, 44237.25, 48100.23, 51948.25,
                                         56104.11, 60592.44, 65439.84, 70675.03, 76329.03],
        'Eau industrie (millions m³/an)': [800, 832, 865.28, 899.89, 935.89, 973.32, 1012.26, 1052.75,
                                          1094.86, 1138.65, 1184.2, 1231.56, 1280.83, 1332.06, 1385.34,
                                          1440.75, 1498.38, 1558.32, 1620.65, 1685.48, 1752.9, 1823.01,
                                          1895.94, 1971.77],
        'Eau domestique (millions m³/an)': [1000, 1055, 1113.02, 1174.24, 1238.82, 1306.96, 1378.84,
                                            1454.68, 1534.69, 1619.09, 1708.14, 1802.09, 1901.21, 2005.77,
                                            2116.09, 2232.48, 2355.26, 2484.8, 2621.47, 2765.65, 2917.76,
                                            3078.23, 3247.54, 3426.15]
    })
    data['Eau totale (millions m³/an)'] = data['Eau agricole (millions m³/an)'] + \
                                         data['Eau industrie (millions m³/an)'] + \
                                         data['Eau domestique (millions m³/an)']
    return data

data = load_data()

# Sidebar - Paramètres du modèle
st.sidebar.header("Paramètres du modèle")
test_size = st.sidebar.slider("Pourcentage des données pour le test", 10, 40, 20)
n_estimators = st.sidebar.slider("Nombre d'arbres dans la forêt", 50, 500, 100)

# Séparation des données
X = data[['Population totale (millions)', 'Population urbaine (millions)', 'PIB (USD, milliards)']]
y = data['Eau totale (millions m³/an)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
st.header("Performance du modèle")
col1, col2 = st.columns(2)
col1.metric("Erreur absolue moyenne (MAE)", f"{mae:,.0f} millions m³/an")
col2.metric("Score R²", f"{r2:.2%}")

# Visualisation des résultats
st.header("Visualisation des prédictions")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel("Valeurs réelles")
ax.set_ylabel("Prédictions")
ax.set_title("Comparaison des valeurs réelles et prédites")
st.pyplot(fig)

# Importance des caractéristiques
st.header("Importance des variables")
feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(x='Importance', y='Variable', data=feature_importance, ax=ax2)
ax2.set_title("Importance des variables dans la prédiction")
st.pyplot(fig2)

# Prédiction interactive
st.header("Prédiction personnalisée")
col1, col2, col3 = st.columns(3)
pop_totale = col1.number_input("Population totale (millions)", min_value=20.0, max_value=50.0, value=35.0)
pop_urbaine = col2.number_input("Population urbaine (millions)", min_value=10.0, max_value=30.0, value=20.0)
pib = col3.number_input("PIB (USD, milliards)", min_value=30, max_value=200, value=100)

if st.button("Prédire la consommation d'eau"):
    prediction = model.predict([[pop_totale, pop_urbaine, pib]])[0]+12000
    st.success(f"Consommation d'eau prévue: {prediction :,.0f} (±4,500) millions m³/an")
    
    # Détail par secteur
    st.subheader("Répartition par secteur (estimation)")
    total_known = data['Eau totale (millions m³/an)'].iloc[-1]
    agri_ratio = data['Eau agricole (millions m³/an)'].iloc[-1] / total_known
    ind_ratio = data['Eau industrie (millions m³/an)'].iloc[-1] / total_known
    dom_ratio = data['Eau domestique (millions m³/an)'].iloc[-1] / total_known
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agriculture", f"{prediction * agri_ratio:,.0f} millions m³/an")
    col2.metric("Industrie", f"{prediction * ind_ratio:,.0f} millions m³/an")
    col3.metric("Usage domestique", f"{prediction * dom_ratio:,.0f} millions m³/an")

# Affichage des données
st.header("Données historiques")
st.dataframe(data)

# Téléchargement des prédictions
if st.button("Exporter les prédictions"):
    data['Prédiction'] = model.predict(X)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les prédictions au format CSV",
        data=csv,
        file_name="predictions_eau_maroc.csv",
        mime="text/csv"
    )

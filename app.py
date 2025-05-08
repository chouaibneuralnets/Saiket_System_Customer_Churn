import numpy as np
import streamlit as st
import joblib

# Charger le modèle préalablement sauvegardé
model = joblib.load('my_model.pkl')

# Fonction pour effectuer la prédiction
def predict(input_data):
    # Conversion des colonnes catégorielles en numériques (0, 0.5, 1, etc.)
    # ChargeGroup et TenureGroup sont déjà sous forme de catégories avec des valeurs spécifiques
    charge_group_encoded = input_data['charge_group']  # Valeur numérique déjà (ex: 0, 0.5, 1)
    tenure_group_encoded = input_data['tenure_group']  # Valeur numérique déjà (ex: 0, 0.5, 1)
    
    # Organiser les données dans un tableau 1D (par exemple, une ligne avec 18 colonnes)
    data = np.array([
        input_data['gender'],
        input_data['senior_citizen'],
        input_data['partner'],
        input_data['dependents'],
        input_data['phone_service'],
        input_data['multiple_lines'],
        input_data['internet_service'],
        input_data['online_security'],
        input_data['online_backup'],
        input_data['device_protection'],
        input_data['tech_support'],
        input_data['streaming_tv'],
        input_data['streaming_movies'],
        input_data['contract'],  # Cette variable prend des valeurs comme 0, 0.5, 1
        input_data['paperless_billing'],
        input_data['payment_method'],
        input_data['total_charges'],
        charge_group_encoded,
        tenure_group_encoded
    ]).reshape(1, -1)  # Redimensionner en matrice 2D (1 échantillon, 18 caractéristiques)
    
    # Faire la prédiction avec le modèle
    prediction = model.predict(data)
    
    return prediction

# Interface Streamlit
def main():
    st.title("Prédiction de Churn")

    # Créer des entrées pour les variables
    gender = st.selectbox('Genre', [0, 1], help="0 = Femme, 1 = Homme")
    senior_citizen = st.selectbox('Senior', [0, 1], help="0 = Non, 1 = Oui")
    partner = st.selectbox('Partenaire', [0, 1], help="0 = Non, 1 = Oui")
    dependents = st.selectbox('Dépendants', [0, 1], help="0 = Non, 1 = Oui")
    phone_service = st.selectbox('Service Téléphonique', [0, 1], help="0 = Non, 1 = Oui")
    multiple_lines = st.number_input('Lignes Multiples', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    internet_service = st.number_input('Service Internet', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    online_security = st.number_input('Sécurité en ligne', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    online_backup = st.number_input('Sauvegarde en ligne', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    device_protection = st.number_input('Protection des appareils', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    tech_support = st.number_input('Support Technique', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    streaming_tv = st.number_input('TV en Streaming', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    streaming_movies = st.number_input('Films en Streaming', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    contract = st.number_input('Contrat', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    paperless_billing = st.selectbox('Facturation sans papier', [0, 1], help="0 = Non, 1 = Oui")
    payment_method = st.number_input('Méthode de Paiement', min_value=0.0, max_value=1.0, step=0.5, help="Valeur entre 0, 0.5, 1")
    total_charges = st.number_input('Total des Charges', min_value=0.0, help="Montant total des charges")
    charge_group = st.number_input('Groupe de Charge', min_value=0.0, max_value=2.0, step=0.5, help="0, 0.5, 1, 1.5, 2")
    tenure_group = st.number_input('Groupe de Tenure', min_value=0.0, max_value=2.0, step=0.5, help="0, 0.5, 1, 1.5, 2")

    # Créer un dictionnaire pour les données d'entrée
    input_data = {
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'total_charges': total_charges,
        'charge_group': charge_group,
        'tenure_group': tenure_group
    }

    # Bouton pour prédiction
    if st.button('Faire la Prédiction'):
        prediction = predict(input_data)
        st.write(f"Prédiction du churn : {prediction[0]}")

if __name__ == '__main__':
    main()

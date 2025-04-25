import pandas as pd
import io
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import pickle
import joblib
import matplotlib.pyplot as plt
import time


st.set_page_config(layout="wide")
# Custom HTML/CSS for the banner
custom_html = """
<img src="https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg" width="1000" height="626" alt="Journée mondiale de l'enfant Joyeux enfants multiculturels souriants du monde entier" class="$inline-block $h-auto $w-full $overflow-hidden $object-contain $relative" sizes="(max-width: 480px) 100vw, (min-aspect-ratio: 626/626) 695px, (max-width: 1096px) calc(100vw - 40px), calc(100vw - 540px)" srcset="https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=360 360w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=740 740w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=826 826w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=900 900w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=996 996w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=1060 1060w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=1380 1380w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=1480 1480w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=1800 1800w, https://img.freepik.com/photos-premium/journee-mondiale-enfant-joyeux-enfants-multiculturels-souriants-du-monde-entier_259293-20079.jpg?w=2000 2000w" fetchpriority="high" style="max-width: 1000px; max-height: 695px;">
"""
# Display the custom HTML
st.components.v1.html(custom_html)

def main():
    st.title("Application Machine Learning : Étude du Bonheur dans le Monde")
    st.sidebar.title("Auteur:")
    st.sidebar.markdown("<h5>Brice MOUNGENGUI MYADJI</h5>", unsafe_allow_html=True)
   

    st.sidebar.title("Sommaire")

    pages = ["Exploration", "Visualisation des données","Tests statistiques", "Pre-traitement des données","Modelisation", "Conclusion"]
    page = st.sidebar.radio("Allez vers", pages)

    # Fonction d'importation des données
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("data_happiness.csv")
        data.drop("Unnamed: 0", axis=1, inplace=True)
        data["Year"] = pd.to_datetime(data["Year"], format="%Y")
        return data

    # Initialisation de df pour qu'il soit accessible dans tout le code
    df = load_data()
    selected_columns = [
            "Ladder score",
            "Fertility_Rate",
            "Underweight_Rate_Percent",
            "Life_Expectancy",
            "Logged GDP per capita",
            "Social support",
            "Urban_Population_Percent",
            "Water_Access_Percent",
            "Freedom to make life choices",
            "Sanitary_Expense_Per_Capita",
            "Perceptions of corruption"
        ]
    
        # Créer un nouveau DataFrame avec les colonnes sélectionnées
    filtered_data = df[selected_columns]
    columns_with_outliers = [
            'Fertility_Rate', 'Social support', 'Underweight_Rate_Percent',
            'Water_Access_Percent', 'Sanitary_Expense_Per_Capita', 
            'Perceptions of corruption', 'Freedom to make life choices'
        ]



    if page == pages[0]:
        st.write("### Introduction")
        st.subheader(
            "Découvrez mon projet innovant qui explore les facteurs influençant le bonheur à l'échelle mondiale grâce à l'intelligence artificielle. "
            "Nous analysons des données provenant de sources diverses, telles que des enquêtes sur le bien-être, des indicateurs économiques et des données sociales. "
            "Notre objectif est de comprendre comment différents éléments interagissent et influencent le bonheur des individus dans différentes cultures et régions."
        )

        # Affichage de la table des données
        df_sample = df.sample(100)
        
        if st.sidebar.checkbox("Affichage des données brutes", False):
            st.subheader("Affichage des 100 échantillons du dataframe")
            st.write(df)
            st.subheader("Dimension du dataset")
            st.write(df.shape)
            st.subheader("Information du dataset")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.subheader("Affichage du describe des variables numériques")
            Num_col = df.select_dtypes(include=["float64", "int64"]).columns
            cat_col = df.select_dtypes(include=["object"]).columns
            st.dataframe(df[Num_col].describe())

            if st.checkbox("Show NAs"):
                st.dataframe(df.isna().sum())
                st.subheader("Suppression des NAs")
                for col in df.columns:
                    if df[col].dtype == "float":
                        df[col] = df[col].fillna(df[col].median())
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].fillna(df[col].mode()[0])

            if st.sidebar.checkbox("Suppression des NAs"):
                st.dataframe(df.isna().sum())

    if page == pages[1]:
        st.write("### Visualisation des données")

        # Sélection du type de graphique
        chart_type = st.selectbox("Choisissez le type de graphique :", ["Barres", "Scatter", "Box", "Heatmaps"])

        # Sélection des variables à afficher
        variables = df.columns.tolist()
        x_var = st.selectbox("Choisissez la variable X :", variables)
        y_var = st.selectbox("Choisissez la variable Y :", variables)
        z_var = st.selectbox("Choisissez la variable Z :", variables)

        # Demander à l'utilisateur s'il souhaite utiliser la variable année (w)
        use_w_var = st.checkbox("Voulez-vous utiliser une variable année (w) ?")

        if use_w_var:
            w_var = st.selectbox("Choisissez la variable année w :", variables)
        else:
            w_var = None  # ou une valeur par défaut si nécessaire

        # Création du graphique en fonction du type sélectionné
        if chart_type == "Barres":
            fig = px.bar(df, x=x_var, y=y_var, color=z_var, animation_frame=w_var, title=f"Graphique à barres de {y_var} vs {x_var}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_var, y=y_var, color=z_var, animation_frame=w_var, title=f"Graphique de dispersion de {y_var} vs {x_var}")
        elif chart_type == "Box":
            fig = px.box(df, y=y_var, title=f"Graphique de dispersion de {y_var}")
        elif chart_type == "Heatmaps":
            correlation_matrix = df.corr(numeric_only=True)
            fig = px.imshow(correlation_matrix, title='Heatmap des Corrélations', color_continuous_scale='Viridis')

        # Affichage du graphique
        st.plotly_chart(fig)

        # Option pour réinitialiser
        if st.button("Réinitialiser"):
            st.experimental_rerun()

    if page == pages[2]:
        st.write("### Tests statistiques")
    
    # Remplacement des valeurs manquantes
        for column in filtered_data.columns:
            if filtered_data[column].dtype == 'float64':  # Pour les colonnes de type float
                median_value = filtered_data[column].median()
                filtered_data[column].fillna(median_value, inplace=True)
            elif filtered_data[column].dtype == 'object':  # Pour les colonnes de type objet
             mode_value = filtered_data[column].mode()[0]
             filtered_data[column].fillna(mode_value, inplace=True)

     # Initialisation d'un tableau pour les résultats
        results = pd.DataFrame(columns=['Variable', 'F-statistic', 'p-value'])

     # Test statistique (ANOVA)
        for column in filtered_data.columns[1:]:
            f_stat, p_value = stats.f_oneway(df['Ladder score'], filtered_data[column])
            new_row = pd.DataFrame({
                'Variable': [column],
                'F-statistic': [f"{f_stat:.2e}"],  # Format en écriture scientifique
                'p-value': [f"{p_value:.2e}"]       # Format en écriture scientifique
                })
            results = pd.concat([results, new_row], ignore_index=True)

     # Ajout d'une colonne pour indiquer si p-value < 0.05
        results['p-value < 0.05'] = results['p-value'].apply(lambda x: float(x) < 0.05)

     # Demande à l'utilisateur s'il souhaite afficher les résultats
        afficher_resultats = st.checkbox("Afficher les résultats")

     # Affichage des résultats si l'utilisateur a coché la case
        if afficher_resultats:
         st.write(results)
    if page == pages[3]:
        st.write("### Pré-traitement des données")

       
        
        # Calcul des quartiles et de l'IQR
        Q1 = filtered_data[columns_with_outliers].quantile(0.25)
        Q3 = filtered_data[columns_with_outliers].quantile(0.75)
        IQR = Q3 - Q1  # Détermination des outliers sur IQR

        # Définition des bornes inférieure et supérieure
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identification des outliers
        outliers = filtered_data[((filtered_data[columns_with_outliers] < lower_bound) | 
                                  (filtered_data[columns_with_outliers] > upper_bound)).any(axis=1)]

        # Suppression des outliers
        filtered_data_1 = filtered_data[~((filtered_data[columns_with_outliers] < lower_bound) | 
                                          (filtered_data[columns_with_outliers] > upper_bound)).any(axis=1)]
        
        # Afficher les données filtrées
        if st.checkbox("Afficher les Données Filtrées"):
            st.subheader("Données Filtrées")
            st.write(filtered_data)

        # Normalisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(filtered_data)

        # Convertir le tableau normalisé en DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)

        # Afficher les données normalisées
        if st.checkbox("Afficher les Données Normalisées"):
            st.subheader("Données Normalisées")
            st.write(scaled_df)

    

    if page == pages[4]:
     st.write("### Modélisation")
    
    # Remplacer les NAs par la médiane pour les variables float et par le mode pour les variables object
     for column in filtered_data.columns:
         if filtered_data[column].dtype == 'float64':
            median_value = filtered_data[column].median()
            filtered_data[column].fillna(median_value, inplace=True)
         elif filtered_data[column].dtype == 'object':
            mode_value = filtered_data[column].mode()[0]
            filtered_data[column].fillna(mode_value, inplace=True)

    # Sélectionner la variable cible
     target_variable = st.selectbox("Sélectionnez la variable cible", filtered_data.columns)
     X = filtered_data.drop(target_variable, axis=1)
     y = filtered_data[target_variable]

    # Diviser les données
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choix du modèle
     model_options = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "XGBoost": XGBRegressor()
     }

     selected_model_name = st.selectbox("Choisissez un modèle", list(model_options.keys()))
     model = model_options[selected_model_name]

     # Choix des hyperparamètres
     hyperparam_choice = st.radio("Voulez-vous lancer le modèle avec des hyperparamètres ?", ("Sans hyperparamètres", "Avec hyperparamètres"))

     # Hyperparamètres pour GridSearchCV
     param_grid = {
        "Linear Regression": {},
        "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5]},
        "Extra Trees": {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5]},
        "XGBoost": {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0], 'learning_rate': [0.01, 0.1, 0.2]}
     }

     # GridSearchCV ou ajustement direct
     if hyperparam_choice == "Avec hyperparamètres":
        grid_search = GridSearchCV(model, param_grid[selected_model_name], cv=3, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write("Meilleurs paramètres trouvés : ", grid_search.best_params_)
     else:
        model.fit(X_train, y_train)
        best_model = model

     # Prédictions
     y_train_pred = best_model.predict(X_train)
     y_test_pred = best_model.predict(X_test)

     # Début du temps d'exécution
     start_time = time.time()

     # Simuler un traitement (remplacez cela par votre code réel)
     time.sleep(2) # Simule un délai de traitement

     # Fin du temps d'exécution
     end_time = time.time()
     execution_time = end_time - start_time

     # Affichage du temps d'exécution
     st.write("Temps d'exécution : ", execution_time, "secondes")


     # Calcul des métriques
     mse_train = mean_squared_error(y_train, y_train_pred)
     mse_test = mean_squared_error(y_test, y_test_pred)
     r2_train = r2_score(y_train, y_train_pred)
     r2_test = r2_score(y_test, y_test_pred)

     # Affichage des résultats
     st.write("### Résultats de la modélisation")
     st.write(f"MSE (Train): {mse_train:.4f}")
     st.write(f"MSE (Test): {mse_test:.4f}")
     st.write(f"R² (Train): {r2_train:.4f}")
     st.write(f"R² (Test): {r2_test:.4f}")

       # Enregistrement des résultats dans un tableau
     if 'results_summary' not in st.session_state:
        st.session_state.results_summary = pd.DataFrame(columns=[
            'Modèle', 'MSE (Train)', 'MSE (Test)', 'R² (Train)', 'R² (Test)', 'Différence R² (Train - Test)', 'Hyperparamètres'
            ])

     new_results = pd.DataFrame({
        'Modèle': [selected_model_name],
        'MSE (Train)': [mse_train],
        'MSE (Test)': [mse_test],
        'R² (Train)': [r2_train],
        'R² (Test)': [r2_test],
        'Différence R² (Train - Test)': [r2_train - r2_test],
        'Hyperparamètres': [hyperparam_choice]  # Ajout de la colonne pour indiquer le choix des hyperparamètres

        })

     st.session_state.results_summary = pd.concat([st.session_state.results_summary, new_results], ignore_index=True)

     # Affichage du tableau récapitulatif
     st.write("### Tableau récapitulatif des résultats")
     st.dataframe(st.session_state.results_summary)

     
     # Affichage des courbes des métriques
     fig, ax = plt.subplots(1, 2, figsize=(12, 5))

     # Courbes pour les données d'entraînement
     ax[0].scatter(y_train, y_train_pred, alpha=0.5)
     ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')  # Ligne de référence
     ax[0].set_title('Prédictions sur les données d\'entraînement')
     ax[0].set_xlabel('Vérité terrain (Train)')
     ax[0].set_ylabel('Prédictions (Train)')

     # Courbes pour les données de test
     ax[1].scatter(y_test, y_test_pred, alpha=0.5)
     ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ligne de référence
     ax[1].set_title('Prédictions sur les données de test')
     ax[1].set_xlabel('Vérité terrain (Test)')
     ax[1].set_ylabel('Prédictions (Test)')

     st.pyplot(fig)

     # Importance des caractéristiques
     if selected_model_name in ["Random Forest", "Extra Trees", "XGBoost"]:
        feature_importance = best_model.feature_importances_
        feature_names = X.columns

        # Créer un DataFrame pour trier les caractéristiques par importance
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig_importance, ax_importance = plt.subplots()
        ax_importance.barh(importance_df['Feature'], importance_df['Importance'])
        ax_importance.set_title('Feature Importances')
        st.pyplot(fig_importance)


        # Option pour afficher l'image
        show_image = st.checkbox("Afficher shape du meilleurs model")

        # Chemin de l'image
        image_path = "Shap Value (impact on model output).png"  # Remplacez par le chemin de votre image

        if show_image:
           st.image(image_path, caption="Voici votre image", use_column_width=True)

     # Sauvegarder le meilleur modèle
     model_filename = 'best_model.pkl'
     with open(model_filename, 'wb') as file:
         pickle.dump(best_model, file)

     # Afficher un message de confirmation
     st.write(f"Le meilleur modèle a été sauvegardé sous le nom : {model_filename}")

     # Optionnel : Charger et afficher le modèle sauvegardé
     loaded_model = joblib.load(model_filename)
     st.write("Modèle chargé avec succès : ", loaded_model)

     # Ajuster les longueurs si nécessaire
     if len(y_train) != len(y_train_pred):
      min_length_train = min(len(y_train), len(y_train_pred))
      y_train = y_train[:min_length_train]
      y_train_pred = y_train_pred[:min_length_train]

     if len(y_test) != len(y_test_pred):
      min_length_test = min(len(y_test), len(y_test_pred))
      y_test = y_test[:min_length_test]
      y_test_pred = y_test_pred[:min_length_test]

     # Créer un DataFrame avec les valeurs réelles et prédites pour l'entraînement
     results_df_train = pd.DataFrame({
        'Valeurs réelles (Train)': y_train,
        'Valeurs prédites (Train)': y_train_pred
        })

     # Créer un DataFrame avec les valeurs réelles et prédites pour le test
     results_df_test = pd.DataFrame({
        'Valeurs réelles (Test)': y_test,
        'Valeurs prédites (Test)': y_test_pred
        })

     # Afficher le DataFrame des résultats pour l'entraînement
     st.write("### Résultats des prédictions - Entraînement")
     st.dataframe(results_df_train)

     # Afficher le DataFrame des résultats pour le test
     st.write("### Résultats des prédictions - Test")
     st.dataframe(results_df_test)


    if page == pages[5]:
     st.write("### Conclusion")

     st.subheader("Difficultés rencontrées :")
     st.write(
     "1) Gestion de la volumétrie des données et optimisation des temps de calcul.\n"
     "2) Traitement des valeurs manquantes, nécessitant une approche innovante"
     
      )

     st.write("")  # Ajoute un espace vide

     st.subheader("Points forts du projet :")
     st.write(
     "1) Fusion et enrichissement avancé des données pour une meilleure granularité et des explications plus fines.\n"
     "2) Visualisations pertinentes et lisibles, facilitant l’interprétation des résultats.\n"
     "3) Modèle performant et explicable, offrant des prédictions fiables et exploitables."
     )

     st.write("")  # Ajoute un espace vide

     st.subheader("Perspectives d’amélioration :")
     st.write(
     "1) Test de modèles plus avancés ( Pipeline , réseaux de neurones) pour affiner les prédictions.\n"
     "2) Automatisation de la mise à jour des données en temps réel à partir de sources officielles.\n"
     "3) Intégration des indicateurs liés aux crises économiques et climatiques pour mieux contextualiser les variations du bonheur."
      )

     

if __name__ == '__main__':
    main()

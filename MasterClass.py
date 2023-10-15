from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from io import StringIO
import requests
from sklearn.preprocessing import LabelEncoder

class ClassificationModels:
    def __init__(self,année):
        self.année = année
        url = "https://static.data.gouv.fr/resources/trajets-realises-en-covoiturage-registre-de-preuve-de-covoiturage/20230921-111347/" + self.année + ".csv"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Gérer les erreurs HTTP

            # Lire le CSV directement en DataFrame sans passer par StringIO
            self.df = pd.read_csv(StringIO(response.text), sep=";")
            # Filtrer les données pour Rouen
            data = self.filter_data()
            columns_to_drop = ['trip_id', 'journey_id', 'passenger_seats',"journey_start_date","journey_start_time","journey_end_date","journey_end_time","journey_start_country","journey_end_country"]
            data=data.drop(columns=columns_to_drop,axis=1)

        except requests.exceptions.HTTPError as http_err:
            print(f"Erreur HTTP lors du téléchargement du fichier CSV : {http_err}")
        except Exception as err:
            print(f"Une erreur s'est produite : {err}")
        self.X = data.drop('has_incentive', axis=1)
        self.y = data['has_incentive']
        label = LabelEncoder()
        self.y= label.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        self.results = {}
    def train_models(self):
        for model_name, model in self.models.items():
            # Définissez une grille de recherche aléatoire pour chaque modèle
            param_dist = {}
            if model_name == 'Random Forest':
                param_dist = {
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False],
                }
            elif model_name == 'SVM':
                param_dist = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
                }

            # Créez un objet RandomizedSearchCV pour le modèle actuel
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1
            )

            # Fit le modèle avec la recherche aléatoire des hyperparamètres
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_

            # Évaluez le modèle
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            self.results[model_name] = {
                'best_model': best_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

            print(f"{model_name} Accuracy: {accuracy:.2f}")
            print(f"{model_name} Precision: {precision:.2f}")
            print(f"{model_name} Recall: {recall:.2f}")
            print(f"{model_name} F1 Score: {f1:.2f}")

    def compare_models(self):
        best_model = max(self.results, key=lambda model: self.results[model]['accuracy'])
        print(f"The best model is {best_model} with accuracy {self.results[best_model]['accuracy']:.2f}")

    def filter_data(self):
        try:
            rouen_data = self.df[(self.df['journey_start_town'] == 'Rouen') | (self.df['journey_end_town'] == 'Rouen')]
            rouen_data.to_csv('rouen_data.csv', index=False)
            object_columns = rouen_data.select_dtypes(include=['object'])
            # Créez une instance de LabelBinarizer
            label = LabelEncoder()
            # Appliquez-la à une colonne catégorielle et convertissez les booléens en entiers
            for col in object_columns:
                rouen_data.loc[:, col] = label.fit_transform(rouen_data[col])
            return rouen_data
        except Exception as e:
            print(f"Une erreur s'est produite lors de la filtration des données : {e}")

# Utilisation de la classe
if __name__ == "__main__":
    # data est votre DataFrame
    classification = ClassificationModels("2023-08")
    classification.train_models()
    classification.compare_models()

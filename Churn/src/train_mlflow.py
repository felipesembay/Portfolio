import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clipper de outliers baseado em quantis"""
    
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.q_low_ = X.quantile(self.lower)
        self.q_high_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.q_low_, self.q_high_, axis=1)


def load_and_prepare_data(data_path):
    """Carrega e prepara os dados"""
    df = pd.read_csv(data_path)
    
    # Colunas para remover
    cols_to_drop = [
        "Customer ID",
        "Customer Status",
        "Churn Label",
        "Churn Score",
        "Churn Category",
        "Churn Reason",
        "CLTV",
        "Zip Code",
        "Latitude",
        "Longitude",
        "Country",
        "State",
        "City"
    ]
    
    # Target
    y = (df["Churn Label"] == "Yes").astype(int)
    X = df.drop(columns=cols_to_drop)
    
    return X, y


def create_preprocessor(X):
    """Cria o preprocessador"""
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object']).columns
    
    # Pipeline numérico
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', QuantileClipper(0.01, 0.99))
    ])
    
    # Pipeline categórico
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Preprocessador
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    return preprocessor


def get_models():
    """Retorna dicionário de modelos"""
    random_state = 42
    
    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state
        ),
        "LightGBM": LGBMClassifier(
            random_state=random_state,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        )
    }
    
    return models


def get_scalers():
    """Retorna dicionário de scalers"""
    scalers = {
        "None": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "MaxAbsScaler": MaxAbsScaler(),
        "Normalizer": Normalizer()
    }
    
    return scalers


def train_model(experiment_name="churn_prediction"):
    """Treina modelos com MLFlow tracking"""
    
    random_state = 42
    
    # Definir diretório raiz do projeto
    # Navega de src/ para a raiz do projeto
    project_root = Path(__file__).parent.parent
    mlruns_path = project_root / "mlruns"
    data_path = project_root / "data" / "telco.csv"
    
    # Configurar MLFlow
    mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
    mlflow.set_experiment(experiment_name)
    
    # Carregar dados
    print("📊 Carregando dados...")
    print(f"📁 Diretório MLFlow: {mlruns_path.absolute()}")
    print(f"📁 Arquivo de dados: {data_path.absolute()}")
    X, y = load_and_prepare_data(str(data_path))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"✅ Dados carregados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    
    # Criar preprocessador
    preprocessor = create_preprocessor(X_train)
    
    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Obter modelos e scalers
    models = get_models()
    scalers = get_scalers()
    
    results = []
    best_roc_auc = 0
    best_model_info = {}
    
    print("\n🚀 Iniciando treinamento com MLFlow...\n")
    
    # Loop de treinamento
    for scaler_name, scaler in scalers.items():
        for model_name, model in models.items():
            
            with mlflow.start_run(run_name=f"{scaler_name}_{model_name}"):
                
                # Construir pipeline
                steps = [
                    ('preprocessor', preprocessor),
                    ('nearmiss', NearMiss(version=1))
                ]
                
                if scaler:
                    steps.append(('scaler', scaler))
                
                steps.append(('classifier', model))
                
                pipeline = Pipeline(steps)
                
                # Cross validation
                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
                )
                
                # Treinar no conjunto completo
                pipeline.fit(X_train, y_train)
                
                # Predições
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                # Log de parâmetros
                mlflow.log_param("scaler", scaler_name)
                mlflow.log_param("model", model_name)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("cv_splits", 5)
                mlflow.log_param("random_state", random_state)
                
                # Log de métricas
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                
                # Log de métricas de CV
                mlflow.log_metric("cv_accuracy_mean", cv_results['test_accuracy'].mean())
                mlflow.log_metric("cv_recall_mean", cv_results['test_recall'].mean())
                mlflow.log_metric("cv_precision_mean", cv_results['test_precision'].mean())
                mlflow.log_metric("cv_f1_mean", cv_results['test_f1'].mean())
                mlflow.log_metric("cv_roc_auc_mean", cv_results['test_roc_auc'].mean())
                
                # Registrar modelo
                mlflow.sklearn.log_model(
                    pipeline,
                    "model",
                    registered_model_name=None
                )
                
                # Salvar resultados
                results.append({
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1": f1,
                    "ROC AUC": roc_auc
                })
                
                # Verificar se é o melhor modelo
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_info = {
                        'scaler': scaler_name,
                        'model': model_name,
                        'pipeline': pipeline,
                        'run_id': mlflow.active_run().info.run_id,
                        'metrics': {
                            'accuracy': accuracy,
                            'recall': recall,
                            'precision': precision,
                            'f1': f1,
                            'roc_auc': roc_auc
                        }
                    }
                
                print(f"✓ {scaler_name} + {model_name}: ROC AUC = {roc_auc:.4f}")
    
    # Registrar melhor modelo
    print("\n" + "="*80)
    print("🏆 MELHOR MODELO")
    print("="*80)
    print(f"Scaler: {best_model_info['scaler']}")
    print(f"Model: {best_model_info['model']}")
    print(f"\nMétricas:")
    for metric, value in best_model_info['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print("="*80)
    
    # Registrar o melhor modelo no Model Registry
    with mlflow.start_run(run_id=best_model_info['run_id']):
        mlflow.sklearn.log_model(
            best_model_info['pipeline'],
            "model",
            registered_model_name="churn_best_model"
        )
    
    print(f"\n✅ Melhor modelo registrado no MLFlow Model Registry como 'churn_best_model'")
    print(f"Run ID: {best_model_info['run_id']}")
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ROC AUC", ascending=False)
    
    print("\n📊 Top 10 Modelos por ROC AUC:")
    print(results_df.head(10).to_string(index=False))
    
    return best_model_info


if __name__ == "__main__":
    best_model_info = train_model()

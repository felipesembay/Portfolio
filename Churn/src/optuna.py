import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


# ==========================================================
# QUANTILE CLIPPER
# ==========================================================

class QuantileClipper(BaseEstimator, TransformerMixin):

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


# ==========================================================
# PLOTS
# ==========================================================

def plot_roc_curve(y_test, y_prob):

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC Curve")

    file="roc_curve.png"
    plt.savefig(file)
    plt.close()

    return file


def plot_lift_curve(y_test, y_prob):

    df_lift = pd.DataFrame({"y":y_test,"prob":y_prob})
    df_lift = df_lift.sort_values("prob",ascending=False)

    df_lift["cum_churn"]=df_lift["y"].cumsum()
    df_lift["perc"]=np.arange(len(df_lift))/len(df_lift)

    plt.figure()
    plt.plot(df_lift["perc"],df_lift["cum_churn"]/df_lift["cum_churn"].max())
    plt.title("Lift Curve")

    file="lift_curve.png"
    plt.savefig(file)
    plt.close()

    return file


def plot_profit_curve(y_test,y_prob,value_saved=65,cost=10):

    thresholds=np.linspace(0,1,100)
    profits=[]

    for t in thresholds:

        pred=(y_prob>=t).astype(int)

        tp=((pred==1)&(y_test==1)).sum()
        fp=((pred==1)&(y_test==0)).sum()

        profit=(tp*value_saved)-(fp*cost)
        profits.append(profit)

    plt.figure()
    plt.plot(thresholds,profits)
    plt.title("Profit Curve")

    file="profit_curve.png"
    plt.savefig(file)
    plt.close()

    return file


def plot_confusion(y_test,y_pred):

    cm=confusion_matrix(y_test,y_pred)

    plt.figure()
    plt.imshow(cm,cmap="Blues")
    plt.title("Confusion Matrix")

    file="confusion_matrix.png"
    plt.savefig(file)
    plt.close()

    return file


def plot_shap_values(pipeline,X_train):

    model=pipeline.named_steps["classifier"]

    X_sample=X_train.sample(300)

    explainer=shap.Explainer(model)
    shap_values=explainer(X_sample)

    shap.plots.beeswarm(shap_values,show=False)

    file="shap_summary.png"

    plt.savefig(file,bbox_inches="tight")
    plt.close()

    return file


# ==========================================================
# DATA LOADING
# ==========================================================

def load_and_prepare_data(data_path):

    df = pd.read_csv(data_path)

    cols_to_drop = [
        "Customer ID","Customer Status","Churn Label","Churn Score",
        "Churn Category","Churn Reason","CLTV","Zip Code",
        "Latitude","Longitude","Country","State","City"
    ]

    y=(df["Churn Label"]=="Yes").astype(int)
    X=df.drop(columns=cols_to_drop)

    return X,y


# ==========================================================
# PREPROCESSOR
# ==========================================================

def create_preprocessor(X):

    num_features=X.select_dtypes(include=['int64','float64']).columns
    cat_features=X.select_dtypes(include=['object']).columns

    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('clipper',QuantileClipper(0.01,0.99))
    ])

    cat_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])

    preprocessor=ColumnTransformer([
        ('num',num_pipeline,num_features),
        ('cat',cat_pipeline,cat_features)
    ])

    return preprocessor


# ==========================================================
# MODELS
# ==========================================================

def get_models():

    random_state=42

    models={

        "Logistic Regression":LogisticRegression(
            solver="liblinear",
            random_state=random_state
        ),

        "Decision Tree":DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        ),

        "Random Forest":RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),

        "Gradient Boosting":GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state
        ),

        "LightGBM":LGBMClassifier(
            random_state=random_state,
            verbose=-1
        ),

        "XGBoost":XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        )
    }

    return models


# ==========================================================
# SCALERS
# ==========================================================

def get_scalers():

    return {

        "None":None,
        "StandardScaler":StandardScaler(),
        "MinMaxScaler":MinMaxScaler(),
        "RobustScaler":RobustScaler(),
        "MaxAbsScaler":MaxAbsScaler(),
        "Normalizer":Normalizer()

    }


# ==========================================================
# TRAINING
# ==========================================================

def train_model(experiment_name="churn_prediction"):

    random_state=42

    project_root=Path(__file__).parent.parent
    mlruns_path=project_root/"mlruns"
    data_path=project_root/"data"/"telco.csv"

    mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
    mlflow.set_experiment(experiment_name)

    print("📊 Carregando dados...")
    X,y=load_and_prepare_data(str(data_path))

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=random_state,stratify=y
    )

    preprocessor=create_preprocessor(X_train)

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)

    models=get_models()
    scalers=get_scalers()

    results=[]
    best_roc_auc=0
    best_model_info={}

    print("\n🚀 Iniciando treinamento...\n")

    for scaler_name,scaler in scalers.items():

        for model_name,model in models.items():

            with mlflow.start_run(run_name=f"{scaler_name}_{model_name}"):

                steps=[('preprocessor',preprocessor),
                       ('nearmiss',NearMiss(version=1))]

                if scaler:
                    steps.append(('scaler',scaler))

                steps.append(('classifier',model))

                pipeline=Pipeline(steps)

                cv_results=cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=['accuracy','recall','precision','f1','roc_auc']
                )

                pipeline.fit(X_train,y_train)

                y_pred=pipeline.predict(X_test)
                y_prob=pipeline.predict_proba(X_test)[:,1]

                accuracy=accuracy_score(y_test,y_pred)
                recall=recall_score(y_test,y_pred)
                precision=precision_score(y_test,y_pred)
                f1=f1_score(y_test,y_pred)
                roc_auc=roc_auc_score(y_test,y_prob)

                mlflow.log_param("scaler",scaler_name)
                mlflow.log_param("model",model_name)

                mlflow.log_metric("accuracy",accuracy)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("precision",precision)
                mlflow.log_metric("f1_score",f1)
                mlflow.log_metric("roc_auc",roc_auc)

                roc_file=plot_roc_curve(y_test,y_prob)
                lift_file=plot_lift_curve(y_test,y_prob)
                profit_file=plot_profit_curve(y_test,y_prob)
                cm_file=plot_confusion(y_test,y_pred)

                mlflow.log_artifact(roc_file)
                mlflow.log_artifact(lift_file)
                mlflow.log_artifact(profit_file)
                mlflow.log_artifact(cm_file)

                if model_name in ["Random Forest","Gradient Boosting","LightGBM","XGBoost"]:

                    shap_file=plot_shap_values(pipeline,X_train)
                    mlflow.log_artifact(shap_file)

                mlflow.sklearn.log_model(pipeline,"model")

                results.append({
                    "Scaler":scaler_name,
                    "Model":model_name,
                    "ROC AUC":roc_auc
                })

                if roc_auc>best_roc_auc:

                    best_roc_auc=roc_auc

                    best_model_info={
                        'scaler':scaler_name,
                        'model':model_name,
                        'pipeline':pipeline,
                        'run_id':mlflow.active_run().info.run_id,
                        'roc_auc':roc_auc
                    }

                print(f"✓ {scaler_name} + {model_name}: ROC AUC = {roc_auc:.4f}")

    print("\n🏆 Melhor modelo:")
    print(best_model_info)

    with mlflow.start_run(run_id=best_model_info['run_id']):
        mlflow.sklearn.log_model(
            best_model_info['pipeline'],
            "model",
            registered_model_name="churn_best_model"
        )

    return best_model_info


if __name__=="__main__":
    train_model()
import time

import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
import optuna
import plotly.express as px
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tabulate import tabulate
import os
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

for dirname, _, filenames in os.walk("abyznids"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#########################
#Veri Önişleme :
#eğitim ve test datalarını yüklüyoruz.
egitim=pd.read_csv("Train_data.csv")
test=pd.read_csv("Test_data.csv")


#########################
#Eksik Veri Kontrolü :
total = egitim.shape[0]
missing_columns = [col for col in egitim.columns if egitim[col].isnull().sum()>0]
for col in missing_columns :
    null_count = egitim[col].isnull().sum()
    per = (null_count/total)*100
    print(f"{col}: {null_count} ({round(per,3)}%)")
print(total,"Veri Kontrol Edildi")
#Tekrar edin verileri kontrol edelim :
print(f"Tekrar eden satır sayısı : {egitim.duplicated().sum()}")
#########################
#Aykırılık Kontrolü :

def le(df):
#     for col in df.columns:
#         if col != "class" and is_numeric_dtype(df[col]):
#             fig , ax = plt.subplot(211, figsize=(12,8))
#             g1 = sns.boxplot(x=df[col],ax=ax[0])
#             g2 = sns.scatterplot(data=df, x=df[col], y=df["class "], ax=ax[1])
#             plt.show()
#             print(g1,g2)

#########################
#IDS Paketi Isı Haritası:

    plt.figure(figsize=(40,30))
    sns.heatmap(egitim.corr(),annot=True)
    fig = px.imshow(df.corr(), text_auto=True,aspect="auto")
    fig.show()
print(f"IDS Isı Haritası: ")
le(egitim)



#########################
#Kategorik değerleri dönüştürme :
sns.countplot(x=egitim["class"])

def le(df):

    for col in df.columns:

        if df[col].dtype == "object":

                label_encoder = LabelEncoder()

                df[col] = label_encoder.fit_transform(df[col])

le(egitim)

le(test)
egitim.drop(["num_outbound_cmds"], axis=1, inplace=True)

test.drop(["num_outbound_cmds"], axis=1, inplace=True)

egitim.head()
#########################
#Özellik Seçimi
X_egitim = egitim.drop(["class"], axis=1)

Y_egitim = egitim["class"]

rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)

rfe = rfe.fit(X_egitim, Y_egitim)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_egitim.columns)]

selected_features = [v for i, v in feature_map if i==True]

selected_features

X_egitim = X_egitim[selected_features]
#########################
#Verilerin Bölünmesi ve Ölçeklendirilmesi

scale = StandardScaler()

X_egitim = scale.fit_transform(X_egitim)

test = scale.fit_transform(test)

x_egitim, x_test, y_egitim, y_test = train_test_split(X_egitim, Y_egitim, train_size=0.70, random_state=2)
#########################
#KNN Modeli Oluşturma
def objective(trial):

    n_neighbors = trial.suggest_int("KNN_n_neighbors", 2, 16, log=False)

    classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)

    classifier_obj.fit(x_egitim, y_egitim)

    accuracy = classifier_obj.score(x_test, y_test)

    return accuracy

study_KNN = optuna.create_study(direction="maximize")

study_KNN.optimize(objective, n_trials=1)

print(study_KNN.best_trial)
KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params["KNN_n_neighbors"])

KNN_model.fit(x_egitim, y_egitim)

KNN_egitim, KNN_test = KNN_model.score(x_egitim, y_egitim), KNN_model.score(x_test, y_test)
print(f"KNN Modeli: ")
print(f"Egitim Skoru: {KNN_egitim}")

print(f"Test Skoru: {KNN_test}")
##########################
#Logistik Regresyon Modeli
lg_model = LogisticRegression(random_state = 42)
lg_model.fit(x_egitim, y_egitim)
lg_egitim, lg_test = lg_model.score(x_egitim , y_egitim), lg_model.score(x_test , y_test)

print(f"Logistik Regresyon Modeli : ")
print(f"Egitim Skoru: {lg_egitim}")
print(f"Test Skoru: {lg_test}")

##############################
#Karar ağacı Sınıflandırılması
def objective(trial):
    dt_max_depth = trial.suggest_int("dt_max_depth", 2, 32, log=False)
    dt_max_features = trial.suggest_int("dt_max_features", 2, 10, log=False)
    classifier_obj = DecisionTreeClassifier(max_features = dt_max_features, max_depth = dt_max_depth)
    classifier_obj.fit(x_egitim, y_egitim)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy
study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective, n_trials=30)
print(study_dt.best_trial)

dt = DecisionTreeClassifier(max_features = study_dt.best_trial.params["dt_max_features"], max_depth = study_dt.best_trial.params["dt_max_depth"])
dt.fit(x_egitim, y_egitim)
dt_egitim, dt_test = dt.score(x_egitim, y_egitim), dt.score(x_test, y_test)

print(f"Karar Ağacı : ")
print(f"Egitim Skoru: {dt_egitim}")
print(f"Test Skoru: {dt_test}")

fig = plt.figure(figsize = (40,45))
tree.plot_tree(dt, filled=True);plt.show()
def f_importance(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    # Tum ozelliklerin gosterilmesi
    if top == -1:
        top = len(names)
    plt.barh(range(top), imp[::-1][0:top], align="center")

    plt.yticks(range(top), names[::-1][0:top])

    plt.title("Karar Agaci Siniflandirmasi için Onemli Veriler")

    plt.show()

# ozellik adi farketmeksizin

features_names = selected_features

# Görselleştirmek istediğiniz ilk n özelliğinizi belirtin.

# abs() işlevini de atabilirsiniz

# Ozelliklerin olumsuz etkisini de gormek istiyorsaniz.

f_importance(abs(dt.feature_importances_), features_names, top=10)

# Rassal Orman Algoritması
def objective(trial):

    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=False)

    rf_max_features = trial.suggest_int("rf_max_features", 2, 10, log=False)

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 3, 20, log=False)

    classifier_obj = RandomForestClassifier(max_features = rf_max_features, max_depth = rf_max_depth, n_estimators = rf_n_estimators)

    classifier_obj.fit(x_egitim, y_egitim)

    accuracy = classifier_obj.score(x_test, y_test)

    return accuracy

study_rf = optuna.create_study(direction="maximize")

study_rf.optimize(objective, n_trials=30)

print(study_rf.best_trial)
rf = RandomForestClassifier(max_features = study_rf.best_trial.params["rf_max_features"], max_depth = study_rf.best_trial.params["rf_max_depth"],
                            n_estimators = study_rf.best_trial.params["rf_n_estimators"])

rf.fit(x_egitim, y_egitim)

rf_egitim, rf_test = rf.score(x_egitim, y_egitim), rf.score(x_test, y_test)

print(f"Rassal Orman Algoritması : ")
print(f"Egitim Skoru: {rf_egitim}")
print(f"Test Skoru: {rf_test}")

def f_importance(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align="center")

    plt.yticks(range(top), names[::-1][0:top])

    plt.title("feature importance for dt")
    plt.show()

# özelliklerinizin adı ne olursa olsun

features_names = selected_features

# Görselleştirmek istediğiniz ilk n özelliğinizi belirtin.

# abs() işlevini de atabilirsiniz

# özelliklerin olumsuz katkısıyla ilgileniyorsanız

f_importance(abs(rf.feature_importances_), features_names, top=7)

######################
#SKLearn Gradyan Artırma Modeli
SKGB= GradientBoostingClassifier(random_state=42)
SKGB.fit(x_egitim,y_egitim)

SKGB_egitim, SKGB_test = SKGB.score(x_egitim , y_egitim), SKGB.score(x_test , y_test)
print(f"Sklearn Gradyan Artırma Modeli : ")
print(f"Egitim Skoru: {SKGB_egitim}")
print(f"Test Skoru: {SKGB_test}")

########################
#XGBoost Gradyan Artırma Modeli
xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(x_egitim, y_egitim)

xgb_egitim, xgb_test = xgb_model.score(x_egitim , y_egitim), xgb_model.score(x_test , y_test)
print(f"XGBoost Gradyan Artırma Modeli : ")
print(f"Egitim Skoru: {xgb_egitim}")

print(f"Test Skoru: {xgb_test}")

#################################
#Hafif Gradyan Artırma Modeli
lgb_model = LGBMClassifier(random_state=42)

lgb_model.fit(x_egitim, y_egitim)

lgb_egitim, lgb_test = lgb_model.score(x_egitim , y_egitim), lgb_model.score(x_test , y_test)
print(f"Hafif Gradyan Artırma Modeli : ")
print(f"Egitim Skoru: {lgb_egitim}")

print(f"Test Skoru: {lgb_test}")

###########################
# SkLearn AdaBoost Modeli
ab_model = AdaBoostClassifier(random_state=42)

ab_model.fit(x_egitim, y_egitim)

ab_egitim, ab_test = ab_model.score(x_egitim , y_egitim), ab_model.score(x_test , y_test)
print(f"SKLearn AdaBoost Modeli: ")
print(f"Egitim Skoru: {ab_egitim}")

print(f"Test Skoru: {ab_test}")

######################
# CatBoost Sınıflandırma Modeli
cb_model = CatBoostClassifier(verbose=0)

cb_model.fit(x_egitim, y_egitim)

cb_egitim, cb_test = cb_model.score(x_egitim , y_egitim), cb_model.score(x_test , y_test)
print(f"CatBoost Sınıflandırma Modeli : ")
print(f"Egitim Skoru: {cb_egitim}")

print(f"Test Skoru: {cb_test}")

############################
#Naive Bayes Sınıflandırma Modeli

BNB_model = BernoulliNB()

BNB_model.fit(x_egitim, y_egitim)

BNB_egitim, BNB_test = BNB_model.score(x_egitim , y_egitim), BNB_model.score(x_test , y_test)
print(f"Naive Bayes Sınıflandırma Modeli : ")
print(f"Egitim Skoru: {BNB_egitim}")

print(f"Test Skoru: {BNB_test}")

###############################
#Oylama Modeli

v_clf = VotingClassifier(estimators=[("KNeighborsClassifier", KNN_model), ("XGBClassifier", xgb_model),
("RandomForestClassifier", rf), ("DecisionTree", dt), ("XGBoost", xgb_model),
("LightGB", lgb_model), ("AdaBoost", ab_model), ("Catboost", cb_model)], voting = "hard")

v_clf.fit(x_egitim, y_egitim)

voting_egitim, voting_test = v_clf.score(x_egitim , y_egitim), v_clf.score(x_test , y_test)
print(f"Oylama Modeli : ")
print(f"Egitim Skoru: {voting_egitim}")

print(f"Test Skoru: {voting_test}")

###############################
#Destek Vektör Makineleri Modeli

def objective(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "linearSVC"])
    c = trial.suggest_float("c", 0.02, 1.0, step=0.02)
    if kernel in ["linear", "rbf"]: classifier_obj = SVC(kernel=kernel, C=c).fit(x_egitim, y_egitim)
    elif kernel == "linearSVC": classifier_obj = LinearSVC(C=c).fit(x_egitim, y_egitim)

    elif kernel == "poly":
        degree = trial.suggest_int("degree", 2, 10)
        classifier_obj = SVC(kernel=kernel, C=c, degree=degree).fit(x_egitim, y_egitim)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy
study_svm = optuna.create_study(direction="maximize")
study_svm.optimize(objective, n_trials=30)
print("Destek Vektör Makineleri Modeli = " , study_svm.best_trial)
def objective(trial2):
    kernel = trial2.suggest_categorical("kernel", ["linear", "rbf", "poly", "linearSVC"])
    if study_svm.best_trial.params["kernel"] in ["linear", "rbf"]:
        SVM_model = SVC(kernel=study_svm.best_trial.params["kernel"], C=study_svm.best_trial.params["c"])
        print("SVMMODEL:", SVM_model )

    elif kernel == "linearSVC":
         SVM_model = LinearSVC(C=study_svm.best_trial.params["c"])
         print("SVMMODEL2:", SVM_model )

    elif kernel == "poly":
        SVM_model = SVC(kernel=study_svm.best_trial.params["kernel"], C=study_svm.best_trial.params["c"], degree=study_svm.best_trial.params["degree"])
        SVM_model.fit(x_egitim, y_egitim)

        print("SVMMODEL3:", SVM_model)

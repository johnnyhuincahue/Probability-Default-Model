import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.stats as stat
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

class LogisticRegression_with_p_values:
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.model.fit(X, y)

        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

def get_metrics(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    df_actual_predicted_probs = pd.DataFrame({'targets': y_true, 'proba': y_pred_proba})
    df_actual_predicted_probs = df_actual_predicted_probs.sort_values('proba').reset_index(drop=True)
    
    df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
    df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['targets'].cumsum()
    df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['targets'].cumsum()
    
    total_population = df_actual_predicted_probs.shape[0]
    total_good = df_actual_predicted_probs['targets'].sum()
    total_bad = total_population - total_good

    df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / total_population
    df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / total_good
    df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / total_bad
    
    gini = auroc * 2 - 1
    ks = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])
    
    return fpr, tpr, auroc, df_actual_predicted_probs, gini, ks

def build_scorecard(reg, features, ref_categories):
    summary_table = pd.DataFrame(columns=['Feature name'], data=features)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()
    
    p_values = np.append(np.nan, np.array(reg.p_values))
    summary_table['p_values'] = p_values

    df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
    df_ref_categories['Coefficients'] = 0
    df_ref_categories['p_values'] = np.nan
    
    df_scorecard = pd.concat([summary_table, df_ref_categories]).reset_index(drop=True)
    df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

    min_score = 300
    max_score = 850
    min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
    max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
    
    df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
    df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
    df_scorecard['Score - Final'] = df_scorecard['Score - Calculation'].round()

    return df_scorecard

def predict_simulator(input_dict, model, model_features):
    df_sim = pd.DataFrame([input_dict])
    # Placeholder for complete processing injection
    # For production, apply processing(df_sim) and extract the exact model_features
    # Here we mock the vector directly based on model_features alignment
    vector = np.zeros((1, len(model_features)))
    
    # Mapping dict to vector logic (Simplified for structural representation)
    for i, col in enumerate(model_features):
        feature_root = col.split(':')[0]
        if feature_root in input_dict:
            if str(input_dict[feature_root]) in col:
                vector[0, i] = 1

    prob = model.model.predict_proba(vector)[:, 1][0]
    return prob
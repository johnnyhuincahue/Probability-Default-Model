import plotly.graph_objects as go
import plotly.express as px

def plot_roc(fpr, tpr, auroc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auroc:.3f})', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(title='Receiver Operating Characteristic', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_dark')
    return fig

def plot_lorenz(df_probs, gini_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_probs['Cumulative Perc Population'], y=df_probs['Cumulative Perc Bad'], mode='lines', name=f'Gini = {gini_score:.3f}', line=dict(color='#ff7f0e', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(title='Lorenz Curve', xaxis_title='Cumulative % Population', yaxis_title='Cumulative % Bad', template='plotly_dark')
    return fig

def plot_ks(df_probs, ks_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_probs['proba'], y=df_probs['Cumulative Perc Bad'], mode='lines', name='Cumulative % Bad', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df_probs['proba'], y=df_probs['Cumulative Perc Good'], mode='lines', name='Cumulative % Good', line=dict(color='blue', width=2)))
    fig.update_layout(title='Kolmogorov-Smirnov', xaxis_title='Estimated Probability for being Good', yaxis_title='Cumulative %', template='plotly_dark')
    return fig

def plot_woe(df_woe):
    fig = go.Figure()
    x_col = df_woe.columns[0]
    fig.add_trace(go.Scatter(x=df_woe[x_col].astype(str), y=df_woe['WoE'], mode='lines+markers', line=dict(color='#00CC96', width=2, dash='dash'), marker=dict(size=8)))
    fig.update_layout(title=f'Weight of Evidence - {x_col}', xaxis_title=x_col, yaxis_title='WoE', template='plotly_dark')
    return fig

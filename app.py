import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
from utils import load_data, woe_discrete, processing
from model import get_metrics, build_scorecard, LogisticRegression_with_p_values
from plots import plot_roc, plot_gini, plot_ks, plot_woe

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="PD Model & Scorecard", layout="wide")
st.title("Probability of Default Model")
st.markdown("---")

@st.cache_data
def load_all_data():
    try:
        X_train = pd.read_csv('./data/loan_data_inputs_train.csv', index_col=0)
        y_train = pd.read_csv('./data/loan_data_targets_train.csv', index_col=0)
        X_test = pd.read_csv('./data/loan_data_inputs_test.csv', index_col=0)
        y_test = pd.read_csv('./data/loan_data_targets_test.csv', index_col=0)
        logging.info("Archivos de datos cargados exitosamente.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error al cargar los datos: {e}")
        st.error("Fallo crítico en la carga de datos. Revise app.log.")
        return None, None, None, None

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('pd_model.sav', 'rb'))
        logging.info("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        st.error("Fallo crítico en la carga del modelo. Revise app.log.")
        return None

X_train, y_train, X_test, y_test = load_all_data()
reg2 = load_model()

if X_train is None or reg2 is None:
    logging.warning("Deteniendo ejecución por falta de dependencias críticas.")
    st.stop()

features_all = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    'addr_state:ND_NE_IA_NV_FL_HI_AL', 'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT', 'addr_state:KS_SC_CO_VT_AK_MS',
    'addr_state:WV_NH_WY_DC_ME_ID', 'verification_status:Not Verified',
    'verification_status:Source Verified', 'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house', 'purpose:credit_card',
    'purpose:debt_consolidation', 'purpose:oth__med__vacation',
    'purpose:major_purch__car__home_impr', 'initial_list_status:f', 'initial_list_status:w',
    'term:36', 'term:60', 'emp_length:0', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6',
    'emp_length:7-9', 'emp_length:10', 'mths_since_issue_d:<38', 'mths_since_issue_d:38-39',
    'mths_since_issue_d:40-41', 'mths_since_issue_d:42-48', 'mths_since_issue_d:49-52',
    'mths_since_issue_d:53-64', 'mths_since_issue_d:65-84', 'mths_since_issue_d:>84',
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74', 'int_rate:15.74-20.281',
    'int_rate:>20.281', 'mths_since_earliest_cr_line:<140', 'mths_since_earliest_cr_line:141-164',
    'mths_since_earliest_cr_line:165-247', 'mths_since_earliest_cr_line:248-270',
    'mths_since_earliest_cr_line:271-352', 'mths_since_earliest_cr_line:>352', 'inq_last_6mths:0',
    'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'inq_last_6mths:>6', 'acc_now_delinq:0',
    'acc_now_delinq:>=1', 'annual_inc:<20K', 'annual_inc:20K-30K', 'annual_inc:30K-40K',
    'annual_inc:40K-50K', 'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K',
    'annual_inc:80K-90K', 'annual_inc:90K-100K', 'annual_inc:100K-120K', 'annual_inc:120K-140K',
    'annual_inc:>140K', 'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5', 'dti:10.5-16.1',
    'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35', 'dti:>35',
    'mths_since_last_delinq:Missing', 'mths_since_last_delinq:0-3', 'mths_since_last_delinq:4-30',
    'mths_since_last_delinq:31-56', 'mths_since_last_delinq:>=57', 'mths_since_last_record:Missing',
    'mths_since_last_record:0-2', 'mths_since_last_record:3-20', 'mths_since_last_record:21-31',
    'mths_since_last_record:32-80', 'mths_since_last_record:81-86', 'mths_since_last_record:>86'
]

ref_categories = [
    'grade:G', 'home_ownership:RENT_OTHER_NONE_ANY', 'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified', 'purpose:educ__sm_b__wedd__ren_en__mov__house',
    'initial_list_status:f', 'term:60', 'emp_length:0', 'mths_since_issue_d:>84',
    'int_rate:>20.281', 'mths_since_earliest_cr_line:<140', 'inq_last_6mths:>6',
    'acc_now_delinq:0', 'annual_inc:<20K', 'dti:>35', 'mths_since_last_delinq:0-3',
    'mths_since_last_record:0-2'
]

inputs_test_with_ref_cat = X_test.loc[:, features_all]
inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis=1)
trained_features = inputs_test.columns.values

tab1, tab2, tab3, tab4 = st.tabs(["Variables & WOE", "Model Metrics", "Scorecard", "Simulator"])

with tab1:
    st.header("Variable Analysis")
    col1, col2 = st.columns([1, 3])
    
    df_train_full = pd.concat([X_train, y_train], axis=1)
    target_col = y_train.columns[0]
    
    categorical_columns = ['grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status']
    available_vars = [col for col in categorical_columns if col in df_train_full.columns]
    
    if available_vars:
        with col1:
            selected_var = st.selectbox("Select a variable:", available_vars)
        
        try:
            df_woe = woe_discrete(df_train_full, selected_var, target_col)
            with col2:
                st.plotly_chart(plot_woe(df_woe), use_container_width=True)
            st.dataframe(df_woe, use_container_width=True)
            logging.info(f"Cálculo WOE exitoso para: {selected_var}")
        except Exception as e:
            logging.error(f"Error al calcular WOE para {selected_var}: {e}")
            st.error("Error al procesar la variable seleccionada.")
    else:
        st.warning("Las variables originales no se encuentran en el dataset preprocesado. Ejecute WOE sobre datos crudos.")

with tab2:
    st.header("Métricas de Desempeño del Modelo")
    
    try:
        y_hat_test_proba = reg2.model.predict_proba(inputs_test)[:, 1]
        y_test_values = y_test[y_test.columns[0]].values
        
        fpr, tpr, auroc, df_probs, gini, ks = get_metrics(y_test_values, y_hat_test_proba)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("AUROC", f"{auroc:.4f}")
        col2.metric("Gini", f"{gini:.4f}")
        col3.metric("KS Statistic", f"{ks:.4f}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(plot_roc(fpr, tpr, auroc), use_container_width=True)
        with c2:
            st.plotly_chart(plot_gini(df_probs, gini), use_container_width=True)
        with c3:
            st.plotly_chart(plot_ks(df_probs, ks), use_container_width=True)
        logging.info("Métricas de desempeño calculadas exitosamente.")
    except Exception as e:
        logging.error(f"Error al calcular métricas: {e}")
        st.error("Error al procesar métricas del modelo.")

with tab3:
    st.header("Scorecard Operativo")
    
    try:
        df_scorecard = build_scorecard(reg2, trained_features, ref_categories)
        df_scorecard.loc[77, 'Score - Final'] = 16
        df_scorecard.loc[55, 'Score - Final'] = -5
        min_sum_score = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
        max_sum_score = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
        
        col1, col2 = st.columns(2)
        col1.metric("Score Mínimo Teórico", int(min_sum_score))
        col2.metric("Score Máximo Teórico", int(max_sum_score))
        
        st.dataframe(df_scorecard, use_container_width=True, height=600)
        logging.info("Scorecard construida exitosamente.")
    except Exception as e:
        logging.error(f"Error en construcción de Scorecard: {e}")
        st.error("Fallo al generar la Scorecard operativa.")

with tab4:
    st.header("Client Simulator")
    
    with st.form("simulator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            grade_in = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            home_in = st.selectbox("Home Ownership", ['MORTGAGE', 'OWN', 'RENT', 'OTHER', 'NONE', 'ANY'])
            purpose_in = st.selectbox("Purpose", ['credit_card', 'debt_consolidation', 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house', 'other', 'medical', 'vacation', 'major_purchase', 'car', 'home_improvement'])
        with col2:
            term_in = st.selectbox("Term (months)", [36, 60])
            emp_len_in = st.selectbox("Employment Length (years)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            veri_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
        with col3:
            ann_inc_in = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
            dti_in = st.number_input("DTI", min_value=0.0, value=15.0, step=0.5)
            state_in = st.text_input("State (2 letters)", value="CA")
            
        submitted = st.form_submit_button("Calcular PD & Score")
        
        if submitted:
            try:
                sim_data = pd.DataFrame([{
                    'grade': grade_in,
                    'home_ownership': home_in,
                    'purpose': purpose_in,
                    'term_int': term_in,
                    'emp_length_int': emp_len_in,
                    'verification_status': veri_status,
                    'annual_inc': ann_inc_in,
                    'dti': dti_in,
                    'addr_state': state_in,
                    'mths_since_issue_d': 40, 
                    'int_rate': 12.0, 
                    'mths_since_earliest_cr_line': 150,
                    'inq_last_6mths': 0,
                    'acc_now_delinq': 0,
                    'mths_since_last_delinq': 0,
                    'mths_since_last_record': 0,
                    'delinq_2yrs': 0,
                    'open_acc': 5,
                    'pub_rec': 0,
                    'total_acc': 10,
                    'total_rev_hi_lim': 10000.0,
                    'installment': 300.0,
                    'funded_amnt': 10000.0
                }])
                
                expected_dummies = [
                    'home_ownership:RENT', 'home_ownership:OTHER', 'home_ownership:NONE', 'home_ownership:ANY',
                    'purpose:educational', 'purpose:small_business', 'purpose:wedding', 'purpose:renewable_energy',
                    'purpose:moving', 'purpose:house', 'purpose:other', 'purpose:medical', 'purpose:vacation',
                    'purpose:major_purchase', 'purpose:car', 'purpose:home_improvement'
                ] + [f"addr_state:{s}" for s in [
                    'ND', 'NE', 'IA', 'NV', 'FL', 'HI', 'AL', 'NM', 'VA', 'OK', 'TN', 'MO', 'LA', 'MD', 'NC', 
                    'UT', 'KY', 'AZ', 'NJ', 'AR', 'MI', 'PA', 'OH', 'MN', 'RI', 'MA', 'DE', 'SD', 'IN', 'GA', 
                    'WA', 'OR', 'WI', 'MT', 'IL', 'CT', 'KS', 'SC', 'CO', 'VT', 'AK', 'MS', 'WV', 'NH', 'WY', 
                    'DC', 'ME', 'ID'
                ]]
                
                for dummy in expected_dummies:
                    sim_data[dummy] = 0
                    
                sim_data[f"home_ownership:{home_in}"] = 1
                sim_data[f"purpose:{purpose_in}"] = 1
                sim_data[f"addr_state:{state_in}"] = 1
                
                sim_data_processed = processing(sim_data)
                
                vector = np.zeros((1, len(trained_features)))
                for i, col in enumerate(trained_features):
                    if col in sim_data_processed.columns:
                        vector[0, i] = sim_data_processed[col].iloc[0]
                    elif ":" in col:
                        root, val = col.split(":", 1)
                        if root in sim_data.columns and str(sim_data[root].iloc[0]) == val:
                            vector[0, i] = 1
                
                prob = reg2.model.predict_proba(vector)[:, 1][0]
                
                min_score = 300
                max_score = 850
                min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
                max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
                
                score = ((np.log(prob / (1 - prob)) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score)
                
                st.success("Calculation completed successfully.")
                sc1, sc2 = st.columns(2)
                sc1.metric("Probabilidad de No Default (Good)", f"{prob * 100:.2f}%")
                sc2.metric("Credit Score", int(round(score)))
                logging.info(f"Simulación ejecutada. PD calculada: {prob:.4f}, Score: {int(round(score))}")          
            except Exception as e:
                logging.error(f"Error durante simulación: {e}")
                st.error("Error al ejecutar la simulación. Revise app.log.")   
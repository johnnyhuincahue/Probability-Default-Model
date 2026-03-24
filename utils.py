import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        return None
def preprocessing(loan_data: pd.DataFrame) -> pd.DataFrame:
    loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('+ years', '')
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
    loan_data['emp_length_int'] = loan_data['emp_length_int'].replace(np.nan, str(0))
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
    loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
    loan_data['term_int'] = loan_data['term'].str.replace(' months','')
    loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
    loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')
    loan_data['mths_since_earliest_cr_line'] = round((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']).dt.days / 30.44)
    loan_data.loc[loan_data['mths_since_earliest_cr_line'] < 0, 'mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].max()
    loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')
    loan_data['mths_since_issue_d'] = round((pd.to_datetime('2017-12-01') - loan_data['issue_d_date']).dt.days / 30.44)
    loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                         pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                         pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                         pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                         pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                         pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                         pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                         pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]
    loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)
    loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)
    loan_data.fillna({'total_rev_hi_lim' : loan_data['funded_amnt']}, inplace = True)
    columns_to_fill = ['mths_since_earliest_cr_line', 'acc_now_delinq', 'total_acc', 'pub_rec', 'open_acc'
                       , 'inq_last_6mths', 'delinq_2yrs', 'emp_length_int']
    loan_data.fillna({col : 0 for col in columns_to_fill}, inplace = True)
    loan_data.fillna({'annual_inc' : loan_data['annual_inc'].mean()}, inplace = True)
    loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                           'Does not meet the credit policy. Status:Charged Off',
                                                           'Late (31-120 days)']), 0, 1)
    return loan_data
def woe_discrete(df: pd.DataFrame, discrete_variable_name: str, good_bad_variable_name: str) -> pd.DataFrame:
    df_woe = pd.concat([df[discrete_variable_name], df[good_bad_variable_name]], axis=1)
    df_woe = pd.concat([
        df_woe.groupby(df_woe.columns.values[0], as_index=False)[df_woe.columns.values[1]].count(),
        df_woe.groupby(df_woe.columns.values[0], as_index=False)[df_woe.columns.values[1]].mean()
    ], axis=1)
    df_woe = df_woe.iloc[:, [0, 1, 3]]
    df_woe.columns = [df_woe.columns.values[0], 'n_obs', 'prop_good']
    df_woe['prop_n_obs'] = df_woe['n_obs'] / df_woe['n_obs'].sum()
    df_woe['n_good'] = df_woe['prop_good'] * df_woe['n_obs']
    df_woe['n_bad'] = (1 - df_woe['prop_good']) * df_woe['n_obs']
    df_woe['prop_n_good'] = df_woe['n_good'] / df_woe['n_good'].sum()
    df_woe['prop_n_bad'] = df_woe['n_bad'] / df_woe['n_bad'].sum()
    df_woe['WoE'] = np.log(df_woe['prop_n_good'] / df_woe['prop_n_bad'])
    df_woe = df_woe.sort_values(['WoE']).reset_index(drop=True)
    df_woe['IV'] = (df_woe['prop_n_good'] - df_woe['prop_n_bad']) * df_woe['WoE']
    return df_woe
def safe_sum(df, new_col, cols_to_sum):
    existing_cols = [c for c in cols_to_sum if c in df.columns]
    if existing_cols:
        df[new_col] = df[existing_cols].sum(axis=1)
    else:
        df[new_col] = 0
    return df
def processing(df_inputs_prepr):
    # Home ownership
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'home_ownership:RENT_OTHER_NONE_ANY',
                               ['home_ownership:RENT', 'home_ownership:OTHER', 'home_ownership:NONE', 'home_ownership:ANY'])
    
    # Addr state
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:ND', ['addr_state:ND'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:ND_NE_IA_NV_FL_HI_AL',
                               ['addr_state:ND', 'addr_state:NE', 'addr_state:IA', 'addr_state:NV', 
                                'addr_state:FL', 'addr_state:HI', 'addr_state:AL'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:NM_VA', ['addr_state:NM', 'addr_state:VA'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:OK_TN_MO_LA_MD_NC',
                               ['addr_state:OK', 'addr_state:TN', 'addr_state:MO', 'addr_state:LA', 
                                'addr_state:MD', 'addr_state:NC'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:UT_KY_AZ_NJ', ['addr_state:UT', 'addr_state:KY', 'addr_state:AZ', 'addr_state:NJ'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:AR_MI_PA_OH_MN',
                               ['addr_state:AR', 'addr_state:MI', 'addr_state:PA', 'addr_state:OH', 'addr_state:MN'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:RI_MA_DE_SD_IN',
                               ['addr_state:RI', 'addr_state:MA', 'addr_state:DE', 'addr_state:SD', 'addr_state:IN'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:GA_WA_OR', ['addr_state:GA', 'addr_state:WA', 'addr_state:OR'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:WI_MT', ['addr_state:WI', 'addr_state:MT'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:IL_CT', ['addr_state:IL', 'addr_state:CT'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:KS_SC_CO_VT_AK_MS',
                               ['addr_state:KS', 'addr_state:SC', 'addr_state:CO', 'addr_state:VT', 'addr_state:AK', 'addr_state:MS'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'addr_state:WV_NH_WY_DC_ME_ID',
                               ['addr_state:WV', 'addr_state:NH', 'addr_state:WY', 'addr_state:DC', 'addr_state:ME', 'addr_state:ID'])
    
    # Purpose
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'purpose:educ__sm_b__wedd__ren_en__mov__house',
                               ['purpose:educational', 'purpose:small_business', 'purpose:wedding',
                                'purpose:renewable_energy', 'purpose:moving', 'purpose:house'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'purpose:oth__med__vacation',
                               ['purpose:other', 'purpose:medical', 'purpose:vacation'])
    df_inputs_prepr = safe_sum(df_inputs_prepr, 'purpose:major_purch__car__home_impr',
                               ['purpose:major_purchase', 'purpose:car', 'purpose:home_improvement'])
    df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
    df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)
    df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
    df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
    df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
    df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
    df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
    df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)
    df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)
    df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
    df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)
    df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
    df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
    df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
    df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
    df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
    df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)
    df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
    df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
    df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)
    df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
    df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
    df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)
    df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
    df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
    df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
    df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)
    df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
    df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
    df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
    df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
    df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
    df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
    df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
    df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)
    df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
    df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
    df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)
    df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
    df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
    df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
    df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)
    df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
    df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)
    df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
    df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)
    df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
    df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)
    df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
    df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
    df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
    df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
    df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
    df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
    df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
    df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
    df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
    df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
    df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
    df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
    df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)
    df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
    df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
    df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
    df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
    df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
    df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
    df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
    df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
    df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
    df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
    df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)
    df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
    df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
    df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
    df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
    df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
    df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
    df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)
    return df_inputs_prepr
def woe_ordered_continuous(df, continuous_variable_name, target_variable_name):
    # Crear copia para evitar SettingWithCopyWarning
    df_woe = df[[continuous_variable_name, target_variable_name]].copy()
    
    # Agrupar y calcular n_obs y prop_good en un solo paso
    df_woe = df_woe.groupby(continuous_variable_name)[target_variable_name].agg(['count', 'mean']).reset_index()
    
    # Asignar nombres correctos (ahora coinciden con las 3 columnas resultantes)
    df_woe.columns = [continuous_variable_name, 'n_obs', 'prop_good']
    
    df_woe['prop_n_obs'] = df_woe['n_obs'] / df_woe['n_obs'].sum()
    df_woe['n_good'] = df_woe['prop_good'] * df_woe['n_obs']
    df_woe['n_bad'] = (1 - df_woe['prop_good']) * df_woe['n_obs']
    df_woe['prop_n_good'] = df_woe['n_good'] / df_woe['n_good'].sum()
    df_woe['prop_n_bad'] = df_woe['n_bad'] / df_woe['n_bad'].sum()
    
    # Cálculo de WoE e IV por fila
    df_woe['WoE'] = np.log(df_woe['prop_n_good'] / df_woe['prop_n_bad'])
    df_woe['IV'] = (df_woe['prop_n_good'] - df_woe['prop_n_bad']) * df_woe['WoE']
    
    return df_woe
---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
# esto detecta automáticamente cualquier cambio en los módulos sin necesidad de reiniciar el kernel
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
```

```python
import pandas as pd
import numpy as np
from datetime import datetime

from bank_clients_ml.utils import print_df_personalizado, print_threshold_violations, describe_full, print_value_counts, filter_nonzero
from bank_clients_ml.features import (
    compute_percentage, replace_null_with_value, calculate_target_percentage_by_category,
    columnas_minimo_entre, columnas_con_ceros, safe_denominator, a_datetime, group_columns_by_source,
    min_max_normalize, min_max_normalize_weighted, apply_binning_by_ranges,
)
from bank_clients_ml.models import train_and_get_feature_importances, evaluate_deciles_train, evaluate_deciles_test, process_model_results

```

```python
data = pd.read_parquet('../data/data.parquet', engine='pyarrow', dtype_backend='pyarrow')

print_df_personalizado("data", data)
```

```python
describe_full(data)
```

```python
less_than_zero_columns = [
    "SavingAccount_Balance_Average",
    "CreditCard_Balance_ARG",
    "CreditCard_Balance_DOLLAR",
    "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment",
    "CreditCard_Spending_Aut_Debits",
    "CreditCard_Revolving",
]

greater_than_thirty_one_columns = [
    "SavingAccount_Days_with_use",
    "SavingAccount_Days_with_Debits",
]

print_threshold_violations(data, column_name="SavingAccount_Balance_Average", threshold=0, condition="lt")
print("")

for column in less_than_zero_columns:
    print_threshold_violations(data, column_name=column, threshold=0, condition="lt")
print("")

for column in greater_than_thirty_one_columns:
    print_threshold_violations(data, column_name=column, threshold=31, condition="gt")
```

```python
for x in data.columns:
    print_value_counts(data, x)
```

## Obtener meses relevantes

```python
meses_validos = pd.to_datetime(data['Month'], format='%Y-%m-%d').drop_duplicates()
ultimo_mes = meses_validos.max()
primer_mes = ultimo_mes - pd.DateOffset(months=8)

ultimo_mes_entrenamiento = (ultimo_mes - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
meses_entrenamiento = pd.date_range(start=primer_mes, end=ultimo_mes_entrenamiento, freq='MS').strftime('%Y-%m-%d').tolist()
print("meses_entrenamiento:", meses_entrenamiento)
print("")

primer_mes_prediccion = (ultimo_mes - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
meses_prediccion = pd.date_range(start=primer_mes_prediccion, end=ultimo_mes, freq='MS').strftime('%Y-%m-%d').tolist()
print("meses_prediccion:", meses_prediccion)
```

# Definir Universo y Target

```python
# filtro clientes que tengan menos de 9 meses de historia
cant_meses_x_cliente = pd.DataFrame(data['client_id'].value_counts().reset_index())
cant_meses_x_cliente.columns = ['client_id', 'cant_meses']
print("cant_meses_x_cliente['cant_meses']:", cant_meses_x_cliente['cant_meses'].value_counts())
print("")

clientes_validos_1 = cant_meses_x_cliente[cant_meses_x_cliente['cant_meses'] == 9] [['client_id']]
print("clientes_validos_1:", clientes_validos_1.shape)
print("")

# filtro clientes con 'Package_Active' y 'CreditCard_CoBranding' en el ultimo mes de la ventana de entrenamiento
clientes_validos_2 = data[(data['Package_Active'] == 'No') & (data['CreditCard_CoBranding'] == 'No') & (data['Month'] == ultimo_mes_entrenamiento)] [['client_id']]
print("clientes_validos_2:", clientes_validos_2.shape)
print("")

# universo
universo = clientes_validos_1.merge(clientes_validos_2, on='client_id', how='inner')
print("universo:", universo.shape)
print("")

# Ventana de Prediccion
tgt = data[(data['Month'].isin(meses_prediccion))] [['client_id', 'Target']].drop_duplicates()
print("tgt['Target']:", tgt['Target'].value_counts())
print("")

universo_con_target = universo.merge(tgt, how='left', on='client_id')
print("universo_con_target['Target']:", universo_con_target['Target'].value_counts())
print("universo_con_target:", universo_con_target.shape)
print("")

# Ventana de Entrenamiento
data_entrenamiento = data[data['Month'].isin(meses_entrenamiento)]
print("data_entrenamiento['client_id']:", len(data_entrenamiento['client_id'].drop_duplicates()))
print("")

data_entrenamiento = data_entrenamiento.merge(universo_con_target[['client_id']], how='inner', on='client_id')
print("data_entrenamiento['Month']:", data_entrenamiento['Month'].value_counts())
```

```python
print_df_personalizado("data_entrenamiento", data_entrenamiento)
```

## Balancear a 50%/50% aprox (oversampling) <- a modo de ejemplo, esto despues no lo uso

```python
print(data_entrenamiento['Target'].value_counts())
print(data_entrenamiento.shape)

minoria = data_entrenamiento[data_entrenamiento['Target'] == 1]
minoria_2 = minoria.copy()
id_maximo = int(data_entrenamiento['client_id'].max())
ids_nuevos = range(id_maximo + 1, id_maximo + 1 + len(minoria))

print("")
print(minoria_2['client_id'])
minoria_2['client_id'] = ids_nuevos

print("")
print(id_maximo)
print(ids_nuevos)
print("")
print(minoria_2['client_id'])

data_entrenamiento_b = pd.concat([data_entrenamiento, minoria_2])
print("")
print(data_entrenamiento_b['Target'].value_counts())
print(data_entrenamiento_b.shape)
```

# EDA (Exploratory Data Analysis)


## Valores Nulos

```python
print(data_entrenamiento['SavingAccount_Balance_Average'].value_counts())
print("")
print(data_entrenamiento['Region'].value_counts())
print("")
print(data_entrenamiento['CreditCard_Product'].value_counts())
```

```python
print("cantidad de registros con nulos en SavingAccount_Balance_Average:", data_entrenamiento['SavingAccount_Balance_Average'].isna().sum())
```

```python
# analizando valores monetarios de SavingAccount
saving_account_cols = [
    "SavingAccount_Balance_Average", 
    "SavingAccount_Balance_FirstDate", 
    "SavingAccount_Balance_LastDate",
    "SavingAccount_Total_Amount",
    "SavingAccount_Salary_Payment_Amount",
    "SavingAccount_Transfer_In_Amount",
    "SavingAccount_Credits_Amounts",
    "SavingAccount_ATM_Extraction_Amount",
    "SavingAccount_Service_Payment_Amount",
    "SavingAccount_CreditCard_Payment_Amount",
    "SavingAccount_Transfer_Out_Amount",
    "SavingAccount_DebitCard_Spend_Amount",
    "SavingAccount_Debits_Amounts"
]

# registros con nulos en SavingAccount_Balance_Average
data_entrenamiento[["client_id"] + saving_account_cols][
    data_entrenamiento["SavingAccount_Balance_Average"].isna()
]
```

```python
filter_nonzero(data_entrenamiento, saving_account_cols)
```

# Feature Engineering


## Completando 'SavingAccount_Balance_Average'

Saco el promedio entre "SavingAccount_Balance_FirstDate" y "SavingAccount_Balance_LastDate".
Este no es el calculo correcto para "SavingAccount_Balance_Average", pero no va a afectar tanto al modelo porque son solo 4 registros con nulos ademas de que no hay una forma sencilla de calcular el "SavingAccount_Balance_Average" con los datos que tenemos

```python
data_entrenamiento['SavingAccount_Balance_Average'] = replace_null_with_value(
    data_entrenamiento, 
    'SavingAccount_Balance_Average', 
    (data_entrenamiento.SavingAccount_Balance_FirstDate + data_entrenamiento.SavingAccount_Balance_LastDate)/2
)
```

## Completando 'Region'

```python
regiones_x_cliente = data[data['Month'] == primer_mes_prediccion] [['client_id', 'Region']]
print("columnas con nulos en regiones_x_cliente:", regiones_x_cliente.columns[regiones_x_cliente.isnull().any()].tolist())
print("cantidad de nulos en Region:", regiones_x_cliente.Region.isnull().sum())

regiones_x_cliente['Region'] = replace_null_with_value(regiones_x_cliente, 'Region', 'BUENOS AIRES') # pongo la Region mas comun
print("columnas con nulos en regiones_x_cliente:", regiones_x_cliente.columns[regiones_x_cliente.isnull().any()].tolist())

data_entrenamiento = data_entrenamiento.drop(columns=['Region'])
data_entrenamiento = data_entrenamiento.merge(regiones_x_cliente, how='left', on='client_id')
```

## Completando 'CreditCard_Product'

```python
producto_x_cliente = data[data['Month'] == primer_mes_prediccion] [['client_id', 'CreditCard_Product']]
print(producto_x_cliente.shape)
print("")
print(producto_x_cliente['CreditCard_Product'].value_counts())
```

```python
data_entrenamiento = data_entrenamiento.drop(columns=['CreditCard_Product'])
data_entrenamiento = data_entrenamiento.merge(producto_x_cliente, how='left', on='client_id')

print("")
print(data_entrenamiento['CreditCard_Product'].value_counts())

val1 = int(data_entrenamiento['CreditCard_Product'].value_counts()["J55660104XX012"])

# pongo la mas comun cuando no tiene producto en el futuro pero si tiene producto activo en el pasado
condicion_1 = data_entrenamiento['CreditCard_Product'].isnull() & (data_entrenamiento['CreditCard_Active'] == 'Yes')
data_entrenamiento.loc[condicion_1, 'CreditCard_Product'] = 'J55660104XX012'

print("")
print(data_entrenamiento['CreditCard_Product'].value_counts())

# pongo 0 cuando no tiene producto en el futuro ni en el pasado; o cuando no tiene producto en el pasado, por mas que lo tenga en el futuro
condicion_2 = data_entrenamiento['CreditCard_Product'].isnull() | (data_entrenamiento['CreditCard_Active'] == 'No')
data_entrenamiento.loc[condicion_2, 'CreditCard_Product'] = '0'

print("")
print(data_entrenamiento['CreditCard_Product'].value_counts())

val2 = int(data_entrenamiento['CreditCard_Product'].value_counts()["J55660104XX012"])
```

```python
print("cantidad de clientes que no tienen producto en el futuro pero si tienen producto activo en el pasado:", val2-val1)
```

```python
print_df_personalizado("data_entrenamiento", data_entrenamiento)
```

# Identity Features

```python
diccionario = {"Yes": 1, "No": 0, "M": 1, "F": 0}

columnas_if = [
    'CreditCard_Premium', 'CreditCard_Active', 'CreditCard_CoBranding', 'Loan_Active', 'Mortgage_Active', 'SavingAccount_Active_ARG_Salary', 'SavingAccount_Active_ARG', 'SavingAccount_Active_DOLLAR', 
    'DebitCard_Active', 'Investment_Active', 'Package_Active', 'Insurance_Life', 'Insurance_Home', 'Insurance_Accidents', 'Insurance_Mobile', 'Insurance_ATM', 'Insurance_Unemployment', 'Sex', 'Mobile', 
    'Email'
]

print(data_entrenamiento[['CreditCard_Premium']].value_counts())
print(data_entrenamiento[['Sex']].value_counts())

for c in columnas_if:
    data_entrenamiento[c] = data_entrenamiento[c].map(diccionario)

print("")
print(data_entrenamiento[['CreditCard_Premium']].value_counts())
print(data_entrenamiento[['Sex']].value_counts())

columnas_if = ['client_id', 'Client_Age_grp', 'Region', 'CreditCard_Product', 'First_product_dt', 'Last_product_dt'] + columnas_if
print("")
print(columnas_if)

data_if = data_entrenamiento[data_entrenamiento['Month'] == ultimo_mes_entrenamiento][columnas_if]
```

# Variables Categoricas

```python
data_if = data_if.merge(universo_con_target, how='inner', on='client_id')

for col in ['Client_Age_grp', 'Region', 'CreditCard_Product']:
    print_value_counts(data_if, col)
    porcentajes = calculate_target_percentage_by_category(data_if, col)
    data_if[col] = data_if[col].map(porcentajes.to_dict())
    print_value_counts(data_if, col)
```

# Fechas

```python
data_if['First_product_dt'] = (a_datetime(data_if['Last_product_dt']) - a_datetime(data_if['First_product_dt'])).dt.days
data_if.rename(columns={'First_product_dt': 'Dias_entre_primer_y_ultimo_producto'}, inplace=True)
print("")
print(data_if[['Dias_entre_primer_y_ultimo_producto']].value_counts())

data_if['Last_product_dt'] = (a_datetime(ultimo_mes_entrenamiento) + pd.DateOffset(months=1) - a_datetime(data_if['Last_product_dt'])).dt.days
data_if.rename(columns={'Last_product_dt': 'Recencia_en_dias'}, inplace=True)
print("")
print(data_if[['Recencia_en_dias']].value_counts())

print("")
print("data_if:", data_if.shape)
```

```python
describe_full(data_if)
```

# Transform features

```python
# analizando valores minimos y ceros
columnas_minimo_entre(data_entrenamiento, -1, 1)
```

```python
columnas_con_ceros(data_entrenamiento)
```

```python
data_entrenamiento = data_entrenamiento.sort_values(['client_id', 'Month'])
```

```python
# analizando valores monetarios de CreditCard
credit_card_cols = [
    "CreditCard_Balance_ARG", 
    "CreditCard_Balance_DOLLAR", 
    "CreditCard_Total_Limit",
    "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment",
    "CreditCard_Spending_Installments",
    "CreditCard_Spending_CrossBoarder",
    "CreditCard_Spending_Aut_Debits",
    "CreditCard_Revolving",
]

filter_nonzero(data_entrenamiento, credit_card_cols)
```

```python
print(data_entrenamiento.shape)

transformadas_1 = {

# OPERATION
'Operations_total': (data_entrenamiento['Operations_Bank'] + data_entrenamiento['Operations_Terminal'] + data_entrenamiento[ 'Operations_HomeBanking'] + 
    data_entrenamiento['Operations_Mobile'] + data_entrenamiento['Operations_Ivr'] + data_entrenamiento['Operations_Telemarketer'] + data_entrenamiento['Operations_ATM']),

'Operations_digitales': (data_entrenamiento['Operations_HomeBanking'] + data_entrenamiento['Operations_Mobile'] + data_entrenamiento['Operations_Ivr'] + 
    data_entrenamiento['Operations_Telemarketer']),

'Operations_presenciales': (data_entrenamiento['Operations_Bank'] + data_entrenamiento['Operations_Terminal'] + data_entrenamiento['Operations_ATM']),


    
# CREDIT CARD
'CreditCard_Payment_total': (data_entrenamiento['CreditCard_Payment_Aut_Debit'] + data_entrenamiento['CreditCard_Payment_External'] + data_entrenamiento['CreditCard_Payment_Cash'] + 
    data_entrenamiento['CreditCard_Payment_Web'] + data_entrenamiento['CreditCard_Payment_ATM'] + data_entrenamiento['CreditCard_Payment_TAS']),

'CreditCard_Payment_digitales': data_entrenamiento['CreditCard_Payment_Aut_Debit'] + data_entrenamiento['CreditCard_Payment_Web'],

'CreditCard_Payment_presenciales': (data_entrenamiento['CreditCard_Payment_External'] + data_entrenamiento['CreditCard_Payment_Cash'] + 
                                                         data_entrenamiento['CreditCard_Payment_ATM'] + data_entrenamiento['CreditCard_Payment_TAS']),
    
}


data_entrenamiento = pd.concat(
    [data_entrenamiento, pd.DataFrame(transformadas_1)],
    axis='columns'
)



transformadas_2 = {

# SAVING ACCOUNT
'SavingAccount_Balance_last_minus_first_date': data_entrenamiento['SavingAccount_Balance_LastDate'] - data_entrenamiento['SavingAccount_Balance_FirstDate'],

'SavingAccount_Balance_last_minus_first_date_porc': compute_percentage(data_entrenamiento['SavingAccount_Balance_LastDate'], data_entrenamiento['SavingAccount_Balance_FirstDate']),

'SavingAccount_Days_with_Debits_porc': compute_percentage(data_entrenamiento['SavingAccount_Days_with_Debits'], data_entrenamiento['SavingAccount_Days_with_use']),
'SavingAccount_Days_with_Credits_porc': compute_percentage(data_entrenamiento['SavingAccount_Days_with_Credits'], data_entrenamiento['SavingAccount_Days_with_use']),

'SavingAccount_Credits_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Credits_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']), 
'SavingAccount_Debits_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Debits_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']), 

'SavingAccount_Transactions_Transactions_DAYS_prom': data_entrenamiento['SavingAccount_Credits_Transactions'] / safe_denominator(data_entrenamiento['SavingAccount_Days_with_use']),
'SavingAccount_Credits_Transactions_DAYS_prom': data_entrenamiento['SavingAccount_Credits_Transactions'] / safe_denominator(data_entrenamiento['SavingAccount_Days_with_Credits']),
'SavingAccount_Debits_Transactions_DAYS_prom': data_entrenamiento['SavingAccount_Debits_Transactions'] / safe_denominator(data_entrenamiento['SavingAccount_Days_with_Debits']),


'SavingAccount_Salary_Payment_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Salary_Payment_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),
'SavingAccount_Transfer_In_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_In_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),
'SavingAccount_ATM_Extraction_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_ATM_Extraction_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']), 
'SavingAccount_Service_Payment_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Service_Payment_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),
'SavingAccount_CreditCard_Payment_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_CreditCard_Payment_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),
'SavingAccount_Transfer_Out_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_Out_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),
'SavingAccount_DebitCard_Spend_Transactions_porc': compute_percentage(data_entrenamiento['SavingAccount_DebitCard_Spend_Transactions'], data_entrenamiento['SavingAccount_Transactions_Transactions']),

'SavingAccount_Salary_Payment_Transactions_CR_porc': compute_percentage(data_entrenamiento['SavingAccount_Salary_Payment_Transactions'], data_entrenamiento['SavingAccount_Credits_Transactions']),
'SavingAccount_Transfer_In_Transactions_CR_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_In_Transactions'], data_entrenamiento['SavingAccount_Credits_Transactions']),
'SavingAccount_ATM_Extraction_Transactions_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_ATM_Extraction_Transactions'], data_entrenamiento['SavingAccount_Debits_Transactions']),
'SavingAccount_Service_Payment_Transactions_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_Service_Payment_Transactions'], data_entrenamiento['SavingAccount_Debits_Transactions']),
'SavingAccount_CreditCard_Payment_Transactions_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_CreditCard_Payment_Transactions'], data_entrenamiento['SavingAccount_Debits_Transactions']),
'SavingAccount_Transfer_Out_Transactions_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_Out_Transactions'], data_entrenamiento['SavingAccount_Debits_Transactions']),
'SavingAccount_DebitCard_Spend_Transactions_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_DebitCard_Spend_Transactions'], data_entrenamiento['SavingAccount_Debits_Transactions']),

    
'SavingAccount_Credits_Amounts_porc': compute_percentage(data_entrenamiento['SavingAccount_Credits_Amounts'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_Debits_Amounts_porc': compute_percentage(data_entrenamiento['SavingAccount_Debits_Amounts'], data_entrenamiento['SavingAccount_Total_Amount']),


'SavingAccount_Salary_Payment_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_Salary_Payment_Amount'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_Transfer_In_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_In_Amount'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_ATM_Extraction_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_ATM_Extraction_Amount'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_Service_Payment_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_Service_Payment_Amount'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_CreditCard_Payment_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_CreditCard_Payment_Amount'], data_entrenamiento['SavingAccount_Total_Amount']),
'SavingAccount_Transfer_Out_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_Out_Amount'], data_entrenamiento['SavingAccount_Total_Amount']), 
'SavingAccount_DebitCard_Spend_Amount_porc': compute_percentage(data_entrenamiento['SavingAccount_DebitCard_Spend_Amount'], data_entrenamiento['SavingAccount_Total_Amount']), 

'SavingAccount_Salary_Payment_Amount_CR_porc': compute_percentage(data_entrenamiento['SavingAccount_Salary_Payment_Amount'], data_entrenamiento['SavingAccount_Credits_Amounts']), 
'SavingAccount_Transfer_In_Amount_CR_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_In_Amount'], data_entrenamiento['SavingAccount_Credits_Amounts']), 
'SavingAccount_ATM_Extraction_Amount_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_ATM_Extraction_Amount'], data_entrenamiento['SavingAccount_Debits_Amounts']), 
'SavingAccount_Service_Payment_Amount_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_Service_Payment_Amount'], data_entrenamiento['SavingAccount_Debits_Amounts']), 
'SavingAccount_CreditCard_Payment_Amount_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_CreditCard_Payment_Amount'], data_entrenamiento['SavingAccount_Debits_Amounts']), 
'SavingAccount_Transfer_Out_Amount_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_Transfer_Out_Amount'], data_entrenamiento['SavingAccount_Debits_Amounts']), 
'SavingAccount_DebitCard_Spend_Amount_DE_porc': compute_percentage(data_entrenamiento['SavingAccount_DebitCard_Spend_Amount'], data_entrenamiento['SavingAccount_Debits_Amounts']), 










# OPERATION
'Operations_digitales_porc': compute_percentage(data_entrenamiento['Operations_digitales'], data_entrenamiento['Operations_total']), 
'Operations_presenciales_porc': compute_percentage(data_entrenamiento['Operations_presenciales'], data_entrenamiento['Operations_total']), 

'Operations_Bank_porc': compute_percentage(data_entrenamiento['Operations_Bank'], data_entrenamiento['Operations_total']), 
'Operations_Terminal_porc': compute_percentage(data_entrenamiento['Operations_Terminal'], data_entrenamiento['Operations_total']), 
'Operations_HomeBanking_porc': compute_percentage(data_entrenamiento['Operations_HomeBanking'], data_entrenamiento['Operations_total']), 
'Operations_Mobile_porc': compute_percentage(data_entrenamiento['Operations_Mobile'], data_entrenamiento['Operations_total']), 
'Operations_Ivr_porc': compute_percentage(data_entrenamiento['Operations_Ivr'], data_entrenamiento['Operations_total']), 
'Operations_Telemarketer_porc': compute_percentage(data_entrenamiento['Operations_Telemarketer'], data_entrenamiento['Operations_total']), 
'Operations_ATM_porc': compute_percentage(data_entrenamiento['Operations_ATM'], data_entrenamiento['Operations_total']), 


'Operations_Bank_P_porc': compute_percentage(data_entrenamiento['Operations_Bank'], data_entrenamiento['Operations_presenciales']), 
'Operations_Terminal_P_porc': compute_percentage(data_entrenamiento['Operations_Terminal'], data_entrenamiento['Operations_presenciales']), 
'Operations_HomeBanking_D_porc': compute_percentage(data_entrenamiento['Operations_HomeBanking'], data_entrenamiento['Operations_digitales']), 
'Operations_Mobile_D_porc': compute_percentage(data_entrenamiento['Operations_Mobile'], data_entrenamiento['Operations_digitales']), 
'Operations_Ivr_D_porc': compute_percentage(data_entrenamiento['Operations_Ivr'], data_entrenamiento['Operations_digitales']), 
'Operations_Telemarketer_D_porc': compute_percentage(data_entrenamiento['Operations_Telemarketer'], data_entrenamiento['Operations_digitales']), 
'Operations_ATM_P_porc': compute_percentage(data_entrenamiento['Operations_ATM'], data_entrenamiento['Operations_presenciales']), 








# CREDIT CARD
'CreditCard_Payment_digitales_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_digitales'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_presenciales_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_presenciales'], data_entrenamiento['CreditCard_Payment_total']), 


'CreditCard_Payment_Aut_Debit_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Aut_Debit'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_External_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_External'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_Cash_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Cash'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_Web_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Web'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_ATM_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_ATM'], data_entrenamiento['CreditCard_Payment_total']), 
'CreditCard_Payment_TAS_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_TAS'], data_entrenamiento['CreditCard_Payment_total']), 

'CreditCard_Payment_Aut_Debit_D_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Aut_Debit'], data_entrenamiento['CreditCard_Payment_digitales']), 
'CreditCard_Payment_External_P_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_External'], data_entrenamiento['CreditCard_Payment_presenciales']), 
'CreditCard_Payment_Cash_P_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Cash'], data_entrenamiento['CreditCard_Payment_presenciales']), 
'CreditCard_Payment_Web_D_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_Web'], data_entrenamiento['CreditCard_Payment_digitales']), 
'CreditCard_Payment_ATM_P_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_ATM'], data_entrenamiento['CreditCard_Payment_presenciales']), 
'CreditCard_Payment_TAS_P_porc': compute_percentage(data_entrenamiento['CreditCard_Payment_TAS'], data_entrenamiento['CreditCard_Payment_presenciales']), 


'CreditCard_Balance_ARG_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Balance_ARG'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Balance_DOLLAR_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Balance_DOLLAR'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Total_Spending_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Total_Spending'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Spending_1_Installment_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_1_Installment'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Spending_Installments_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_Installments'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Spending_CrossBoarder_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_CrossBoarder'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Spending_Aut_Debits_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_Aut_Debits'], data_entrenamiento['CreditCard_Total_Limit']), 
'CreditCard_Revolving_limit_porc': compute_percentage(data_entrenamiento['CreditCard_Revolving'], data_entrenamiento['CreditCard_Total_Limit']), 

'CreditCard_Balance_ARG_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Balance_ARG'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Balance_DOLLAR_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Balance_DOLLAR'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Spending_1_Installment_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_1_Installment'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Spending_Installments_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_Installments'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Spending_CrossBoarder_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_CrossBoarder'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Spending_Aut_Debits_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Spending_Aut_Debits'], data_entrenamiento['CreditCard_Total_Spending']), 
'CreditCard_Revolving_SP_porc': compute_percentage(data_entrenamiento['CreditCard_Revolving'], data_entrenamiento['CreditCard_Total_Spending']), 






# OTROS
'Cantidad_Productos_Activos': (
    data_entrenamiento['CreditCard_Premium']
    + data_entrenamiento['CreditCard_Active']
    + data_entrenamiento['Loan_Active']
    + data_entrenamiento['Mortgage_Active']
    + data_entrenamiento['SavingAccount_Active_ARG_Salary']
    + data_entrenamiento['SavingAccount_Active_ARG']
    + data_entrenamiento['SavingAccount_Active_DOLLAR']
    + data_entrenamiento['DebitCard_Active']
    + data_entrenamiento['Investment_Active']
    + data_entrenamiento['Insurance_Life']
    + data_entrenamiento['Insurance_Home']
    + data_entrenamiento['Insurance_Accidents']
    + data_entrenamiento['Insurance_Mobile']
    + data_entrenamiento['Insurance_ATM']
    + data_entrenamiento['Insurance_Unemployment']
),

'Cantidad_Productos_Comunes_Activos': (
    data_entrenamiento['CreditCard_Active']
    + data_entrenamiento['SavingAccount_Active_ARG']
    + data_entrenamiento['SavingAccount_Active_DOLLAR']
    + data_entrenamiento['DebitCard_Active']
),

'Cantidad_Productos_Mas_Comunes_Activos': (
    data_entrenamiento['CreditCard_Active']
    + data_entrenamiento['SavingAccount_Active_ARG']
    + data_entrenamiento['DebitCard_Active']
),


}




data_entrenamiento = pd.concat(
    [data_entrenamiento, pd.DataFrame(transformadas_2)],
    axis='columns'
)

print(data_entrenamiento.shape)
```

```python
describe_full(data_entrenamiento)
```

```python
data_entrenamiento[[
    "client_id", 
    'SavingAccount_Transactions_Transactions', 
    "Operations_total", 
    "CreditCard_Payment_total"
]] [
    (data_entrenamiento["SavingAccount_Transactions_Transactions"] != 0) &
    (data_entrenamiento["Operations_total"] != 0) &
    (data_entrenamiento["CreditCard_Payment_total"] != 0)
    ]
```

```python
greater_than_one_hundred_columns = [
    "SavingAccount_Transfer_In_Amount_porc",
    "SavingAccount_Transfer_In_Amount_CR_porc",
    "SavingAccount_Balance_last_minus_first_date_porc"
]

for column in greater_than_one_hundred_columns:
    print_threshold_violations(data_entrenamiento, column_name=column, threshold=100, condition="gt")
```

# Aggregate Features

```python
columnas_con_valores_monetarios = [
    'SavingAccount_Balance_FirstDate', 
    'SavingAccount_Balance_LastDate', 
    'SavingAccount_Balance_Average',
    'SavingAccount_Salary_Payment_Amount',
    'SavingAccount_Transfer_In_Amount',
    'SavingAccount_ATM_Extraction_Amount',
    'SavingAccount_Service_Payment_Amount',
    'SavingAccount_CreditCard_Payment_Amount',
    'SavingAccount_Transfer_Out_Amount',
    'SavingAccount_DebitCard_Spend_Amount', 
    'SavingAccount_Total_Amount',
    'SavingAccount_Credits_Amounts', 
    'SavingAccount_Debits_Amounts',
    
    'SavingAccount_Balance_last_minus_first_date',
    
    'CreditCard_Balance_ARG', 'CreditCard_Balance_DOLLAR', 
    'CreditCard_Total_Spending', 'CreditCard_Spending_1_Installment', 'CreditCard_Spending_Installments', 'CreditCard_Spending_CrossBoarder', 'CreditCard_Spending_Aut_Debits', 'CreditCard_Revolving'
]

columnas_con_cantidades = [ 
    x for x in data_entrenamiento.columns 
    if x not in columnas_con_valores_monetarios + list(data_if.columns) + ["Month", "First_product_dt", "Last_product_dt", "client_id", "Target"] 
]

# ordenar los registros de cada cliente por mes para que luego funcionen "first" y "last" correctamente
data_entrenamiento = data_entrenamiento.sort_values(['client_id', 'Month'])

# escalar y redondear valores monetarios para aplicar "nunique"
cols_monetarias_redondeadas = [f"{col}_rounded" for col in columnas_con_valores_monetarios]
data_entrenamiento[cols_monetarias_redondeadas] = data_entrenamiento[columnas_con_valores_monetarios].div(1000).round()

# Diccionario de agregaciones usando STRINGS nativos de Pandas (optimizados en C)
agg_funcs = ['min', 'max', 'mean', 'median', 'sum',
'var', # varianza
'std', # desviacion estandar
'first', 'last']

agg_dict = {}

# Para columnas de cantidades
for col in columnas_con_cantidades:
    agg_dict[col] = agg_funcs + ['nunique']
    
# Para columnas monetarias 
for col in columnas_con_valores_monetarios:
    agg_dict[col] = agg_funcs 

# Para nunique redondeado para columnas monetarias 
for col_round in cols_monetarias_redondeadas:
    agg_dict[col_round] = ['nunique'] 

```

```python
data_agg = data_entrenamiento.groupby('client_id').agg(agg_dict)
```

```python
# VECTORIZACIÓN MATRICIAL

# Extraer DataFrames completos filtrando por el nivel de agregación.
cols_base = columnas_con_cantidades + columnas_con_valores_monetarios

# Crear matriz de True/False (!= 0), agrupar por cliente y sumar. 
# Esto cuenta los no-ceros de forma optimizada
df_nonzero = (data_entrenamiento[cols_base] != 0).groupby(data_entrenamiento['client_id']).sum()
df_nonzero.columns = [f"{col}_count_nonzero" for col in df_nonzero.columns]

df_max = data_agg.loc[:, (cols_base, 'max')].copy()
df_min = data_agg.loc[:, (cols_base, 'min')].copy()
df_first = data_agg.loc[:, (cols_base, 'first')].copy()
df_last = data_agg.loc[:, (cols_base, 'last')].copy()

# Eliminar temporalmente el nivel de agregación (max, min...) de las columnas 
# para que los DataFrames queden alineados y se puedan operar entre sí.
df_max.columns = df_max.columns.droplevel(1)
df_min.columns = df_min.columns.droplevel(1)
df_first.columns = df_first.columns.droplevel(1)
df_last.columns = df_last.columns.droplevel(1)

# Operaciones matemáticas matriciales, se realizan sobre toda la matriz a la vez (más rápido)
# PTP: diferencia entre el valor máximo y el valor mínimo
df_ptp = df_max - df_min
# Diff: Diferencia entre último y primer mes (si el ultimo es menor al primero, esto da negativo)
df_diff = df_last - df_first

# Diferencia relativa: (último / primero)
df_diff_rel = compute_percentage(df_last, df_first)
# Variación porcentual: 1 - diferencia relativa
df_variacion = df_diff_rel - 100

# Volver a agregar el nivel (MultiIndex) para identificar las nuevas features
df_ptp.columns = pd.MultiIndex.from_product([df_ptp.columns, ['ptp']])
df_diff.columns = pd.MultiIndex.from_product([df_diff.columns, ['diff']])
df_diff_rel.columns = pd.MultiIndex.from_product([df_diff_rel.columns, ['diff_rel']])
df_variacion.columns = pd.MultiIndex.from_product([df_variacion.columns, ['variacion_porc']])

# Unir todo
data_agg = pd.concat([data_agg, df_ptp, df_diff, df_diff_rel, df_variacion], axis='columns')
```

```python
# Borrar 'first' y 'last' que solo se usaban como auxiliares para los cálculos anteriores
data_agg = data_agg.drop(columns=['first', 'last'], level=1)

# Aplanar el MultiIndex de las columnas (ej: de ('CreditCard_Total', 'min') a 'CreditCard_Total_min')
data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns.values]

# Unir el df_nonzero (que ya estaba aplanado)
data_agg = pd.concat([data_agg, df_nonzero], axis='columns')
```

```python
# TODO: SACAR ESTA CELDA SI PODES: sin esto la matriz de correlaciones saca las variables que necesito
# para sacar esto tendria que agregar manualmente las variables que necesito, o usar otras variables (correlacionadas)

# Restaurar el orden original de las columnas
orden_original = []

for col in columnas_con_cantidades:
    for f in ['min', 'max', 'mean', 'median', 'sum', 'count_nonzero', 'var', 'std', 'nunique']:
        orden_original.append(f"{col}_{f}")

for col in columnas_con_valores_monetarios:
    for f in ['min', 'max', 'mean', 'median', 'sum', 'count_nonzero', 'var', 'std']:
        orden_original.append(f"{col}_{f}")

for col in cols_monetarias_redondeadas:
    orden_original.append(f"{col}_nunique")

# Agregar las derivadas al final
cols_derivadas = [c for c in data_agg.columns if c.endswith(('_ptp', '_diff', '_diff_rel', '_variacion_porc'))]

# Reordenar el DataFrame para que la matriz de correlación funcione igual
data_agg = data_agg[orden_original + cols_derivadas]
```

```python
# reiniciar índice (esto pasa 'client_id' a ser una columna)
data_agg = data_agg.copy().reset_index()

# borrar las columnas "_rounded" auxiliares de data_entrenamiento para liberar RAM
data_entrenamiento = data_entrenamiento.drop(columns=cols_monetarias_redondeadas)

print(data_agg.shape)
```

# ABT

```python
ABT = data_if.merge(data_agg, how='inner', on='client_id')
```

```python
print_df_personalizado("ABT", ABT)
```

## Agrego transformadas extras luego de las operaciones de agregacion

```python
sumatoria_usos = (
    min_max_normalize(ABT["SavingAccount_Days_with_use_count_nonzero"])
    + min_max_normalize(ABT["SavingAccount_Days_with_use_min"])
    + min_max_normalize(ABT["SavingAccount_CreditCard_Payment_Transactions_count_nonzero"])
    + min_max_normalize(ABT["Operations_total_count_nonzero"])
    + min_max_normalize(ABT["CreditCard_Payment_total_max"])
    + min_max_normalize(ABT["CreditCard_Payment_presenciales_max"])
    + (ABT["Operations_presenciales_porc_max"] > 0).astype(int)
    + (ABT["CreditCard_Payment_Aut_Debit_max"] > 0).astype(int)
    + (ABT["CreditCard_Payment_TAS_max"] > 0).astype(int)
)

ABT = ABT.assign(SUMATORIA_USOS=sumatoria_usos)
```

```python
ABT = ABT.assign(

Amount_operations = min_max_normalize_weighted(
    ABT["SavingAccount_CreditCard_Payment_Amount_max"], ABT["Operations_total_count_nonzero"]
),

Amount_transactions = min_max_normalize_weighted(
    ABT["SavingAccount_CreditCard_Payment_Amount_max"], ABT["SavingAccount_CreditCard_Payment_Transactions_count_nonzero"]
),

Amount_payment = min_max_normalize_weighted(
    ABT["SavingAccount_CreditCard_Payment_Amount_max"], ABT["CreditCard_Payment_total_max"]
),

Limit_operations = min_max_normalize_weighted(
    ABT["CreditCard_Total_Limit_diff_rel"], ABT["Operations_total_count_nonzero"]
),

Limit_transactions = min_max_normalize_weighted(
    ABT["CreditCard_Total_Limit_diff_rel"], ABT["SavingAccount_CreditCard_Payment_Transactions_count_nonzero"]
),

Limit_payment = min_max_normalize_weighted(
    ABT["CreditCard_Total_Limit_diff_rel"], ABT["CreditCard_Payment_total_max"]
),

)
```

```python
print(ABT["SUMATORIA_USOS"].isnull().sum())
print(ABT["Amount_operations"].isnull().sum())
print(ABT["Amount_transactions"].isnull().sum())
print(ABT["Amount_payment"].isnull().sum())
print(ABT["Limit_operations"].isnull().sum())
print(ABT["Limit_transactions"].isnull().sum())
print(ABT["Limit_payment"].isnull().sum())
```

```python
print_df_personalizado("ABT", ABT)
```

```python
ABT_sin_client_id = ABT.select_dtypes(include=['number'])
print(ABT_sin_client_id.shape)
```

```python
print(columnas_minimo_entre(ABT, -1, 1))
```

# Reduccion de dimensionalidad


## Elimino columnas con valores unicos

```python
columnas_con_describe = pd.DataFrame(ABT.describe().T)
columnas_con_valores_unicos = columnas_con_describe[columnas_con_describe['min'] == columnas_con_describe['max']].reset_index()
print(columnas_con_valores_unicos)
print("")
print('ABT original: ' , ABT.shape)
ABT_reducida = ABT.drop(columns=columnas_con_valores_unicos['index'])
print('ABT sin columnas con valores unicos: ' , ABT_reducida.shape)
```

## Elimino columnas binarias con baja representatividad

```python
cols_binarias = [col for col in ABT_reducida.columns if ABT_reducida[col].nunique(dropna=False) == 2]
cols_binarias.remove("Target")
print(cols_binarias)
print("")

pocos_representativos = list()

for x in cols_binarias:
    poco_representativo = ABT_reducida[x].value_counts(normalize=True)
    poco_representativo = poco_representativo[poco_representativo < 0.1].index.tolist()
    if poco_representativo:
        pocos_representativos.append(x)

for x in pocos_representativos:
    print(ABT_reducida[x].value_counts(normalize=True))
    print("")

ABT_reducida = ABT_reducida.drop(columns=pocos_representativos)
print('ABT sin columnas binarias poco representativas:' , ABT_reducida.shape)
```

## Elimino columnas correlacionadas entre si

```python
features_temp = ABT_reducida.drop(columns=['client_id', 'Target'])

# Convertir el DataFrame a un array de Numpy, calcular la correlación y volver a Pandas
features_array = features_temp.to_numpy(dtype='float64')

# para usar np.corrcoef() es importante que la ABT_reducida ya no tenga valores nulos
matriz_corr_np = np.abs(np.corrcoef(features_array, rowvar=False))
matriz_corr = pd.DataFrame(matriz_corr_np, columns=features_temp.columns, index=features_temp.columns)

# Selecciono triangulo superior de la matriz de correlacion
triangulo_superior = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)) 

# VECTORIZACIÓN: Calcular el máximo absoluto de cada columna una sola vez
# el método .max() ignora los NaNs
maximos_por_columna = triangulo_superior.max()

# Filtrar usando máscaras booleanas directamente en el índice (más rápido):
# Buscar columnas con correlacion mayor a 70%
a_borrar_1 = maximos_por_columna[maximos_por_columna > 0.70].index.tolist()
# Buscar columnas con correlacion mayor a 80%
a_borrar_2 = maximos_por_columna[maximos_por_columna > 0.80].index.tolist()
# Buscar columnas con correlacion mayor a 90%
a_borrar_3 = maximos_por_columna[maximos_por_columna > 0.90].index.tolist()

print(f"columnas con correlacion mayor a 70%: {len(a_borrar_1)}")
print(f"columnas con correlacion mayor a 80%: {len(a_borrar_2)}")
print(f"columnas con correlacion mayor a 90%: {len(a_borrar_3)}")

# borrar columnas con correlacion mayor a 80%
ABT_reducida_2 = ABT_reducida.drop(columns=a_borrar_2)
print("ABT sin columnas con correlacion mayor a 80%:", ABT_reducida_2.shape)
```

## Estandarizacion con z-score

```python
from sklearn.preprocessing import StandardScaler

columnas_sin_client_id_ni_target = [x for x in ABT_reducida_2.columns if (x != 'client_id') & (x != 'Target')]
scaler = StandardScaler(copy=True)
scaler.fit(ABT_reducida_2[columnas_sin_client_id_ni_target]) # Entrena
datos_estandarizados = scaler.transform(ABT_reducida_2[columnas_sin_client_id_ni_target]) # Standariza el total de la base
datos_estandarizados = pd.DataFrame(datos_estandarizados, columns=columnas_sin_client_id_ni_target, index=ABT_reducida_2.index)

ABT_estandarizado = ABT_reducida_2.drop(columns=columnas_sin_client_id_ni_target)
ABT_estandarizado = pd.concat((ABT_estandarizado, datos_estandarizados), axis='columns', sort=False)
```

```python
print_df_personalizado("ABT_estandarizado", ABT_estandarizado)
```

### Model training

```python
from bank_clients_ml.models import generar_split, generar_modelo_y_buscador, process_model_results
```

```python
grupos = group_columns_by_source(ABT_estandarizado)
columnas_saving_account_days_transactions = grupos["saving_account_days_transactions"]
columnas_saving_account_monetarios = grupos["saving_account_monetary"]
columnas_operation = grupos["operations"]
columnas_credit_card_payment = grupos["credit_card_payment"]
columnas_credit_card_monetarios = grupos["credit_card_monetary"]
columnas_otros = grupos["others"]

print("columnas_saving_account_days_transactions:", len(columnas_saving_account_days_transactions))
print("columnas_saving_account_monetarios:", len(columnas_saving_account_monetarios))
print("columnas_operation:", len(columnas_operation))
print("columnas_credit_card_payment:", len(columnas_credit_card_payment))
print("columnas_credit_card_monetarios:", len(columnas_credit_card_monetarios))
print("columnas_otros:", len(columnas_otros))
print("total deberia ser igual a", len(ABT_estandarizado.columns), "- 2:", 
      len(columnas_saving_account_days_transactions) + len(columnas_saving_account_monetarios) + len(columnas_operation) + len(columnas_credit_card_payment) + len(columnas_credit_card_monetarios) + len(columnas_otros))
print("")
print(columnas_otros)
```

## Ordeno las variables por fuente segun importancia usando lightGBM para quedarme con las mas importantes

```python
X_train, X_test = generar_split(ABT_estandarizado)
```

```python
# todas las variables, para tener roc de referencia
todas_las_columnas = (
    columnas_saving_account_days_transactions +
    columnas_saving_account_monetarios +
    columnas_operation +
    columnas_credit_card_payment +
    columnas_credit_card_monetarios +
    columnas_otros
)

_, _, variables_mas_importantes_0 = train_and_get_feature_importances(
    X_train, todas_las_columnas
)
```

```python
# SAVING ACCOUNT DAYS TRANSACTIONS
_, _, variables_mas_importantes_1 = train_and_get_feature_importances(
    X_train, columnas_saving_account_days_transactions
)
```

```python
# SAVING ACCOUNT MONETARIOS
_, _, variables_mas_importantes_2 = train_and_get_feature_importances(
    X_train, columnas_saving_account_monetarios
)
```

```python
# OPERATION
_, _, variables_mas_importantes_3 = train_and_get_feature_importances(
    X_train, columnas_operation
)
```

```python
# CREDIT CARD PAYMENT
_, _, variables_mas_importantes_4 = train_and_get_feature_importances(
    X_train, columnas_credit_card_payment
)
```

```python
# CREDIT CARD MONETARIOS
_, _, variables_mas_importantes_5 = train_and_get_feature_importances(
    X_train, columnas_credit_card_monetarios
)
```

```python
# OTROS
_, _, variables_mas_importantes_6 = train_and_get_feature_importances(
    X_train, columnas_otros
)
```

```python
variables_mas_importantes_0.nlargest(60).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_0.nlargest(100)
```

```python
variables_mas_importantes_1.nlargest(40).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_1.nlargest(20)
```

```python
variables_mas_importantes_2.nlargest(60).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_2.nlargest(20)
```

```python
variables_mas_importantes_3.nlargest(40).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_3.nlargest(20)
```

```python
variables_mas_importantes_4.nlargest(25).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_4.nlargest(20)
```

```python
variables_mas_importantes_5.nlargest(60).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_5.nlargest(20)
```

```python
variables_mas_importantes_6.nlargest(40).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_6.nlargest(20)
```

# Analisis Bivariado

```python
from bank_clients_ml.graphs import Graficar_Variables, graficar_top_20

destino_analisis_bivariado = 'variables'
```

```python
ABT_reducida_3 = ABT_reducida.copy()
```

```python
columnas_a_graficar = [
    "SavingAccount_Days_with_use_count_nonzero",                      
    "SavingAccount_Transfer_In_Transactions_count_nonzero",           
    "SavingAccount_Transfer_In_Transactions_max",                    
    "SavingAccount_Days_with_use_min",                                
    "SavingAccount_Days_with_Credits_porc_var",                       
    "SavingAccount_CreditCard_Payment_Transactions_max",              
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",     
    
    "SavingAccount_Balance_FirstDate_max",                     
    "SavingAccount_CreditCard_Payment_Amount_max",             
    "SavingAccount_Transfer_In_Amount_max",                    
    "SavingAccount_Total_Amount_min",                           
    "SavingAccount_Total_Amount_diff",                          
    "SavingAccount_Balance_LastDate_diff_rel",                  
    
    "Operations_total_count_nonzero",           
    "Operations_total_min",                     
    "Operations_total_var",                     
    "Operations_Telemarketer_porc_max",        
    "Operations_presenciales_porc_min",         
    "Operations_presenciales_porc_max",        
    
    "CreditCard_Payment_total_max",                   
    "CreditCard_Payment_Aut_Debit_max",                
    "CreditCard_Payment_total_min",                    
    "CreditCard_Payment_TAS_max",                     
    "CreditCard_Payment_Cash_max",                     
    "CreditCard_Payment_Web_max",                      
    "CreditCard_Payment_Aut_Debit_min",                 
    "CreditCard_Payment_Aut_Debit_diff",                
    "CreditCard_Payment_ATM_max",                      
    "CreditCard_Payment_presenciales_porc_diff_rel",    

    "CreditCard_Payment_presenciales_max",

    "CreditCard_Total_Limit_var",                                    
    "CreditCard_Total_Limit_diff_rel",                               
    "CreditCard_Balance_ARG_SP_porc_max",                           
    "CreditCard_Total_Limit_min",                                    
    "CreditCard_Revolving_min",                                      
    "CreditCard_Total_Spending_diff_rel",                            
    "CreditCard_Spending_Aut_Debits_diff_rel",                        
    
    "CreditCard_Product",                                  
    "Recencia_en_dias",                                    
    "Dias_entre_primer_y_ultimo_producto",                 
    "Client_Age_grp",                                      
    "Cantidad_Productos_Activos_min",                      
    "Cantidad_Productos_Activos_nunique",                  
    "SavingAccount_Active_ARG_Salary",                     
    "Sex",                                                 
    "SavingAccount_Active_DOLLAR",                         
    "Region",                                              
    "Cantidad_Productos_Activos_var",                      
    "Investment_Numbers_max",                             
    "Email",                                               
    
    "Cantidad_Productos_Comunes_Activos_count_nonzero",     

    "client_id", "Target"
]

Graficar_Variables(
    ABT_reducida_3[columnas_a_graficar], # dataset con variables sin standarizar
    "client_id",
    "Target",
    20, # cantidad de bines
    destino_analisis_bivariado,
    "Analisis"
)
```

```python
ABT_2 = ABT.copy()

graf = [
    "CreditCard_Payment_total_var",
    
    "Limit_operations",

    "SavingAccount_CreditCard_Payment_Amount_max",
    "CreditCard_Total_Limit_diff_rel",
    
    "SavingAccount_Days_with_use_count_nonzero",
    "SavingAccount_Days_with_use_min",
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",

    "Operations_total_count_nonzero",
    "Operations_presenciales_porc_max",

    "CreditCard_Payment_total_max",
    "CreditCard_Payment_Aut_Debit_max",
    "CreditCard_Payment_presenciales_max",

    "CreditCard_Product",
    "Dias_entre_primer_y_ultimo_producto",
    "Client_Age_grp",
    "Cantidad_Productos_Activos_min",
    "Recencia_en_dias",

    "client_id", "Target"
]

Graficar_Variables(
    ABT_2[graf], # dataset con variables sin standarizar
    "client_id",
    "Target",
    10, # cantidad de bines
    destino_analisis_bivariado,
    "Analisis_2"
)
```

## Re-entreno con las mejores variables

```python
prueba = [
    "SavingAccount_Days_with_use_count_nonzero",                      
    "SavingAccount_Transfer_In_Transactions_count_nonzero",           
    "SavingAccount_Transfer_In_Transactions_max",                    
    "SavingAccount_Days_with_use_min",                                
    "SavingAccount_Days_with_Credits_porc_var",                       
    "SavingAccount_CreditCard_Payment_Transactions_max",              
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",                       
    
    "Operations_total_count_nonzero",           
    "Operations_total_min",                     
    "Operations_total_var",                     
    "Operations_Telemarketer_porc_max",        
    "Operations_presenciales_porc_min",         
    "Operations_presenciales_porc_max",        

    "CreditCard_Payment_total_var",
    "CreditCard_Payment_total_max",                   
    "CreditCard_Payment_Aut_Debit_max",                
    "CreditCard_Payment_total_min",                    
    "CreditCard_Payment_TAS_max",                     
    "CreditCard_Payment_Cash_max",                     
    "CreditCard_Payment_Web_max",                      
    "CreditCard_Payment_Aut_Debit_min",                 
    "CreditCard_Payment_Aut_Debit_diff",                
    "CreditCard_Payment_ATM_max",                      
    "CreditCard_Payment_presenciales_porc_diff_rel",    

    "CreditCard_Payment_presenciales_max"
]

_, _, variables_mas_importantes_7 = train_and_get_feature_importances(
    X_train, prueba
)
```

```python
variables_mas_importantes_7.nlargest(25).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_7.nlargest(25)
```

## Re-entreno con las mejores variables

```python
prueba_2 = [
    "SavingAccount_Days_with_use_count_nonzero",
    "SavingAccount_Days_with_use_min",
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",

    "SavingAccount_CreditCard_Payment_Amount_max",

    "Operations_total_count_nonzero",
    "Operations_presenciales_porc_max",

    "CreditCard_Payment_total_max",
    "CreditCard_Payment_Aut_Debit_max",
    "CreditCard_Payment_presenciales_max",

    "CreditCard_Total_Limit_diff_rel",

    "CreditCard_Product",
    "Dias_entre_primer_y_ultimo_producto",
    "Client_Age_grp",
    "Cantidad_Productos_Activos_min",
]

_, _, variables_mas_importantes_8 = train_and_get_feature_importances(
    X_train, prueba_2
)
```

```python
variables_mas_importantes_8.nlargest(25).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_8.nlargest(25)
```

## Buscando variables correlacionadas eliminadas anteriormente

```python
# si estan correlacionadas se pueden intercambiar sin problemas
# la idea es agarrar variables faciles de explicar a gente no tecnica

for x in [
    "SavingAccount_CreditCard_Payment_Amount_max",

    "Operations_total_count_nonzero",

    "CreditCard_Total_Limit_diff_rel",

    "CreditCard_Product",
    "Dias_entre_primer_y_ultimo_producto",
    "Client_Age_grp",
    "Cantidad_Productos_Activos_min"    
]:
    print("Columnas correlacionadas con", x, ":")
    print(triangulo_superior[x][triangulo_superior[x] > 0.50])
    print("")
```

## Re-entreno con las mejores variables

```python
prueba_3 = [                      
    #"SavingAccount_CreditCard_Payment_Amount_max",
    #"CreditCard_Total_Limit_diff_rel",
    #"Dias_entre_primer_y_ultimo_producto",
    #"Cantidad_Productos_Activos_min"
    "Operations_total_count_nonzero",
    "CreditCard_Product",
    "Client_Age_grp"
]

_, _, variables_mas_importantes_9 = train_and_get_feature_importances(
    X_train, prueba_3
)
```

```python
variables_mas_importantes_9.nlargest(25).plot(kind='barh', figsize=(8,10))
```

```python
variables_mas_importantes_9.nlargest(25)
```

# Transformando mejores variables segun analisis bivariado y LightGBM

```python
# transformo variables agregando porcentaje de target a cada variable y agrupo valores. y agrupo variables categoricas (ya las transforme anteriormente)
# hice los calculos en un excel

# CreditCard_Product
# junto los tipos de tarjetas de bajo porcentaje de target y los tipos de tarjetas poco representativas en un solo bin
ABT_reducida_3['CreditCard_Product_t'] = apply_binning_by_ranges(
    ABT_reducida_3['CreditCard_Product'],
    ranges=[(36.890, 36.950), (45.680, 45.690)],  # tipo tarjeta 202 y 104 respectivamente
    values=[36.940, 45.686],
    default=9.005,
)
# default -> agrupo totas las demas tarjetas (sin tarjeta de credito + 102 + 123 + 124 + 702 + 1002)
print(ABT_reducida_3['CreditCard_Product_t'].value_counts())
print("")

# Client_Age_grp
ABT_reducida_3['Client_Age_grp_t'] = apply_binning_by_ranges(
    ABT_reducida_3['Client_Age_grp'],
    ranges=[(34.880, 39.772)], # agrupo "Entre 50 y 59 años" + "Entre 60 y 64 años" + "Entre 65 y 69 años" (final: "Entre 50 y 69 años")
    values=[36.224],
    default=25.093,
)
# default -> totas las demas edades ("Entre 18 y 29 años" + "Entre 30 y 39 años" + "Entre 40 y 49 años" + "Mayor a 70 años")
print(ABT_reducida_3['Client_Age_grp_t'].value_counts())
print("")

# Operations_total_count_nonzero
ABT_reducida_3['Operations_total_count_nonzero_t'] = apply_binning_by_ranges(
    ABT_reducida_3['Operations_total_count_nonzero'],
    ranges=[(0, 0), (1, 3), (4, 5)],
    values=[10.148, 18.896, 29.131],
    default=47.939,
)
print(ABT_reducida_3['Operations_total_count_nonzero_t'].value_counts())
print("")


# # Region
# ABT_reducida_3['Region_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['Region'],
#     ranges=[(24.370, 24.375)],  # mantengo "REGION CENTRO"
#     values=[24.372],
#     default=30.663,
# )
# # default -> totas las demas regiones (NORTE GRANDE ARGENTINO + CUYO + CABA Centro/Norte + AMBA Resto + BUENOS AIRES + REGION PATAGONICA)
# print(ABT_reducida_3['Region_t'].value_counts())
# print("")

# # Cantidad_Productos_Activos_min
# ABT_reducida_3['Cantidad_Productos_Activos_min_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['Cantidad_Productos_Activos_min'],
#     ranges=[(0, 3), (5, 8)],
#     values=[17.664, 61.729],
#     default=40.000,
# )
# print(ABT_reducida_3['Cantidad_Productos_Activos_min_t'].value_counts())
# print("")

# # Operations_presenciales_max
# ABT_reducida_3['Operations_presenciales_max_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['Operations_presenciales_max'],
#     ranges=[(1, 2), (3, 44)],
#     values=[36.971, 54.786],
#     default=17.000,
# )
# print(ABT_reducida_3['Operations_presenciales_max_t'].value_counts())
# print("")

# # Dias_entre_primer_y_ultimo_producto
# ABT_reducida_3['Dias_entre_primer_y_ultimo_producto_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['Dias_entre_primer_y_ultimo_producto'],
#     ranges=[(0, 441), (442, 1142), (1143, 2130)],
#     values=[21.764, 25.244, 33.003],
#     default=48.858,
# )
# print(ABT_reducida_3['Dias_entre_primer_y_ultimo_producto_t'].value_counts())
# print("")

# # Recencia_en_dias
# ABT_reducida_3['Recencia_en_dias_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['Recencia_en_dias'],
#     ranges=[(1, 408), (409, 650)],
#     values=[33.822, 29.220],
#     default=23.845,
# )
# print(ABT_reducida_3['Recencia_en_dias_t'].value_counts())
# print("")

# # CreditCard_Total_Spending_median
# ABT_reducida_3['CreditCard_Total_Spending_median_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['CreditCard_Total_Spending_median'],
#     ranges=[(0.5, 1979.9), (1980.2, 4078.7), (4079.0, 117452)],
#     values=[33.866, 41.667, 46.154],
#     default=9.000,
# )
# print(ABT_reducida_3['CreditCard_Total_Spending_median_t'].value_counts())
# print("")

# # SavingAccount_Balance_Average_median
# ABT_reducida_3['SavingAccount_Balance_Average_median_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['SavingAccount_Balance_Average_median'],
#     ranges=[(163.3, 2823.9), (2824.0, 1515662.7)],
#     values=[30.999, 50.143],
#     default=22.243,
# )
# print(ABT_reducida_3['SavingAccount_Balance_Average_median_t'].value_counts())
# print("")

# # SavingAccount_Transactions_Transactions_median
# ABT_reducida_3['SavingAccount_Transactions_Transactions_median_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['SavingAccount_Transactions_Transactions_median'],
#     ranges=[(0, 3), (3.5, 7)],
#     values=[21.781, 31.686],
#     default=54.346,
# )
# print(ABT_reducida_3['SavingAccount_Transactions_Transactions_median_t'].value_counts())
# print("")

# # SavingAccount_CreditCard_Payment_Amount_median
# ABT_reducida_3['SavingAccount_CreditCard_Payment_Amount_median_t'] = apply_binning_by_ranges(
#     ABT_reducida_3['SavingAccount_CreditCard_Payment_Amount_median'],
#     ranges=[(0, 0)],
#     values=[21.000],
#     default=55.253,
# )
# print(ABT_reducida_3['SavingAccount_CreditCard_Payment_Amount_median_t'].value_counts())
# print("")
```

```python
mejores_variables = [
    "CreditCard_Product_t",
    "Client_Age_grp_t",
    "Operations_total_count_nonzero_t", 

    "client_id", "Target"
]

Graficar_Variables(
    ABT_reducida_3[mejores_variables], # dataset con variables sin standarizar
    "client_id",
    "Target",
    20,
    destino_analisis_bivariado,
    "Analisis_t"
)
```

```python
mejores_variables.remove("client_id")
mejores_variables.remove("Target")
print_df_personalizado("ABT_reducida_3", ABT_reducida_3)
```

# Modelado


## Vuelvo a correr con las mejores variables y mejores hiperparametros

```python
X_train_final, X_test_final = generar_split(ABT_reducida_3) # dataframe con variables transformadadas segun analisis bivariado
```

```python
_, buscador_mejores_hiperparametros, variables_mas_importantes = train_and_get_feature_importances(
    X_train_final, mejores_variables, n_iter=100
)
```

```python
variables_mas_importantes.nlargest(20).plot(kind='barh', figsize=(8,10))
```

## Hiperparametros optimos

```python
buscador_mejores_hiperparametros.best_estimator_
```

```python
dic = {
    'CreditCard_Product_t': 'Tipo de Tarjeta de crédito',
    "Client_Age_grp_t": "Tiene entre 50 y 69 años",
    "Operations_total_count_nonzero_t": "Cantidad de Meses con al menos una Operación",
}

variables_mas_importantes_renombradas = variables_mas_importantes.rename(index=dic)

graficar_top_20(variables_mas_importantes_renombradas)
```

# Performance del modelo

```python
modelo_LightGBM_clasificador_final = buscador_mejores_hiperparametros.best_estimator_

# Predice si es 0 o 1, si la probabilidad es > 0.5 lo pone como 1
y_pred = modelo_LightGBM_clasificador_final.predict(X_test_final[mejores_variables])

probabilities_train = modelo_LightGBM_clasificador_final.predict_proba(X_train_final[mejores_variables])
probabilities_test  = modelo_LightGBM_clasificador_final.predict_proba(X_test_final[mejores_variables])
```

```python
_ = evaluate_deciles_train(
    X_train_final, probabilities_train
)
```

```python
# Cotas fijas....
# basado en los porcentajes de training
cotas = [-np.inf, 0.035669, 0.054470, 0.204116, 0.241664, 0.269271, 0.393079, 0.443684, 0.541165, 0.548265, np.inf]

result_test = evaluate_deciles_test(
    X_test_final, probabilities_test, cotas
)
```

# ROC

```python
from bank_clients_ml.graphs import plot_roc_and_metrics

plot_roc_and_metrics(result_test["Target"], result_test["Prob1"], y_pred)
```

# Resultados del excel


## Training


- ordena casi todos los deciles bien
- deciles masomenos parejos
- lift del primer decil = 2,1
- KS = 43,1 en el 4to decil


## Testing


- ordena casi todos los deciles bien
- deciles masomenos parejos
- lift del primer decil = 2,1
- KS = 46,7 en el 4to decil 


## Diferencias 


- lift -> 0
- KS -> 3,6


# Verificando si el problema se puede resolver con una regresion logistica

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
modelo.fit(X_train_final[mejores_variables], X_train_final["Target"])

y_pred_log = modelo.predict(X_test_final[mejores_variables])
probabilities_train_log = modelo.predict_proba(X_train_final[mejores_variables])
probabilities_test_log       = modelo.predict_proba(X_test_final[mejores_variables])
```

```python
_ = evaluate_deciles_train(
    X_train_final, probabilities_train_log
)
```

```python
# Cotas fijas....
# basado en los porcentajes de training
cotas = [-np.inf, 0.051719, 0.079377, 0.170491, 0.197105, 0.267983, 0.366586, 0.467006, 0.579763, 0.580742, np.inf]

result_test_log = evaluate_deciles_test(
    X_test_final, probabilities_test_log, cotas
)
```

```python
plot_roc_and_metrics(result_test_log["Target"], result_test_log["Prob1"], y_pred_log)
```

```python

```

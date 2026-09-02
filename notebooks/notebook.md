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

# Setup inicial e imports

```python
from bank_clients_ml.notebook_config import setup_notebook

setup_notebook()
```

```python
import polars as pl

from bank_clients_ml.config import get_settings
from bank_clients_ml.features import (
    CorrelationAnalyzer,
    compute_percentage,
    get_constant_columns,
    get_date_windows,
    get_imbalanced_binary_columns,
    group_bins_by_ranges,
    group_columns_by_source,
    min_max_normalize,
    min_max_normalize_weighted,
    safe_denominator,
    standardize,
    target_encode_columns,
)
from bank_clients_ml.graphs import (
    generate_bivariate_charts,
    plot_roc_and_metrics,
    plot_top_features,
)
from bank_clients_ml.models import (
    compute_prediction_deciles,
    get_feature_importances,
    get_scoring,
    print_test_deciles,
    print_train_deciles,
    stratified_train_test_split,
)
from bank_clients_ml.utils import (
    columns_with_zeros,
    count_row_matches,
    filter_columns_by_cardinality,
    filter_nonzero,
    low_cardinality_value_counts,
    mins_in_range,
    print_without_trunc,
    scan_anomalies,
)

settings = get_settings()
```

# EDA (Exploratory Data Analysis)

```python
data = pl.read_parquet("../data/data.parquet")
```

```python
print(data.shape)
data.describe()
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

print(
    count_row_matches(data, columns=less_than_zero_columns, threshold=0, condition="<")
)
print(
    count_row_matches(
        data, columns=greater_than_thirty_one_columns, threshold=31, condition=">"
    )
)
```

```python
print_without_trunc(low_cardinality_value_counts(data))
```

```python
print_without_trunc(filter_columns_by_cardinality(data))
```

```python
print_without_trunc(scan_anomalies(data))
```

```python
print_without_trunc(data.null_count())
print_without_trunc(data.filter(pl.col(settings.col_target).is_null()))
print(data.filter(pl.col(settings.col_target).is_null()).select(settings.col_id))
```

## Limpieza inicial

```python
print(data.shape)
clean_data = data.filter(pl.col(settings.col_target).is_not_null())
print(clean_data.shape)
```

```python
print_without_trunc(scan_anomalies(clean_data))
```

```python
clean_data = clean_data.with_columns(
    pl.col("Month", "First_product_dt", "Last_product_dt").str.to_date(),
    pl.col(settings.col_id).cast(pl.Int64)
)
```

## Obtener meses relevantes

```python
training_months, prediction_months = get_date_windows(
    clean_data, "Month", prediction_window_size=2
)

last_training_month = training_months[-1]
first_prediction_month = prediction_months[0]

print("training_months:", training_months)
print("prediction_months:", prediction_months)
print("last_training_month:", last_training_month)
print("first_prediction_month:", first_prediction_month)
```

## Definir Universo y Target

```python
month_count_by_client = clean_data.group_by(settings.col_id).len(name="month_count")
print(month_count_by_client['month_count'].value_counts())
```

Mantengo en el universo los clientes que:
- tienen 9 meses de historia
- no tienen 'Package_Active' y 'CreditCard_CoBranding' en el ultimo mes de la ventana de entrenamiento

y para cada cliente del universo mantengo la columna de target de la ventana de prediccion

```python
is_valid_client = (
    (pl.col("Month") == last_training_month)
    & (pl.col("Package_Active") == "No")
    & (pl.col("CreditCard_CoBranding") == "No")
)

universe_and_target = (
    clean_data.filter(
        (pl.len().over(settings.col_id) == 9)
        & is_valid_client.any().over(settings.col_id)
        & pl.col("Month").is_in(prediction_months)
    )
    .select(settings.col_id, settings.col_target)
    .unique()
)

clean_data = clean_data.drop(settings.col_target).join(
    universe_and_target, on=settings.col_id, how="inner"
)

training_data = clean_data.filter(pl.col("Month").is_in(training_months))
prediction_data = clean_data.filter(pl.col("Month").is_in(prediction_months))

print(f"universe_and_target.shape: {universe_and_target.shape} \n")
print(f"training_data.shape: {training_data.shape} \n")
print(f"training_data['Month'].value_counts(): {training_data['Month'].value_counts()}")
print(f"prediction_data.shape: {prediction_data.shape} \n")
```

```python
scan_anomalies(training_data)
```

# Feature Engineering
## Valores Nulos

```python
null_cols = ["SavingAccount_Balance_Average", "Region", "CreditCard_Product"]
print(f"{training_data.select(null_cols).describe()} \n")
print(
    f"SavingAccount_Balance_Average n_unique: "
    f"{training_data.select('SavingAccount_Balance_Average').n_unique()} \n"
)
```

### Completando 'SavingAccount_Balance_Average'
Primero se analizan registros con nulos en SavingAccount_Balance_Average y valores monetarios de SavingAccount sin nulos:

```python
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
    "SavingAccount_Debits_Amounts",
]

training_data.select([settings.col_id, *saving_account_cols]).filter(
    pl.col("SavingAccount_Balance_Average").is_null()
)
```

```python
filter_nonzero(training_data, saving_account_cols)
```

Luego saco el promedio entre "SavingAccount_Balance_FirstDate" y "SavingAccount_Balance_LastDate".
Este no es el calculo correcto para "SavingAccount_Balance_Average", pero no va a afectar tanto al modelo porque son solo 4 registros con nulos ademas de que no hay una forma sencilla de calcular el "SavingAccount_Balance_Average" con los datos que tenemos

```python
training_data = training_data.with_columns(
    pl.col("SavingAccount_Balance_Average").fill_null(
        (
            pl.col("SavingAccount_Balance_FirstDate")
            + pl.col("SavingAccount_Balance_LastDate")
        )
        / 2.0
    )
)

print(f"{training_data.select('SavingAccount_Balance_Average').describe()} \n")
```

```python
training_data.shape
```

### Completando 'Region'
Traigo las regiones de  los clientes desde la ventana de prediccion y pongo la Region mas comun para llenar los nulos restantes

```python
clients_region = prediction_data.select(settings.col_id, "Region").unique()

training_data = training_data.drop("Region").join(
    clients_region.with_columns(pl.col("Region").fill_null("BUENOS AIRES")),
    on=settings.col_id,
    how="left",
)

print(f"{clean_data.select('Region').describe()} \n")
print(f"{clients_region.describe()} \n")
print(f"{clients_region.shape} \n")
print(f"{clients_region['Region'].value_counts(sort=True)} \n")
print(f"{training_data['Region'].value_counts(sort=True)} \n")
print(f"{training_data.select('Region').describe()} \n")
```

```python
training_data.shape
```

### Completando 'CreditCard_Product'

Traigo los CreditCard_Product de la ventana de prediccion. 

Hay algunos clientes que tienen un CreditCard_Product en el primer mes de prediccion y otro CreditCard_Product en el segundo mes de prediccion

Por lo tanto se obtiene el valor del primer mes (prediction_months[0]) y, si este es null o no existe, toma el valor del segundo mes (prediction_months[1]) como fallback, incluso si también es null

Luego para llenar los nulos restantes, pongo la CreditCard_Product mas comun cuando el cliente no tiene `CreditCard_Active` en la ventana de prediccion pero si tiene `CreditCard_Active` en la ventana de entrenamiento. en los demas casos lleno los nulls con "0" (cuando no tiene `CreditCard_Active` en la ventana de prediccion ni en la ventana de entrenamiento o cuando no tiene `CreditCard_Active` en la ventana de entrenamiento, por mas que lo tenga en la ventana de prediccion)

```python
clients_creditcard_product = (
    prediction_data.sort(pl.col("Month") == prediction_months[0], descending=True)
    .group_by(settings.col_id)
    .agg(pl.col("CreditCard_Product").drop_nulls().first())
)

training_data = (
    training_data.drop("CreditCard_Product")
    .join(
        clients_creditcard_product,
        on=settings.col_id,
        how="left",
    )
    .with_columns(
        pl.when(
            pl.col("CreditCard_Product").is_null()
            & (pl.col("CreditCard_Active") == "Yes")
        )
        .then(pl.col("CreditCard_Product").fill_null("J55660104XX012"))
        .otherwise(pl.col("CreditCard_Product").fill_null("0"))
        .alias("CreditCard_Product")
    )
)

print(f"{clean_data.select('CreditCard_Product').describe()} \n")
print(f"{clients_creditcard_product.describe()} \n")
print(f"{clients_creditcard_product['CreditCard_Product'].value_counts(sort=True)} \n")
print(f"{training_data['CreditCard_Product'].value_counts(sort=True)} \n")
print(f"{training_data.select('CreditCard_Product').describe()} \n")
```

```python
scan_anomalies(training_data)
```

```python
training_data.shape
```

## Identity Features

```python
binary_dict: dict[str, int] = {"Yes": 1, "No": 0, "M": 1, "F": 0}

binary_identity_features_columns = [
    "CreditCard_Premium",
    "CreditCard_Active",
    "CreditCard_CoBranding",
    "Loan_Active",
    "Mortgage_Active",
    "SavingAccount_Active_ARG_Salary",
    "SavingAccount_Active_ARG",
    "SavingAccount_Active_DOLLAR",
    "DebitCard_Active",
    "Investment_Active",
    "Package_Active",
    "Insurance_Life",
    "Insurance_Home",
    "Insurance_Accidents",
    "Insurance_Mobile",
    "Insurance_ATM",
    "Insurance_Unemployment",
    "Sex",
    "Mobile",
    "Email",
]

identity_features_columns = [
    settings.col_id,
    settings.col_target,
    "Client_Age_grp",
    "Region",
    "CreditCard_Product",
    "First_product_dt",
    "Last_product_dt",
    *binary_identity_features_columns,
]

training_data = training_data.with_columns(
    pl.col(binary_identity_features_columns).replace_strict(binary_dict).cast(pl.UInt8)
)

identity_features = training_data.filter(pl.col("Month") == last_training_month).select(
    identity_features_columns
)

print(identity_features.shape)
print(identity_features["CreditCard_Premium"].value_counts())
print(f"{identity_features["Sex"].value_counts()} \n")
```

```python
identity_features.schema
```

```python
identity_features.describe()
```

## Variables Categoricas

```python
categorical_cols = ["Client_Age_grp", "Region", "CreditCard_Product"]

print_without_trunc(
    low_cardinality_value_counts(identity_features.select(categorical_cols))
)

identity_features = target_encode_columns(identity_features, categorical_cols)

print_without_trunc(
    low_cardinality_value_counts(identity_features.select(categorical_cols))
)

print(f"identity_features: {identity_features.shape}")
print(f"training_data: {training_data.shape}")
training_data = training_data.drop(categorical_cols)
print(f"training_data: {training_data.shape}")
```

## Fechas

```python
identity_features = identity_features.with_columns(
    [
        (pl.col("Last_product_dt") - pl.col("First_product_dt"))
        .dt.total_days()
        .alias("Days_between_first_and_last_product"),
        (pl.lit(last_training_month).dt.offset_by("1mo") - pl.col("Last_product_dt"))
        .dt.total_days()
        .alias("Recency_in_days"),
    ]
).drop(["First_product_dt", "Last_product_dt"])

print(f"identity_features: {identity_features.shape} \n")
print(
    f"{
        identity_features.select(
            'Days_between_first_and_last_product', 'Recency_in_days'
        ).describe()
    }"
)
```

```python
identity_features.describe()
```

## Transform features
### Analizando valores minimos y ceros

```python
mins_in_range(training_data, -1, 1)
```

```python
print_without_trunc(columns_with_zeros(training_data))
```

### Analizando valores monetarios de CreditCard

```python
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

filter_nonzero(training_data, credit_card_cols)
```

### Generando transformaciones

```python
print(f"training_data antes de generar transformadas: {training_data.shape}")

training_data = training_data.with_columns(
    [
        # OPERATION
        (
            pl.col("Operations_Bank")
            + pl.col("Operations_Terminal")
            + pl.col("Operations_HomeBanking")
            + pl.col("Operations_Mobile")
            + pl.col("Operations_Ivr")
            + pl.col("Operations_Telemarketer")
            + pl.col("Operations_ATM")
        ).alias("Operations_total"),
        (
            pl.col("Operations_HomeBanking")
            + pl.col("Operations_Mobile")
            + pl.col("Operations_Ivr")
            + pl.col("Operations_Telemarketer")
        ).alias("Operations_remote"),
        (
            pl.col("Operations_Bank")
            + pl.col("Operations_Terminal")
            + pl.col("Operations_ATM")
        ).alias("Operations_in_person"),
        # CREDIT CARD
        (
            pl.col("CreditCard_Payment_Aut_Debit")
            + pl.col("CreditCard_Payment_External")
            + pl.col("CreditCard_Payment_Cash")
            + pl.col("CreditCard_Payment_Web")
            + pl.col("CreditCard_Payment_ATM")
            + pl.col("CreditCard_Payment_TAS")
        ).alias("CreditCard_Payment_total"),
        (
            pl.col("CreditCard_Payment_Aut_Debit") + pl.col("CreditCard_Payment_Web")
        ).alias("CreditCard_Payment_remote"),
        (
            pl.col("CreditCard_Payment_External")
            + pl.col("CreditCard_Payment_Cash")
            + pl.col("CreditCard_Payment_ATM")
            + pl.col("CreditCard_Payment_TAS")
        ).alias("CreditCard_Payment_in_person"),
        # SAVING ACCOUNT
        (
            pl.col("SavingAccount_Balance_LastDate")
            - pl.col("SavingAccount_Balance_FirstDate")
        ).alias("SavingAccount_Balance_last_minus_first_date"),
        compute_percentage(
            "SavingAccount_Balance_LastDate", "SavingAccount_Balance_FirstDate"
        ).alias("SavingAccount_Balance_last_minus_first_date_porc"),
        compute_percentage(
            "SavingAccount_Days_with_Debits", "SavingAccount_Days_with_use"
        ).alias("SavingAccount_Days_with_Debits_porc"),
        compute_percentage(
            "SavingAccount_Days_with_Credits", "SavingAccount_Days_with_use"
        ).alias("SavingAccount_Days_with_Credits_porc"),
        compute_percentage(
            "SavingAccount_Credits_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Credits_Transactions_porc"),
        compute_percentage(
            "SavingAccount_Debits_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Debits_Transactions_porc"),
        (
            pl.col("SavingAccount_Credits_Transactions")
            / safe_denominator("SavingAccount_Days_with_use")
        ).alias("SavingAccount_Transactions_Transactions_DAYS_prom"),
        (
            pl.col("SavingAccount_Credits_Transactions")
            / safe_denominator("SavingAccount_Days_with_Credits")
        ).alias("SavingAccount_Credits_Transactions_DAYS_prom"),
        (
            pl.col("SavingAccount_Debits_Transactions")
            / safe_denominator("SavingAccount_Days_with_Debits")
        ).alias("SavingAccount_Debits_Transactions_DAYS_prom"),
        compute_percentage(
            "SavingAccount_Salary_Payment_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Salary_Payment_Transactions_porc"),
        compute_percentage(
            "SavingAccount_Transfer_In_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Transfer_In_Transactions_porc"),
        compute_percentage(
            "SavingAccount_ATM_Extraction_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_ATM_Extraction_Transactions_porc"),
        compute_percentage(
            "SavingAccount_Service_Payment_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Service_Payment_Transactions_porc"),
        compute_percentage(
            "SavingAccount_CreditCard_Payment_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_CreditCard_Payment_Transactions_porc"),
        compute_percentage(
            "SavingAccount_Transfer_Out_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_Transfer_Out_Transactions_porc"),
        compute_percentage(
            "SavingAccount_DebitCard_Spend_Transactions",
            "SavingAccount_Transactions_Transactions",
        ).alias("SavingAccount_DebitCard_Spend_Transactions_porc"),
        compute_percentage(
            "SavingAccount_Salary_Payment_Transactions",
            "SavingAccount_Credits_Transactions",
        ).alias("SavingAccount_Salary_Payment_Transactions_CR_porc"),
        compute_percentage(
            "SavingAccount_Transfer_In_Transactions",
            "SavingAccount_Credits_Transactions",
        ).alias("SavingAccount_Transfer_In_Transactions_CR_porc"),
        compute_percentage(
            "SavingAccount_ATM_Extraction_Transactions",
            "SavingAccount_Debits_Transactions",
        ).alias("SavingAccount_ATM_Extraction_Transactions_DE_porc"),
        compute_percentage(
            "SavingAccount_Service_Payment_Transactions",
            "SavingAccount_Debits_Transactions",
        ).alias("SavingAccount_Service_Payment_Transactions_DE_porc"),
        compute_percentage(
            "SavingAccount_CreditCard_Payment_Transactions",
            "SavingAccount_Debits_Transactions",
        ).alias("SavingAccount_CreditCard_Payment_Transactions_DE_porc"),
        compute_percentage(
            "SavingAccount_Transfer_Out_Transactions",
            "SavingAccount_Debits_Transactions",
        ).alias("SavingAccount_Transfer_Out_Transactions_DE_porc"),
        compute_percentage(
            "SavingAccount_DebitCard_Spend_Transactions",
            "SavingAccount_Debits_Transactions",
        ).alias("SavingAccount_DebitCard_Spend_Transactions_DE_porc"),
        compute_percentage(
            "SavingAccount_Credits_Amounts", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Credits_Amounts_porc"),
        compute_percentage(
            "SavingAccount_Debits_Amounts", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Debits_Amounts_porc"),
        compute_percentage(
            "SavingAccount_Salary_Payment_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Salary_Payment_Amount_porc"),
        compute_percentage(
            "SavingAccount_Transfer_In_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Transfer_In_Amount_porc"),
        compute_percentage(
            "SavingAccount_ATM_Extraction_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_ATM_Extraction_Amount_porc"),
        compute_percentage(
            "SavingAccount_Service_Payment_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Service_Payment_Amount_porc"),
        compute_percentage(
            "SavingAccount_CreditCard_Payment_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_CreditCard_Payment_Amount_porc"),
        compute_percentage(
            "SavingAccount_Transfer_Out_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_Transfer_Out_Amount_porc"),
        compute_percentage(
            "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Total_Amount"
        ).alias("SavingAccount_DebitCard_Spend_Amount_porc"),
        compute_percentage(
            "SavingAccount_Salary_Payment_Amount", "SavingAccount_Credits_Amounts"
        ).alias("SavingAccount_Salary_Payment_Amount_CR_porc"),
        compute_percentage(
            "SavingAccount_Transfer_In_Amount", "SavingAccount_Credits_Amounts"
        ).alias("SavingAccount_Transfer_In_Amount_CR_porc"),
        compute_percentage(
            "SavingAccount_ATM_Extraction_Amount", "SavingAccount_Debits_Amounts"
        ).alias("SavingAccount_ATM_Extraction_Amount_DE_porc"),
        compute_percentage(
            "SavingAccount_Service_Payment_Amount", "SavingAccount_Debits_Amounts"
        ).alias("SavingAccount_Service_Payment_Amount_DE_porc"),
        compute_percentage(
            "SavingAccount_CreditCard_Payment_Amount", "SavingAccount_Debits_Amounts"
        ).alias("SavingAccount_CreditCard_Payment_Amount_DE_porc"),
        compute_percentage(
            "SavingAccount_Transfer_Out_Amount", "SavingAccount_Debits_Amounts"
        ).alias("SavingAccount_Transfer_Out_Amount_DE_porc"),
        compute_percentage(
            "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Debits_Amounts"
        ).alias("SavingAccount_DebitCard_Spend_Amount_DE_porc"),
    ]
)

training_data = training_data.with_columns(
    [
        # OPERATION
        compute_percentage("Operations_remote", "Operations_total").alias(
            "Operations_remote_porc"
        ),
        compute_percentage("Operations_in_person", "Operations_total").alias(
            "Operations_in_person_porc"
        ),
        compute_percentage("Operations_Bank", "Operations_total").alias(
            "Operations_Bank_porc"
        ),
        compute_percentage("Operations_Terminal", "Operations_total").alias(
            "Operations_Terminal_porc"
        ),
        compute_percentage("Operations_HomeBanking", "Operations_total").alias(
            "Operations_HomeBanking_porc"
        ),
        compute_percentage("Operations_Mobile", "Operations_total").alias(
            "Operations_Mobile_porc"
        ),
        compute_percentage("Operations_Ivr", "Operations_total").alias(
            "Operations_Ivr_porc"
        ),
        compute_percentage("Operations_Telemarketer", "Operations_total").alias(
            "Operations_Telemarketer_porc"
        ),
        compute_percentage("Operations_ATM", "Operations_total").alias(
            "Operations_ATM_porc"
        ),
        compute_percentage("Operations_Bank", "Operations_in_person").alias(
            "Operations_Bank_IP_porc"
        ),
        compute_percentage("Operations_Terminal", "Operations_in_person").alias(
            "Operations_Terminal_IP_porc"
        ),
        compute_percentage("Operations_HomeBanking", "Operations_remote").alias(
            "Operations_HomeBanking_R_porc"
        ),
        compute_percentage("Operations_Mobile", "Operations_remote").alias(
            "Operations_Mobile_R_porc"
        ),
        compute_percentage("Operations_Ivr", "Operations_remote").alias(
            "Operations_Ivr_R_porc"
        ),
        compute_percentage("Operations_Telemarketer", "Operations_remote").alias(
            "Operations_Telemarketer_R_porc"
        ),
        compute_percentage("Operations_ATM", "Operations_in_person").alias(
            "Operations_ATM_IP_porc"
        ),
        # CREDIT CARD
        compute_percentage(
            "CreditCard_Payment_remote", "CreditCard_Payment_total"
        ).alias("CreditCard_Payment_remote_porc"),
        compute_percentage(
            "CreditCard_Payment_in_person", "CreditCard_Payment_total"
        ).alias("CreditCard_Payment_in_person_porc"),
        compute_percentage(
            "CreditCard_Payment_Aut_Debit", "CreditCard_Payment_total"
        ).alias("CreditCard_Payment_Aut_Debit_porc"),
        compute_percentage(
            "CreditCard_Payment_External", "CreditCard_Payment_total"
        ).alias("CreditCard_Payment_External_porc"),
        compute_percentage("CreditCard_Payment_Cash", "CreditCard_Payment_total").alias(
            "CreditCard_Payment_Cash_porc"
        ),
        compute_percentage("CreditCard_Payment_Web", "CreditCard_Payment_total").alias(
            "CreditCard_Payment_Web_porc"
        ),
        compute_percentage("CreditCard_Payment_ATM", "CreditCard_Payment_total").alias(
            "CreditCard_Payment_ATM_porc"
        ),
        compute_percentage("CreditCard_Payment_TAS", "CreditCard_Payment_total").alias(
            "CreditCard_Payment_TAS_porc"
        ),
        compute_percentage(
            "CreditCard_Payment_Aut_Debit", "CreditCard_Payment_remote"
        ).alias("CreditCard_Payment_Aut_Debit_R_porc"),
        compute_percentage(
            "CreditCard_Payment_External", "CreditCard_Payment_in_person"
        ).alias("CreditCard_Payment_External_IP_porc"),
        compute_percentage(
            "CreditCard_Payment_Cash", "CreditCard_Payment_in_person"
        ).alias("CreditCard_Payment_Cash_IP_porc"),
        compute_percentage("CreditCard_Payment_Web", "CreditCard_Payment_remote").alias(
            "CreditCard_Payment_Web_R_porc"
        ),
        compute_percentage(
            "CreditCard_Payment_ATM", "CreditCard_Payment_in_person"
        ).alias("CreditCard_Payment_ATM_IP_porc"),
        compute_percentage(
            "CreditCard_Payment_TAS", "CreditCard_Payment_in_person"
        ).alias("CreditCard_Payment_TAS_IP_porc"),
        compute_percentage("CreditCard_Balance_ARG", "CreditCard_Total_Limit").alias(
            "CreditCard_Balance_ARG_limit_porc"
        ),
        compute_percentage("CreditCard_Balance_DOLLAR", "CreditCard_Total_Limit").alias(
            "CreditCard_Balance_DOLLAR_limit_porc"
        ),
        compute_percentage("CreditCard_Total_Spending", "CreditCard_Total_Limit").alias(
            "CreditCard_Total_Spending_limit_porc"
        ),
        compute_percentage(
            "CreditCard_Spending_1_Installment", "CreditCard_Total_Limit"
        ).alias("CreditCard_Spending_1_Installment_limit_porc"),
        compute_percentage(
            "CreditCard_Spending_Installments", "CreditCard_Total_Limit"
        ).alias("CreditCard_Spending_Installments_limit_porc"),
        compute_percentage(
            "CreditCard_Spending_CrossBoarder", "CreditCard_Total_Limit"
        ).alias("CreditCard_Spending_CrossBoarder_limit_porc"),
        compute_percentage(
            "CreditCard_Spending_Aut_Debits", "CreditCard_Total_Limit"
        ).alias("CreditCard_Spending_Aut_Debits_limit_porc"),
        compute_percentage("CreditCard_Revolving", "CreditCard_Total_Limit").alias(
            "CreditCard_Revolving_limit_porc"
        ),
        compute_percentage("CreditCard_Balance_ARG", "CreditCard_Total_Spending").alias(
            "CreditCard_Balance_ARG_SP_porc"
        ),
        compute_percentage(
            "CreditCard_Balance_DOLLAR", "CreditCard_Total_Spending"
        ).alias("CreditCard_Balance_DOLLAR_SP_porc"),
        compute_percentage(
            "CreditCard_Spending_1_Installment", "CreditCard_Total_Spending"
        ).alias("CreditCard_Spending_1_Installment_SP_porc"),
        compute_percentage(
            "CreditCard_Spending_Installments", "CreditCard_Total_Spending"
        ).alias("CreditCard_Spending_Installments_SP_porc"),
        compute_percentage(
            "CreditCard_Spending_CrossBoarder", "CreditCard_Total_Spending"
        ).alias("CreditCard_Spending_CrossBoarder_SP_porc"),
        compute_percentage(
            "CreditCard_Spending_Aut_Debits", "CreditCard_Total_Spending"
        ).alias("CreditCard_Spending_Aut_Debits_SP_porc"),
        compute_percentage("CreditCard_Revolving", "CreditCard_Total_Spending").alias(
            "CreditCard_Revolving_SP_porc"
        ),
        # OTHERS
        (
            pl.col("CreditCard_Premium")
            + pl.col("CreditCard_Active")
            + pl.col("Loan_Active")
            + pl.col("Mortgage_Active")
            + pl.col("SavingAccount_Active_ARG_Salary")
            + pl.col("SavingAccount_Active_ARG")
            + pl.col("SavingAccount_Active_DOLLAR")
            + pl.col("DebitCard_Active")
            + pl.col("Investment_Active")
            + pl.col("Insurance_Life")
            + pl.col("Insurance_Home")
            + pl.col("Insurance_Accidents")
            + pl.col("Insurance_Mobile")
            + pl.col("Insurance_ATM")
            + pl.col("Insurance_Unemployment")
        ).alias("Quantity_Active_Products"),
        (
            pl.col("CreditCard_Active")
            + pl.col("SavingAccount_Active_ARG")
            + pl.col("SavingAccount_Active_DOLLAR")
            + pl.col("DebitCard_Active")
        ).alias("Quantity_Common_Active_Product"),
        (
            pl.col("CreditCard_Active")
            + pl.col("SavingAccount_Active_ARG")
            + pl.col("DebitCard_Active")
        ).alias("Quantity_Most_Common_Active_Product"),
    ]
)
print(f"training_data despues de generar transformadas: {training_data.shape}")
```

```python
training_data.describe()
```

```python
print_without_trunc(scan_anomalies(training_data))
```

```python
training_data.schema
```

```python
training_data.select(pl.col(pl.String))
```

```python
training_data.select(
    [
        settings.col_id,
        "SavingAccount_Transactions_Transactions",
        "Operations_total",
        "CreditCard_Payment_total",
    ]
).filter(
    (pl.col("SavingAccount_Transactions_Transactions") != 0)
    & (pl.col("Operations_total") != 0)
    & (pl.col("CreditCard_Payment_total") != 0)
)
```

```python
greater_than_one_hundred_columns = [
    "SavingAccount_Transfer_In_Amount_porc",
    "SavingAccount_Transfer_In_Amount_CR_porc",
    "SavingAccount_Balance_last_minus_first_date_porc",
]

count_row_matches(
    training_data,
    columns=greater_than_one_hundred_columns,
    threshold=100,
    condition=">",
)
```

## Aggregate Features

Antes de la agregacion se ordenan los registros de cada cliente por mes para que luego funcionen "first" y "last" correctamente

diff_rel (Diferencia relativa): (último / primero)

pct_var (Variación porcentual): 1 - diferencia relativa

```python
columns_with_monetary_values = [
    "SavingAccount_Balance_FirstDate",
    "SavingAccount_Balance_LastDate",
    "SavingAccount_Balance_Average",
    "SavingAccount_Salary_Payment_Amount",
    "SavingAccount_Transfer_In_Amount",
    "SavingAccount_ATM_Extraction_Amount",
    "SavingAccount_Service_Payment_Amount",
    "SavingAccount_CreditCard_Payment_Amount",
    "SavingAccount_Transfer_Out_Amount",
    "SavingAccount_DebitCard_Spend_Amount",
    "SavingAccount_Total_Amount",
    "SavingAccount_Credits_Amounts",
    "SavingAccount_Debits_Amounts",
    "SavingAccount_Balance_last_minus_first_date",
    "CreditCard_Balance_ARG",
    "CreditCard_Balance_DOLLAR",
    "CreditCard_Total_Spending",
    "CreditCard_Spending_1_Installment",
    "CreditCard_Spending_Installments",
    "CreditCard_Spending_CrossBoarder",
    "CreditCard_Spending_Aut_Debits",
    "CreditCard_Revolving",
]

columns_with_quantities = [
    c
    for c in training_data.columns
    if c
    not in {
        *columns_with_monetary_values,
        *identity_features.columns,
        "Month",
        "First_product_dt",
        "Last_product_dt",
        settings.col_id,
        settings.col_target,
    }
]

cols = pl.col([*columns_with_quantities, *columns_with_monetary_values])

agg_exprs = [
    cols.min().name.suffix("_min"),
    cols.max().name.suffix("_max"),
    cols.mean().name.suffix("_mean"),
    cols.median().name.suffix("_median"),
    cols.sum().name.suffix("_sum"),
    (cols != 0).sum().name.suffix("_count_nonzero"),
    cols.var().name.suffix("_var"),
    cols.std().name.suffix("_std"),
    pl.col(columns_with_quantities).n_unique().name.suffix("_nunique"),
    (pl.col(columns_with_monetary_values) / 1000.0)
    .round()
    .n_unique()
    .name.suffix("_rounded_nunique"),
    (cols.max() - cols.min()).name.suffix("_ptp"),
    (cols.last() - cols.first()).name.suffix("_diff"),
    compute_percentage(cols.last(), cols.first()).name.suffix("_diff_rel"),
    (compute_percentage(cols.last(), cols.first()) - 100.0).name.suffix("_pct_var"),
]

data_agg = (
    training_data.sort([settings.col_id, "Month"])
    .group_by(settings.col_id)
    .agg(agg_exprs)
)

print(data_agg.shape)
```

```python
scan_anomalies(data_agg)
```

```python
# TODO: SACAR ESTA CELDA SI PODES:
# Esto restaura el orden original de las columnas del DataFrame
# para que la matriz de correlación funcione igual
# sin esto la matriz de correlaciones saca las variables que necesito
# para sacar esto tendria que agregar manualmente las variables que necesito,
# o usar otras variables (correlacionadas)

"""
original_order = [
    *(
        f"{col}_{f}"
        for col in [*columns_with_quantities, *columns_with_monetary_values]
        for f in ["var", "std"]
    ),
    *(f"{col}_nunique" for col in columns_with_quantities),
    *(f"{col}_rounded_nunique" for col in columns_with_monetary_values),
]

data_agg = data_agg.select(
    settings.col_id,
    pl.col("^.*(_min|_max|_mean|_median|_sum|_count_nonzero)$"),
    *original_order,
    pl.col("^.*(_ptp|_diff|_diff_rel|_pct_var)$"),
)

print(data_agg.shape)
data_agg.describe()
"""
```

# ABT

```python
ABT = identity_features.join(data_agg, on=settings.col_id, how="inner")
```

```python
print(ABT.shape)
scan_anomalies(ABT)
```

## Agrego transformadas extras luego de las operaciones de agregacion

```python
ABT = ABT.with_columns(
    (
        min_max_normalize("SavingAccount_Days_with_use_count_nonzero")
        + min_max_normalize("SavingAccount_Days_with_use_min")
        + min_max_normalize(
            "SavingAccount_CreditCard_Payment_Transactions_count_nonzero"
        )
        + min_max_normalize("Operations_total_count_nonzero")
        + min_max_normalize("CreditCard_Payment_total_max")
        + min_max_normalize("CreditCard_Payment_in_person_max")
        + (pl.col("Operations_in_person_porc_max") > 0).cast(pl.Float64)
        + (pl.col("CreditCard_Payment_Aut_Debit_max") > 0).cast(pl.Float64)
        + (pl.col("CreditCard_Payment_TAS_max") > 0).cast(pl.Float64)
    ).alias("SUM_OF_USES"),
    min_max_normalize_weighted(
        "SavingAccount_CreditCard_Payment_Amount_max", "Operations_total_count_nonzero"
    ).alias("Amount_operations"),
    min_max_normalize_weighted(
        "SavingAccount_CreditCard_Payment_Amount_max",
        "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
    ).alias("Amount_transactions"),
    min_max_normalize_weighted(
        "SavingAccount_CreditCard_Payment_Amount_max", "CreditCard_Payment_total_max"
    ).alias("Amount_payment"),
    min_max_normalize_weighted(
        "CreditCard_Total_Limit_diff_rel", "Operations_total_count_nonzero"
    ).alias("Limit_operations"),
    min_max_normalize_weighted(
        "CreditCard_Total_Limit_diff_rel",
        "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
    ).alias("Limit_transactions"),
    min_max_normalize_weighted(
        "CreditCard_Total_Limit_diff_rel", "CreditCard_Payment_total_max"
    ).alias("Limit_payment"),
)
print(ABT.shape)
```

```python
print(ABT.select(pl.col(pl.String)))
scan_anomalies(ABT)
```

```python
print_without_trunc(mins_in_range(ABT, -1, 1))
```

## Reduccion de dimensionalidad
### Elimino columnas con valores unicos

```python
constant_cols = get_constant_columns(ABT)
reduced_ABT = ABT.drop(constant_cols)

print(f"{constant_cols} \n")
print("ABT original: ", ABT.shape)
print("ABT sin columnas con valores unicos: ", reduced_ABT.shape)
```

### Elimino columnas binarias con baja representatividad

```python
imbalanced_binary_columns = get_imbalanced_binary_columns(reduced_ABT)

print_without_trunc(
    low_cardinality_value_counts(reduced_ABT.select(imbalanced_binary_columns))
)

reduced_ABT = reduced_ABT.drop(imbalanced_binary_columns)
print("ABT sin columnas binarias poco representativas:", reduced_ABT.shape)
```

### Elimino columnas correlacionadas entre si

```python
correlated_ABT = reduced_ABT.clone()
ABT_Correlation_analyzer = CorrelationAnalyzer(correlated_ABT)

to_delete = ABT_Correlation_analyzer.get_redundant_correlated_columns(threshold=0.80)
uncorrelated_ABT = correlated_ABT.drop(to_delete)

print(f"columnas con correlacion mayor a 80%: {len(to_delete)}")
print("ABT sin columnas con correlacion mayor a 80%:", uncorrelated_ABT.shape)
```

## Estandarizacion (z-score) con Polars

```python
standardized_ABT = standardize(uncorrelated_ABT)
print(scan_anomalies(standardized_ABT))
standardized_ABT.describe()
```

# Analizando variables

## Ordeno las variables por fuente segun importancia usando lightGBM para quedarme con las mas importantes

- primero entreno con todas las variables, para tener roc de referencia:
- luego entreno con cada grupo por separado
- luego entreno con los mejores de cada grupo al mismo tiempo

```python
all_cols = [
    col
    for col in standardized_ABT.columns
    if col not in {settings.col_id, settings.col_target}
]

columns_by_source = group_columns_by_source(all_cols)

print(
    f"cols_saving_account_days_transactions: "
    f"{len(columns_by_source['saving_account_days_transactions'])} \n"
    f"cols_saving_account_monetary: {len(columns_by_source['saving_account_monetary'])} \n"
    f"cols_operations: {len(columns_by_source['operations'])} \n"
    f"cols_credit_card_payment: {len(columns_by_source['credit_card_payment'])} \n"
    f"cols_credit_card_monetary: {len(columns_by_source['credit_card_monetary'])} \n"
    f"cols_others: {len(columns_by_source['others'])}"
)
columns_by_source["others"]
```

```python
X_train, X_test = stratified_train_test_split(standardized_ABT)
```

```python
all_cols_searcher, all_cols_importances = get_feature_importances(
    X_train,
    all_cols,
)
all_cols_searcher
```

```python
(
    cols_saving_account_days_transactions_searcher,
    cols_saving_account_days_transactions_importances,
) = get_feature_importances(
    X_train, columns_by_source["saving_account_days_transactions"]
)
cols_saving_account_days_transactions_searcher
```

```python
cols_saving_account_monetary_searcher, cols_saving_account_monetary_importances = (
    get_feature_importances(X_train, columns_by_source["saving_account_monetary"])
)
cols_saving_account_monetary_searcher
```

```python
cols_operations_searcher, cols_operations_importances = get_feature_importances(
    X_train, columns_by_source["operations"]
)
cols_operations_searcher
```

```python
cols_credit_card_payment_searcher, cols_credit_card_payment_importances = (
    get_feature_importances(X_train, columns_by_source["credit_card_payment"])
)

cols_credit_card_payment_searcher
```

```python
cols_credit_card_monetary_searcher, cols_credit_card_monetary_importances = (
    get_feature_importances(X_train, columns_by_source["credit_card_monetary"])
)

cols_credit_card_monetary_searcher
```

```python
cols_others_searcher, cols_others_importances = get_feature_importances(
    X_train, columns_by_source["others"]
)
cols_others_searcher
```

```python
plot_top_features(all_cols_importances, "all_cols_importances", all_cols_searcher)
plot_top_features(
    cols_saving_account_days_transactions_importances,
    "cols_saving_account_days_transactions_importances",
    cols_saving_account_days_transactions_searcher,
)
plot_top_features(
    cols_saving_account_monetary_importances,
    "cols_saving_account_monetary_importances",
    cols_saving_account_monetary_searcher,
    30,
)
plot_top_features(
    cols_operations_importances, "cols_operations_importances", cols_operations_searcher
)
plot_top_features(
    cols_credit_card_payment_importances,
    "cols_credit_card_payment_importances",
    cols_credit_card_payment_searcher,
)
plot_top_features(
    cols_credit_card_monetary_importances,
    "cols_credit_card_monetary_importances",
    cols_credit_card_monetary_searcher,
)
plot_top_features(
    cols_others_importances, "cols_others_importances", cols_others_searcher
)
```

![all_cols_importances](images/plot_top_features/all_cols_importances.svg)


![cols_saving_account_days_transactions_importances](images/plot_top_features/cols_saving_account_days_transactions_importances.svg)


![cols_saving_account_monetary_importances](images/plot_top_features/cols_saving_account_monetary_importances.svg)


![cols_operations_importances](images/plot_top_features/cols_operations_importances.svg)


![cols_credit_card_payment_importances](images/plot_top_features/cols_credit_card_payment_importances.svg)


![cols_credit_card_monetary_importances](images/plot_top_features/cols_credit_card_monetary_importances.svg)


![cols_others_importances](images/plot_top_features/cols_others_importances.svg)

```python
most_important_features = [
    *cols_saving_account_days_transactions_importances.head(1)
    .get_column("Feature")
    .to_list(),
    *cols_saving_account_monetary_importances.head(1).get_column("Feature").to_list(),
    *cols_operations_importances.head(1).get_column("Feature").to_list(),
    *cols_credit_card_payment_importances.head(1).get_column("Feature").to_list(),
    *cols_credit_card_monetary_importances.head(2).get_column("Feature").to_list(),
    *cols_others_importances.head(3).get_column("Feature").to_list(),
]

most_important_features_searcher, most_important_features_importances = (
    get_feature_importances(X_train, most_important_features)
)
plot_top_features(
    most_important_features_importances, "most_important_features", most_important_features_searcher
)
most_important_features_searcher
```

![most_important_features](images/plot_top_features/most_important_features.svg)


## Analisis Bivariado

```python
tables_analysis = generate_bivariate_charts(
    correlated_ABT, most_important_features, "analysis"
)
```

![SavingAccount_Days_with_use_count_nonzero](images/analysis/SavingAccount_Days_with_use_count_nonzero.svg)
![SavingAccount_Transfer_In_Transactions_count_nonzero](images/analysis/SavingAccount_Transfer_In_Transactions_count_nonzero.svg)
![SavingAccount_Transfer_In_Transactions_max](images/analysis/SavingAccount_Transfer_In_Transactions_max.svg)
![SavingAccount_Days_with_use_min](images/analysis/SavingAccount_Days_with_use_min.svg)
![SavingAccount_Days_with_Credits_porc_var](images/analysis/SavingAccount_Days_with_Credits_porc_var.svg)
![SavingAccount_CreditCard_Payment_Transactions_max](images/analysis/SavingAccount_CreditCard_Payment_Transactions_max.svg)
![SavingAccount_CreditCard_Payment_Transactions_count_nonzero](images/analysis/SavingAccount_CreditCard_Payment_Transactions_count_nonzero.svg)

![SavingAccount_Balance_FirstDate_max](images/analysis/SavingAccount_Balance_FirstDate_max.svg)
![SavingAccount_CreditCard_Payment_Amount_max](images/analysis/SavingAccount_CreditCard_Payment_Amount_max.svg)
![SavingAccount_Transfer_In_Amount_max](images/analysis/SavingAccount_Transfer_In_Amount_max.svg)
![SavingAccount_Total_Amount_min](images/analysis/SavingAccount_Total_Amount_min.svg)
![SavingAccount_Total_Amount_diff](images/analysis/SavingAccount_Total_Amount_diff.svg)
![SavingAccount_Balance_LastDate_diff_rel](images/analysis/SavingAccount_Balance_LastDate_diff_rel.svg)

![Operations_total_count_nonzero](images/analysis/Operations_total_count_nonzero.svg)
![Operations_total_min](images/analysis/Operations_total_min.svg)
![Operations_total_var](images/analysis/Operations_total_var.svg)
![Operations_Telemarketer_porc_max](images/analysis/Operations_Telemarketer_porc_max.svg)
![Operations_in_person_porc_min](images/analysis/Operations_in_person_porc_min.svg)
![Operations_in_person_porc_max](images/analysis/Operations_in_person_porc_max.svg)

![CreditCard_Payment_total_max](images/analysis/CreditCard_Payment_total_max.svg)
![CreditCard_Payment_Aut_Debit_max](images/analysis/CreditCard_Payment_Aut_Debit_max.svg)
![CreditCard_Payment_total_min](images/analysis/CreditCard_Payment_total_min.svg)
![CreditCard_Payment_TAS_max](images/analysis/CreditCard_Payment_TAS_max.svg)
![CreditCard_Payment_Cash_max](images/analysis/CreditCard_Payment_Cash_max.svg)
![CreditCard_Payment_Web_max](images/analysis/CreditCard_Payment_Web_max.svg)
![CreditCard_Payment_Aut_Debit_min](images/analysis/CreditCard_Payment_Aut_Debit_min.svg)
![CreditCard_Payment_Aut_Debit_diff](images/analysis/CreditCard_Payment_Aut_Debit_diff.svg)
![CreditCard_Payment_ATM_max](images/analysis/CreditCard_Payment_ATM_max.svg)
![CreditCard_Payment_in_person_porc_diff_rel](images/analysis/CreditCard_Payment_in_person_porc_diff_rel.svg)

![CreditCard_Payment_in_person_max](images/analysis/CreditCard_Payment_in_person_max.svg)

![CreditCard_Total_Limit_var](images/analysis/CreditCard_Total_Limit_var.svg)
![CreditCard_Total_Limit_diff_rel](images/analysis/CreditCard_Total_Limit_diff_rel.svg)
![CreditCard_Balance_ARG_SP_porc_max](images/analysis/CreditCard_Balance_ARG_SP_porc_max.svg)
![CreditCard_Total_Limit_min](images/analysis/CreditCard_Total_Limit_min.svg)
![CreditCard_Revolving_min](images/analysis/CreditCard_Revolving_min.svg)
![CreditCard_Total_Spending_diff_rel](images/analysis/CreditCard_Total_Spending_diff_rel.svg)
![CreditCard_Spending_Aut_Debits_diff_rel](images/analysis/CreditCard_Spending_Aut_Debits_diff_rel.svg)

![CreditCard_Product](images/analysis/CreditCard_Product.svg)
![Recency_in_days](images/analysis/Recency_in_days.svg)
![Days_between_first_and_last_product](images/analysis/Days_between_first_and_last_product.svg)
![Client_Age_grp](images/analysis/Client_Age_grp.svg)
![Quantity_Active_Products_min](images/analysis/Quantity_Active_Products_min.svg)
![Quantity_Active_Products_nunique](images/analysis/Quantity_Active_Products_nunique.svg)
![SavingAccount_Active_ARG_Salary](images/analysis/SavingAccount_Active_ARG_Salary.svg)
![Sex](images/analysis/Sex.svg)
![SavingAccount_Active_DOLLAR](images/analysis/SavingAccount_Active_DOLLAR.svg)
![Region](images/analysis/Region.svg)
![Quantity_Active_Products_var](images/analysis/Quantity_Active_Products_var.svg)
![Investment_Numbers_max](images/analysis/Investment_Numbers_max.svg)
![Email](images/analysis/Email.svg)

![Quantity_Common_Active_Product_count_nonzero](images/analysis/Quantity_Common_Active_Product_count_nonzero.svg)


```python
"""
graf = [
    "CreditCard_Payment_total_var",

    "Limit_operations",

    "SavingAccount_CreditCard_Payment_Amount_max",
    "CreditCard_Total_Limit_diff_rel",

    "SavingAccount_Days_with_use_count_nonzero",
    "SavingAccount_Days_with_use_min",
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",

    "Operations_total_count_nonzero",
    "Operations_in_person_porc_max",

    "CreditCard_Payment_total_max",
    "CreditCard_Payment_Aut_Debit_max",
    "CreditCard_Payment_in_person_max",

    "CreditCard_Product",
    "Days_between_first_and_last_product",
    "Client_Age_grp",
    "Quantity_Active_Products_min",
    "Recency_in_days",
]

generate_bivariate_charts(
    ABT, # dataset con variables sin standarizar
    graf,
    "analysis_2"
)
"""
```

![CreditCard_Payment_total_var](images/analysis_2/CreditCard_Payment_total_var.svg)

![Limit_operations](images/analysis_2/Limit_operations.svg)

![SavingAccount_CreditCard_Payment_Amount_max](images/analysis_2/SavingAccount_CreditCard_Payment_Amount_max.svg)
![CreditCard_Total_Limit_diff_rel](images/analysis_2/CreditCard_Total_Limit_diff_rel.svg)

![SavingAccount_Days_with_use_count_nonzero](images/analysis_2/SavingAccount_Days_with_use_count_nonzero.svg)
![SavingAccount_Days_with_use_min](images/analysis_2/SavingAccount_Days_with_use_min.svg)
![SavingAccount_CreditCard_Payment_Transactions_count_nonzero](images/analysis_2/SavingAccount_CreditCard_Payment_Transactions_count_nonzero.svg)

![Operations_total_count_nonzero](images/analysis_2/Operations_total_count_nonzero.svg)
![Operations_in_person_porc_max](images/analysis_2/Operations_in_person_porc_max.svg)

![CreditCard_Payment_total_max](images/analysis_2/CreditCard_Payment_total_max.svg)
![CreditCard_Payment_Aut_Debit_max](images/analysis_2/CreditCard_Payment_Aut_Debit_max.svg)
![CreditCard_Payment_in_person_max](images/analysis_2/CreditCard_Payment_in_person_max.svg)

![CreditCard_Product](images/analysis_2/CreditCard_Product.svg)
![Days_between_first_and_last_product](images/analysis_2/Days_between_first_and_last_product.svg)
![Client_Age_grp](images/analysis_2/Client_Age_grp.svg)
![Quantity_Active_Products_min](images/analysis_2/Quantity_Active_Products_min.svg)
![Recency_in_days](images/analysis_2/Recency_in_days.svg)


### Re-entreno con las mejores variables

```python
"""
to_test = [
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
    "Operations_in_person_porc_min",
    "Operations_in_person_porc_max",

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
    "CreditCard_Payment_in_person_porc_diff_rel",

    "CreditCard_Payment_in_person_max"
]

searcher_7, most_important_variables_7 = get_feature_importances(
    X_train, to_test
)
plot_top_features(most_important_variables_7, "most_important_variables_7", searcher_7)
searcher_7
"""
```

![most_important_variables_7](images/most_important_variables_7.svg)


### Re-entreno con las mejores variables

```python
"""
to_test_2 = [
    "SavingAccount_Days_with_use_count_nonzero",
    "SavingAccount_Days_with_use_min",
    "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",

    "SavingAccount_CreditCard_Payment_Amount_max",

    "Operations_total_count_nonzero",
    "Operations_in_person_porc_max",

    "CreditCard_Payment_total_max",
    "CreditCard_Payment_Aut_Debit_max",
    "CreditCard_Payment_in_person_max",

    "CreditCard_Total_Limit_diff_rel",

    "CreditCard_Product",
    "Days_between_first_and_last_product",
    "Client_Age_grp",
    "Quantity_Active_Products_min",
]

searcher_8, most_important_variables_8 = get_feature_importances(
    X_train, to_test_2
)
plot_top_features(most_important_variables_8, "most_important_variables_8", searcher_8)
searcher_8
"""
```

![most_important_variables_8](images/most_important_variables_8.svg)


### Buscando variables correlacionadas eliminadas anteriormente
Para poder intercambiar las variables mas importantes por variables mas faciles de interpretar, si es que existen.

```python
most_important_features_2 = (
    most_important_features_importances.head(5).get_column("Feature").to_list()
)

for x in most_important_features_2:
    print(f"\n Columnas correlacionadas con {x}:")
    print_without_trunc(ABT_Correlation_analyzer.get_correlations_for(x))
```

```python
most_important_features_correlated = [
    "Operations_total_mean",
    "Operations_total_median",
    "CreditCard_Active",
    "CreditCard_Balance_ARG_SP_porc_std",
    "CreditCard_Balance_ARG_SP_porc_mean",
    "Quantity_Active_Products_median",
    "Quantity_Active_Products_mean",
]

tables_analysis_2 = generate_bivariate_charts(
    correlated_ABT, most_important_features_correlated, "analysis_2"
)
```

![Operations_total_mean](images/analysis_2/Operations_total_mean.svg)
![Operations_total_median](images/analysis_2/Operations_total_median.svg)
![CreditCard_Active](images/analysis_2/CreditCard_Active.svg)
![CreditCard_Balance_ARG_SP_porc_std](images/analysis_2/CreditCard_Balance_ARG_SP_porc_std.svg)
![CreditCard_Balance_ARG_SP_porc_mean](images/analysis_2/CreditCard_Balance_ARG_SP_porc_mean.svg)
![Quantity_Active_Products_median](images/analysis_2/Quantity_Active_Products_median.svg)
![Quantity_Active_Products_mean](images/analysis_2/Quantity_Active_Products_mean.svg)


### Transformando mejores variables segun analisis bivariado y LightGBM

Transformo variables agregandoles el porcentaje de target y agrupo los valores.
A las variables categoricas solo las agrupo (ya las transforme anteriormente)

variables a modificar:
- Client_Age_grp
    - agrupo "Entre 50 y 59 años" + "Entre 60 y 64 años" + "Entre 65 y 69 años" (final: "Entre 50 y 69 años")
    - default -> junto totas las demas edades ("Entre 18 y 29 años" + "Entre 30 y 39 años" + "Entre 40 y 49 años" + "Mayor a 70 años")
- Operations_total_mean
- Operations_total_median
- CreditCard_Product
    - mantengo tipo tarjeta 202 y 104 separados
    - default ->  junto los demas tipos de tarjetas de bajo porcentaje de target y los tipos de tarjetas poco representativas en un solo bin (sin tarjeta de credito + 102 + 123 + 124 + 702 + 1002)
- CreditCard_Active (no hace falta transformar)
- Quantity_Active_Products_min

```python
final_ABT = correlated_ABT.clone()
```

```python
final_ABT = final_ABT.with_columns(
    group_bins_by_ranges(
        "Client_Age_grp",
        ranges=[(4, 5), (6, 7)],
        table=tables_analysis["Client_Age_grp"],
    ).alias("Client_Age_grp_t"),
    group_bins_by_ranges(
        "Operations_total_mean",
        ranges=[(2, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
        table=tables_analysis_2["Operations_total_mean"],
    ).alias("Operations_total_mean_t"),
    group_bins_by_ranges(
        "Operations_total_median",
        ranges=[(2, 4), (5, 7), (8, 9), (10, 11)],
        table=tables_analysis_2["Operations_total_median"],
    ).alias("Operations_total_median_t"),
    group_bins_by_ranges(
        "CreditCard_Product",
        ranges=[(5, 5), (7, 7)],
        table=tables_analysis["CreditCard_Product"],
    ).alias("CreditCard_Product_t"),
    group_bins_by_ranges(
        "Quantity_Active_Products_min",
        ranges=[(1, 4), (6, 9)],
        table=tables_analysis["Quantity_Active_Products_min"],
    ).alias("Quantity_Active_Products_min_t"),
)

scan_anomalies(final_ABT)

# # Intento agrupar demas variables (calculado en excel)

# final_ABT = final_ABT.with_columns(
#     group_bins_by_ranges(
#         "Operations_total_count_nonzero",
#         ranges=[(1, 1), (2, 3), (4, 5), (6, 6)],
#         table=tables_analysis_2["Operations_total_count_nonzero"],
#     ).alias("Operations_total_count_nonzero_t")
# )

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'Region',
#     ranges=[(24.370, 24.375)],  # mantengo "REGION CENTRO"
#     values=[24.372],
#     default=30.663,
# ).alias("Region_t"))
# # default -> totas las demas regiones
# # (NORTE GRANDE ARGENTINO + CUYO + CABA Centro/Norte + AMBA Resto + BUENOS AIRES
# # + REGION PATAGONICA)

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'Operations_in_person_max',
#     ranges=[(1, 2), (3, 44)],
#     values=[36.971, 54.786],
#     default=17.000,
# ).alias("Operations_in_person_max_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'Days_between_first_and_last_product',
#     ranges=[(0, 441), (442, 1142), (1143, 2130)],
#     values=[21.764, 25.244, 33.003],
#     default=48.858,
# ).alias("Days_between_first_and_last_product_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'Recency_in_days',
#     ranges=[(1, 408), (409, 650)],
#     values=[33.822, 29.220],
#     default=23.845,
# ).alias("Recency_in_days_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'CreditCard_Total_Spending_median',
#     ranges=[(0.5, 1979.9), (1980.2, 4078.7), (4079.0, 117452)],
#     values=[33.866, 41.667, 46.154],
#     default=9.000,
# ).alias("CreditCard_Total_Spending_median_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'SavingAccount_Balance_Average_median',
#     ranges=[(163.3, 2823.9), (2824.0, 1515662.7)],
#     values=[30.999, 50.143],
#     default=22.243,
# ).alias("SavingAccount_Balance_Average_median_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'SavingAccount_Transactions_Transactions_median',
#     ranges=[(0, 3), (3.5, 7)],
#     values=[21.781, 31.686],
#     default=54.346,
# ).alias("SavingAccount_Transactions_Transactions_median_t"))

# final_ABT = final_ABT.with_columns(group_bins_by_ranges(
#     'SavingAccount_CreditCard_Payment_Amount_median',
#     ranges=[(0, 0)],
#     values=[21.000],
#     default=55.253,
# ).alias("SavingAccount_CreditCard_Payment_Amount_median_t"))
```

```python
best_features = [
    "Client_Age_grp_t",
    "Operations_total_mean_t",
    #"Operations_total_median_t",
    "CreditCard_Product_t",
    #"CreditCard_Active",  # sin modificar
    "Quantity_Active_Products_min_t",
]

_ = generate_bivariate_charts(final_ABT, best_features, "analysis_t")
```

![CreditCard_Product_t](images/analysis_t/CreditCard_Product_t.svg)
![Client_Age_grp_t](images/analysis_t/Client_Age_grp_t.svg)
![Operations_total_mean_t](images/analysis_t/Operations_total_mean_t.svg)
![Quantity_Active_Products_min_t](images/analysis_t/Quantity_Active_Products_min_t.svg)


## Comparar importancias de las mejores variables

```python
X_train_int, X_test_int = stratified_train_test_split(final_ABT)
```

```python
best_features_searcher, best_features_importances = get_feature_importances(
    X_train_int, best_features
)
plot_top_features(best_features_importances, "best_features", best_features_searcher)
best_features_searcher
```

![best_features](images/plot_top_features/best_features.svg)


# Model Training


## Balancear a 50%/50% aprox (oversampling) <- a modo de ejemplo, esto despues no lo uso

```python
# TODO: en teoria no deberia hacer esto sobre la ABT, deberia hacerlo sobre X_train

max_id = clean_data.select(pl.col(settings.col_id).max()).item()

balanced_ABT = pl.concat(
    [
        final_ABT,
        final_ABT.filter(pl.col(settings.col_target) == 1).with_columns(
            (max_id + 1 + pl.int_range(0, pl.len())).alias(settings.col_id)
        ),
    ],
    how="vertical",
)

print(f"max_id: {max_id} \n")

print(f"{final_ABT.shape} \n")
print(f"{balanced_ABT.shape} \n")

print(f"{final_ABT[settings.col_target].value_counts()} \n")
print(f"{balanced_ABT[settings.col_target].value_counts()} \n")
```

## Entreno con las mejores variables y mejores hiperparametros

```python
X_train_final, X_test_final = stratified_train_test_split(final_ABT)
```

```python
best_hyperparameters_searcher, best_importances = get_feature_importances(
    X_train_final, best_features, n_iter=20
)

renames_dict = {
    "Client_Age_grp_t": "Age range",
    "Operations_total_mean_t": "Average quantity of operations",
    "CreditCard_Product_t": "Credit Card Type",
    "Quantity_Active_Products_min_t": "Minimum quantity of active products",
}
best_importances_renamed = best_importances.with_columns(
    pl.col(settings.col_feature)
    .replace_strict(renames_dict, default=pl.col(settings.col_feature))
    .alias(settings.col_feature)
)
plot_top_features(best_importances_renamed, "best_features_final", best_hyperparameters_searcher)

best_hyperparameters_searcher
```

![best_features_final](images/plot_top_features/best_features_final.svg)


# Performance del modelo

```python
y_pred, probabilities_train, probabilities_test = get_scoring(
    best_hyperparameters_searcher, X_train_final, X_test_final, best_features
)

# Cotas fijas....
# basado en los porcentajes de training
bins = [
    0.035669,
    0.054470,
    0.204116,
    0.241664,
    0.269271,
    0.393079,
    0.443684,
    0.541165,
    0.548265,
]

print_train_deciles(compute_prediction_deciles(X_train_final, probabilities_train))

print_test_deciles(
    compute_prediction_deciles(X_test_final, probabilities_test, bins),
    X_test_final,
    probabilities_test,
)

plot_roc_and_metrics(
    X_test_final[settings.col_target],
    probabilities_test,
    y_pred,
    graphic_name="lightgbm",
)
```

## ROC


![roc_lightgbm](images/plot_roc_and_metrics/lightgbm.svg)


## Resultados del excel

### Training
- ordena todos los deciles bien
- deciles masomenos parejos
- lift del primer decil = 2,4
- KS = 46,6 en el 4to decil

### Testing
- ~~ordena casi todos los deciles bien~~
- ~~deciles masomenos parejos~~
- lift del primer decil = 2
- KS = 46,4 en el 4to decil

### Diferencias
- lift -> 0,4
- KS -> 0,2


# Verificando si el problema se puede resolver con una regresion logistica

```python
"""
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
modelo.fit(X_train_final.select(best_features), X_train_final[settings.col_target])

y_pred_log = modelo.predict(X_test_final.select(best_features))
probabilities_train_log = modelo.predict_proba(X_train_final.select(best_features))
probabilities_test_log = modelo.predict_proba(X_test_final.select(best_features))

# Cotas fijas....
# basado en los porcentajes de training
bins = [
    0.051719,
    0.079377,
    0.170491,
    0.197105,
    0.267983,
    0.366586,
    0.467006,
    0.579763,
    0.580742,
]

print_train_deciles(compute_prediction_deciles(X_train_final, probabilities_train_log))

print_test_deciles(
    compute_prediction_deciles(X_test_final, probabilities_test_log, bins),
    X_test_final,
    probabilities_test_log,
)

plot_roc_and_metrics(
    X_test_final[settings.col_target],
    probabilities_test_log,
    y_pred_log,
    graphic_name="logistic_regression",
)
"""
```

![roc_logistic_regression](images/plot_roc_and_metrics/logistic_regression.svg)

```python

```

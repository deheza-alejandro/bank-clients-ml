import polars as pl

from bank_clients_ml.features import (
    compute_percentage,
    min_max_normalize,
    min_max_normalize_weighted,
    safe_denominator,
)


def add_transformations(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns(
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
                pl.col("CreditCard_Payment_Aut_Debit")
                + pl.col("CreditCard_Payment_Web")
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
                "SavingAccount_CreditCard_Payment_Amount",
                "SavingAccount_Debits_Amounts",
            ).alias("SavingAccount_CreditCard_Payment_Amount_DE_porc"),
            compute_percentage(
                "SavingAccount_Transfer_Out_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_Transfer_Out_Amount_DE_porc"),
            compute_percentage(
                "SavingAccount_DebitCard_Spend_Amount", "SavingAccount_Debits_Amounts"
            ).alias("SavingAccount_DebitCard_Spend_Amount_DE_porc"),
        ]
    )

    df = df.with_columns(
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
            compute_percentage(
                "CreditCard_Payment_Cash", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_Cash_porc"),
            compute_percentage(
                "CreditCard_Payment_Web", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_Web_porc"),
            compute_percentage(
                "CreditCard_Payment_ATM", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_ATM_porc"),
            compute_percentage(
                "CreditCard_Payment_TAS", "CreditCard_Payment_total"
            ).alias("CreditCard_Payment_TAS_porc"),
            compute_percentage(
                "CreditCard_Payment_Aut_Debit", "CreditCard_Payment_remote"
            ).alias("CreditCard_Payment_Aut_Debit_R_porc"),
            compute_percentage(
                "CreditCard_Payment_External", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_External_IP_porc"),
            compute_percentage(
                "CreditCard_Payment_Cash", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_Cash_IP_porc"),
            compute_percentage(
                "CreditCard_Payment_Web", "CreditCard_Payment_remote"
            ).alias("CreditCard_Payment_Web_R_porc"),
            compute_percentage(
                "CreditCard_Payment_ATM", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_ATM_IP_porc"),
            compute_percentage(
                "CreditCard_Payment_TAS", "CreditCard_Payment_in_person"
            ).alias("CreditCard_Payment_TAS_IP_porc"),
            compute_percentage(
                "CreditCard_Balance_ARG", "CreditCard_Total_Limit"
            ).alias("CreditCard_Balance_ARG_limit_porc"),
            compute_percentage(
                "CreditCard_Balance_DOLLAR", "CreditCard_Total_Limit"
            ).alias("CreditCard_Balance_DOLLAR_limit_porc"),
            compute_percentage(
                "CreditCard_Total_Spending", "CreditCard_Total_Limit"
            ).alias("CreditCard_Total_Spending_limit_porc"),
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
            compute_percentage(
                "CreditCard_Balance_ARG", "CreditCard_Total_Spending"
            ).alias("CreditCard_Balance_ARG_SP_porc"),
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
            compute_percentage(
                "CreditCard_Revolving", "CreditCard_Total_Spending"
            ).alias("CreditCard_Revolving_SP_porc"),
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

    return df


def add_extra_transformations(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
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
            "SavingAccount_CreditCard_Payment_Amount_max",
            "Operations_total_count_nonzero",
        ).alias("Amount_operations"),
        min_max_normalize_weighted(
            "SavingAccount_CreditCard_Payment_Amount_max",
            "SavingAccount_CreditCard_Payment_Transactions_count_nonzero",
        ).alias("Amount_transactions"),
        min_max_normalize_weighted(
            "SavingAccount_CreditCard_Payment_Amount_max",
            "CreditCard_Payment_total_max",
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

    return df

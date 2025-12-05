# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error



@st.cache_data
def load_data(csv_path: str = "laptop_price.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", index_col=0)

    df["Price_euros"] = pd.to_numeric(df["Price_euros"], errors="coerce")
    df = df.dropna(subset=["Price_euros"]).reset_index(drop=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    df_clean["Ram"] = (
        df_clean["Ram"]
        .astype(str)
        .str.replace("GB", "", regex=False)
        .astype("int64")
    )

    df_clean["Weight"] = (
        df_clean["Weight"]
        .astype(str)
        .str.replace("kg", "", regex=False)
        .astype("float64")
    )

    df_clean["Memory_GB"] = (
        df_clean["Memory"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype("int64")
    )
    df_clean = df_clean.drop(columns=["Memory"])

    #    معالجه الاوتلاير 
    numeric_cols = [
        c
        for c in df_clean.select_dtypes(include=["int64", "float64"]).columns
        if c != "Price_euros"
    ]

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])

    return df_clean


#  Random Forest  
@st.cache_resource
def train_random_forest(df_clean: pd.DataFrame):
    y = df_clean["Price_euros"]
    X = df_clean.drop("Price_euros", axis=1)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse =  mean_squared_error(y_test, preds) ** 0.5


    category_values = {col: sorted(X[col].unique().tolist()) for col in categorical_features}

    return pipe, X, y, r2, rmse, category_values


# Streamlit

def main():
    st.set_page_config(
        page_title="Laptop Price Analysis & Prediction",
        layout="wide",
    )

    st.title("💻 تحليل أسعار اللابتوبات")

    st.title("By (Mohamed Elhanafy, Mohamed Mostafa, Anas Ashraf) ")
    
    st.title(" Supervisor   Eng. leqaa hani ")

    
    # تدريب الموديل
    df = load_data()
    df_clean = clean_data(df)
    model, X, y, r2, rmse, category_values = train_random_forest(df_clean)

    tab_overview, tab_eda, tab_predict = st.tabs(
        [" Overview", " EDA / Analysis", " Model Prediction"]
    )

    # Overview

    with tab_overview:
        st.header("Overview")

        st.write(f"Number of rows    **{df.shape[0]}**")
        st.write(f"Number of columns **{df.shape[1]}**")

        st.subheader("First 10 rows")
        st.dataframe(df.head(10))

        st.subheader("Numeric features")
        st.dataframe(df_clean.describe().T)

        st.subheader("(Price_euros)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_clean["Price_euros"], bins=40, kde=True, ax=ax)
        ax.set_xlabel("Price in Euros")
        st.pyplot(fig)



    #EDA / Analysis
    with tab_eda:
        st.header(" Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Count To Company")
            company_counts = df_clean["Company"].value_counts()
            st.bar_chart(company_counts)

        with col2:
            st.subheader("Average Price To Company")
            company_price = (
                df_clean.groupby("Company")["Price_euros"].mean().sort_values()
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            company_price.plot(kind="barh", ax=ax)
            ax.set_xlabel("Average Price (EUR)")
            st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Count Of TypeName")
            type_counts = df_clean["TypeName"].value_counts()
            st.bar_chart(type_counts)

        with col4:
            st.subheader("Inches To Price")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_clean,
                x="Inches",
                y="Price_euros",
                hue="TypeName",
                alpha=0.7,
                ax=ax,
            )
            st.pyplot(fig)

        st.subheader("كورلاشن ماتركس")
        numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_clean[numeric_cols].corr(), annot=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    # Model Prediction
    with tab_predict:
        st.header("Model Prediction ((((Random Forest)))) ")

        st.markdown("---")
        st.subheader("أدخل مواصفات اللابتوب")

        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                company = st.selectbox(
                    "Company", options=category_values.get("Company", ["Apple", "HP"])
                )

                typename = st.selectbox(
                    "TypeName",
                    options=category_values.get("TypeName", ["Ultrabook", "Notebook"]),
                )

                product = st.text_input(
                    "Product (اسم الموديل)",
                    value="My Laptop Model",
                )

                opsys = st.selectbox(
                    "Operating System (OpSys)",
                    options=category_values.get("OpSys", ["Windows 10", "macOS"]),
                )

                screen_res = st.text_input(
                    "Screen Resolution (ScreenResolution)",
                    value="1920x1080",
                )

            with col_b:
                inches = st.number_input(
                    "Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6
                )
                ram = st.slider("RAM (GB)", min_value=2, max_value=64, value=8, step=2)
                memory_gb = st.slider(
                    "Memory (GB)", min_value=64, max_value=2048, value=256, step=64
                )
                weight = st.number_input(
                    "Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1
                )

                cpu = st.text_input("CPU", value="Intel Core i5")
                gpu = st.text_input("GPU", value="Nvidia GeForce")

            submitted = st.form_submit_button("احسب السعر ")

        if submitted:
            input_dict = {
                "Company": [company],
                "Product": [product],
                "TypeName": [typename],
                "Inches": [inches],
                "ScreenResolution": [screen_res],
                "Cpu": [cpu],
                "Ram": [ram],
                "Gpu": [gpu],
                "OpSys": [opsys],
                "Weight": [weight],
                "Memory_GB": [memory_gb],
            }

            input_df = pd.DataFrame(input_dict)

            #  نديله الداتا مباشرة
            pred_price = model.predict(input_df)[0]

            st.success(f"السعر المتوقع للابتوب  **((({pred_price:,.2f})))**")


if __name__ == "__main__":
    main()

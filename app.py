from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("WA_Fn-UseC_-HR-Employee-Attrition.csv")
IGNORED_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


@st.cache_resource
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")

    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=IGNORED_COLUMNS, errors="ignore")
    target = (data["Attrition"] == "Yes").astype(int)
    features = data.drop(columns="Attrition")

    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    return {
        "model": model,
        "features": features,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "attrition_recall": recall_score(y_test, predictions),
        "attrition_rate": target.mean(),
    }


def add_profile_inputs(features: pd.DataFrame) -> dict:
    profile = {}
    groups = [
        (
            "Personal",
            ["Age", "Gender", "MaritalStatus", "DistanceFromHome"],
        ),
        (
            "Role and satisfaction",
            [
                "Department",
                "JobRole",
                "BusinessTravel",
                "OverTime",
                "JobLevel",
                "JobInvolvement",
                "JobSatisfaction",
                "EnvironmentSatisfaction",
                "RelationshipSatisfaction",
                "WorkLifeBalance",
                "Education",
                "EducationField",
            ],
        ),
        (
            "Experience and pay",
            [
                "TotalWorkingYears",
                "YearsAtCompany",
                "YearsInCurrentRole",
                "YearsSinceLastPromotion",
                "YearsWithCurrManager",
                "NumCompaniesWorked",
                "TrainingTimesLastYear",
                "PercentSalaryHike",
                "PerformanceRating",
                "DailyRate",
                "HourlyRate",
                "MonthlyIncome",
                "MonthlyRate",
                "StockOptionLevel",
            ],
        ),
    ]

    for group_name, columns in groups:
        with st.expander(group_name, expanded=group_name == "Personal"):
            col1, col2 = st.columns(2)
            for index, column in enumerate(columns):
                series = features[column]
                container = col1 if index % 2 == 0 else col2
                with container:
                    if not pd.api.types.is_numeric_dtype(series):
                        options = sorted(series.dropna().unique().tolist())
                        mode = series.mode().iloc[0]
                        profile[column] = st.selectbox(
                            column,
                            options,
                            index=options.index(mode),
                        )
                    else:
                        minimum = int(series.min())
                        maximum = int(series.max())
                        default = int(series.median())
                        profile[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=1,
                        )
    return profile


st.set_page_config(page_title="Employee Attrition Risk", page_icon="📊", layout="centered")
st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #142634;
        color: #f2eee5;
    }

    [data-testid="stHeader"] {
        background-color: #142634;
    }

    h1, h2, h3 {
        color: #f2eee5;
        font-family: Georgia, "Times New Roman", serif;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        color: #d2dce0;
    }

    [data-testid="stExpander"] {
        background-color: #203747;
        border: 1px solid #385160;
        border-left: 3px solid #e65f3e;
        border-radius: 2px;
    }

    [data-testid="stMetric"] {
        background-color: #203747;
        border: 1px solid #385160;
        border-radius: 2px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #d2dce0;
    }

    [data-testid="stMetricValue"] {
        color: #f07858;
        font-family: Georgia, "Times New Roman", serif;
    }

    input,
    [data-baseweb="select"] > div {
        background-color: #203747;
        color: #f2eee5;
    }

    [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #f2eee5;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #e65f3e;
        color: #ffffff;
        border: 1px solid #e65f3e;
        border-radius: 2px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #c94c30;
        color: #ffffff;
        border-color: #c94c30;
    }

    input:focus,
    [data-baseweb="select"]:focus-within {
        border-color: #e65f3e;
        box-shadow: 0 0 0 1px #e65f3e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption("PEOPLE ANALYTICS  /  MODEL 01")
st.title("Employee Attrition Risk Predictor")
st.write(
    "Enter an employee profile to estimate attrition risk. "
    "The model is trained from the HR dataset when the app starts."
)

try:
    model_bundle = load_model()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()

with st.form("employee_profile"):
    profile = add_profile_inputs(model_bundle["features"])
    submitted = st.form_submit_button("Estimate attrition risk")

if submitted:
    employee = pd.DataFrame([profile], columns=model_bundle["features"].columns)
    risk = float(model_bundle["model"].predict_proba(employee)[0, 1])
    risk_level = "Low" if risk <= 0.3 else "Medium" if risk <= 0.6 else "High"

    st.subheader("Model estimate")
    st.metric("Estimated attrition risk", f"{risk:.1%}")
    st.write(f"**Risk band:** {risk_level}")
    st.caption(
        "This is a model estimate, not a certainty or a recommendation for employment decisions."
    )

st.divider()
st.subheader("Model check")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Validation ROC-AUC", f"{model_bundle['roc_auc']:.2f}")
metric2.metric("Attrition recall", f"{model_bundle['attrition_recall']:.2f}")
metric3.metric("Dataset attrition rate", f"{model_bundle['attrition_rate']:.1%}")
st.caption(
    "Validation metrics use a stratified 80/20 split and the same class-balanced "
    "Logistic Regression approach as the notebook."
)

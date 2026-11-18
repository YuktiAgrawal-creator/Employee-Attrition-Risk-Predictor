# Employee Attrition Risk Predictor

A simple Streamlit interface for the model in `employee_attrition_predictor.ipynb`.
It trains the notebook's class-balanced Logistic Regression pipeline from the
included HR attrition CSV and estimates risk for a supplied employee profile.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `WA_Fn-UseC_-HR-Employee-Attrition.csv` beside `app.py`.

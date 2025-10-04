import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR Attrition Dashboard", page_icon="📊", layout="wide")

# --- DARK THEME STYLING ---
st.markdown(
    """
    <style>
    :root{
      --bg: #070707;
      --surface: #0d0d0d;
      --muted: #a8a8a8;
      --text: #eaeaea;
      --accent: #FFC700;
      --accent-600: #ffb800;
      --border: rgba(255,255,255,0.04);
      --card-shadow: 0 8px 30px rgba(0,0,0,0.6);
    }
    .stApp, .css-18e3th9 { background-color: var(--bg); color:var(--text); }
    .stApp .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      box-shadow: var(--card-shadow);
      color:var(--text);
    }
    h1 { color: var(--text); font-family:Inter, sans-serif; }
    h2 { color: var(--text); font-family:Inter, sans-serif; border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
    .stButton>button {
      background: linear-gradient(90deg,var(--accent), var(--accent-600));
      color:#0a0a0a;
      border-radius:8px;
      padding:10px 18px;
      border: none;
      font-weight:700;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.6); }
    input, textarea, select { background: #0b0b0b !important; color: var(--text) !important; border:1px solid rgba(255,255,255,0.04) !important; }
    div[data-testid="stMetric"] { background: transparent; color:var(--text); }
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
    .css-1d391kg { color:var(--text); }
    a { color: var(--accent) !important; }
    @media (max-width: 768px) {
      .stApp .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data(path: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
        df = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
        if 'Attrition' in df.columns and df['Attrition'].dtype == object:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found. Please place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' next to this app.")
        return None

df = load_data()

# --- MODEL TRAINING ---
@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, Dict], pd.Index]:
    """
    Trains a RandomForest model and returns the trained model,
    encoders for categorical features, and the feature columns.
    """
    if data is None:
        return None, {}, pd.Index([])

    df_encoded = data.copy()
    encoders: Dict[str, Dict] = {}

    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        if col == 'Attrition':
            continue
        cats = list(df_encoded[col].astype('category').cat.categories)
        mapping = {cat: i for i, cat in enumerate(cats)}
        encoders[col] = mapping
        df_encoded[col] = df_encoded[col].map(mapping).astype('Int64')

    if 'Attrition' not in df_encoded.columns:
        st.error("Dataset missing 'Attrition' column.")
        return None, {}, pd.Index([])

    X = df_encoded.drop(columns=['Attrition']).fillna(-1)
    y = df_encoded['Attrition'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    try:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        model._cv_accuracy = float(acc)
    except Exception:
        model._cv_accuracy = None

    feature_cols = X.columns
    return model, encoders, feature_cols

model, encoders, feature_columns = train_model(df)

# --- UI LAYOUT ---
st.title("📊 HR Analytics — Attrition Dashboard")
st.markdown(
    """
    <div class="card">
      <strong style="font-size:1.05rem">Interactive dashboard</strong>
      <div style="color:var(--muted); margin-top:6px;">
        Explore employee attrition patterns and try a simple predictor. Trained model: RandomForest (cached).
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔮 Employee Attrition Predictor", expanded=True):
    if df is None:
        st.warning("Data not available for predictor.")
    else:
        st.markdown("Fill in a sample employee profile and click **Predict Attrition**.")
        input_cols = [c for c in df.columns if c != 'Attrition']
        cols = st.columns(3)
        input_values = {}
        for i, col in enumerate(input_cols):
            target_col = cols[i % 3]
            series = df[col]
            label = f"{col}"
            if series.dtype == object or series.dtype.name == 'category':
                opts_sorted = sorted(list(series.dropna().unique()), key=lambda x: str(x))
                input_values[col] = target_col.selectbox(label, opts_sorted, index=0, key=f"inp_{col}")
            elif pd.api.types.is_integer_dtype(series) and series.nunique() < 20:
                input_values[col] = target_col.slider(label, int(series.min()), int(series.max()), int(series.median()), key=f"inp_{col}")
            else:
                input_values[col] = target_col.number_input(label, value=float(series.median()), key=f"inp_{col}")

        if st.button("Predict Attrition"):
            if model is None:
                st.error("Model not available.")
            else:
                input_df = pd.DataFrame([input_values])
                input_transformed = input_df.copy()

                for col in input_transformed.columns:
                    if col in encoders:
                        mapping = encoders[col]
                        val = input_transformed.at[0, col]
                        code = mapping.get(val, -1)
                        input_transformed.at[0, col] = code
                    else:
                        try:
                            input_transformed[col] = pd.to_numeric(input_transformed[col])
                        except Exception:
                            input_transformed[col] = -1

                for expected in feature_columns:
                    if expected not in input_transformed.columns:
                        input_transformed[expected] = -1
                input_transformed = input_transformed[feature_columns].fillna(-1).astype(float)

                try:
                    proba = model.predict_proba(input_transformed)[0]
                    classes = list(model.classes_)
                    pos_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
                    confidence = proba[pos_idx]
                    prediction = model.predict(input_transformed)[0]
                except Exception as e:
                    st.exception(e)
                    st.error("Prediction failed.")
                    prediction, confidence = None, None

                st.markdown("---")
                st.subheader("Prediction Result")
                left, right = st.columns([2,1])
                with left:
                    if prediction is not None:
                        if int(prediction) == 1:
                            st.markdown("<h3 style='color:#ffb703;'>High risk of attrition</h3>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 style='color:#6ee7b7;'>Low risk of attrition</h3>", unsafe_allow_html=True)
                        if getattr(model, "_cv_accuracy", None) is not None:
                            st.caption(f"Model CV accuracy (test split): {model._cv_accuracy:.2f}")
                with right:
                    if confidence is not None:
                        st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

st.header("📈 Data Insights")
if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Overtime")
        if 'OverTime' in df.columns:
            fig1 = px.pie(
                df, names='OverTime', color='Attrition',
                title='Attrition Distribution by Overtime', hole=0.33,
                color_discrete_map={1: "#e76f51", 0: "#2a9d8f", 'Yes': "#e76f51", 'No': "#2a9d8f"}
            )
            fig1.update_traces(textinfo='percent+label')
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Attrition Rate by Job Role")
        if 'JobRole' in df.columns:
            temp = df.copy()
            if temp['Attrition'].dtype != int and temp['Attrition'].dtype != float:
                temp['Attrition'] = temp['Attrition'].map({'Yes': 1, 'No': 0}).fillna(0)
            role_pct = temp.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False) * 100
            role_df = role_pct.reset_index().rename(columns={'Attrition': 'AttritionRate'})
            fig2 = px.bar(role_df, x='JobRole', y='AttritionRate', title='Attrition Rate (%) by Job Role',
                          labels={'AttritionRate': 'Attrition Rate (%)'}, color='AttritionRate',
                          color_continuous_scale=px.colors.sequential.OrRd)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Income Distribution vs. Attrition")
    if 'MonthlyIncome' in df.columns:
        fig3 = px.histogram(df, x='MonthlyIncome', color='Attrition', barmode='overlay', marginal='rug',
                            color_discrete_map={1: "#e76f51", 0: "#2a9d8f", 'Yes': "#e76f51", 'No': "#2a9d8f"})
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Model: RandomForestClassifier · UI: dark theme (yellow accent)")
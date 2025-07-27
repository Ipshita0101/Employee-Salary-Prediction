import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
#import plotly.express as px
#import plotly.graph_objects as go

# Load model and encoder
@st.cache_resource
def load_models():
   
    with gzip.open("model/rf_model_compressed.pkl.gz", "rb") as f:
        model = pickle.load(f)

    with open("model/ordinal_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_models()

# App setup
st.set_page_config("Employee Salary Prediction", "💼", layout="wide")
st.title("💼 Employee Salary Predictor")
st.caption("Predict if a person's income is above or below $50K based on demographic and work features.")

# Sidebar
with st.sidebar:
    st.header("📊 Prediction Settings")
    mode = st.radio("Choose mode:", ["🔍 Predict", "📈 Feature Analysis"])
    st.success("✅ Model Loaded Sucessfully!")

    st.markdown("### 🎯 Quick Tips")
    st.markdown("""
- Fill all inputs for accurate results  
- Education and hours/week affect salary  
- Capital gains/losses are strong indicators  
- Marital status and occupation matter  
""")

# Early stop
if model is None:
    st.error("Model not loaded.")
    st.stop()

# --- Predict Mode ---
if mode == "🔍 Predict":
    st.subheader("👤 User Details")
    col1, col2, col3 = st.columns(3)

    age = col1.slider("Age", 18, 90, 30)
    gender = col1.selectbox("Gender", ['Male', 'Female'])
    race = col2.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Black', 'Other'])
    marital = col2.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced'])
    relationship = col3.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Unmarried'])
    native_country = col3.selectbox("Native Country", ['United-States', 'India', 'Canada'])

    st.subheader("💼 Work & Education")
    col4, col5 = st.columns(2)
    workclass = col4.selectbox("Work Class", ['Private', 'Self-emp-not-inc', 'Federal-gov'])
    occupation = col4.selectbox("Occupation", ['Tech-support', 'Sales', 'Exec-managerial'])
    hours = col4.slider("Hours/Week", 1, 100, 40)

    education = col5.selectbox("Education", ['Bachelors', 'Masters', 'HS-grad'])
    edu_num = col5.slider("Education Years", 1, 16, 10)

    st.subheader("💰 Financials")
    fnlwgt = st.number_input("Final Weight", 10000, 1000000, 50000)
    gain = st.number_input("Capital Gain", 0)
    loss = st.number_input("Capital Loss", 0)

    if st.button("🔮 Predict Salary"):
        input_df = pd.DataFrame([{
            "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
            "education": education, "educational-num": edu_num, "marital-status": marital,
            "occupation": occupation, "relationship": relationship, "race": race,
            "gender": gender, "capital-gain": gain, "capital-loss": loss,
            "hours-per-week": hours, "native-country": native_country
        }])

        # Encode categorical
        cat_cols = input_df.select_dtypes(include="object").columns
        input_df[cat_cols] = encoder.transform(input_df[cat_cols])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = proba[1] * 100 if pred == ">50K" else proba[0] * 100

        st.subheader("🎯 Prediction Result")
        if pred == ">50K":
            st.success("🎉 Income likely > 50K")
        else:
            st.info("💼 Income likely ≤ 50K")
        st.metric("Confidence", f"{confidence:.1f}%")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence,
            title={"text": "Prediction Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00cc44" if pred == ">50K" else "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': "#444"},
                    {'range': [50, 80], 'color': "#ffaa00"},
                    {'range': [80, 100], 'color': "#00cc44"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 90}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Simulated Feature Importance
        imp = pd.DataFrame({
            "Factor": ['Education', 'Age', 'Hours', 'Occupation', 'Marital', 'Capital Gains'],
            "Importance": np.random.uniform(5, 25, 6)
        }).sort_values("Importance")
        fig_imp = px.bar(imp, x='Importance', y='Factor', orientation='h',
                         color='Importance', color_continuous_scale='viridis')
        st.plotly_chart(fig_imp, use_container_width=True)

# --- Feature Analysis ---
if mode == "📈 Feature Analysis":
    st.subheader("📈 Feature Impact Analysis")
    col1, col2 = st.columns(2)

    # Simulate Age Distribution
    ages = np.random.normal(40, 12, 1000)
    ages = ages[(ages >= 18) & (ages <= 90)]
    fig1 = px.histogram(x=ages, nbins=30, title="Age Distribution", labels={'x': 'Age'})
    col1.plotly_chart(fig1, use_container_width=True)

    # Simulate Education Impact
    edu_impact = {'Doctorate': 85, 'Masters': 72, 'Bachelors': 58, 'HS-grad': 23}
    fig2 = px.bar(x=list(edu_impact.keys()), y=list(edu_impact.values()),
                  labels={'x': 'Education', 'y': 'Prob >50K'},
                  title="Education vs High Income")
    col2.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.caption("💼 Employee Salary Prediction")

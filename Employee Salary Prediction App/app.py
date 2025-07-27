import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt

# Mock model and encoder for demonstration
class MockModel:
    def predict(self, X):
        return [">50K" if np.random.random() > 0.5 else "<=50K"]
    
    def predict_proba(self, X):
        prob = np.random.uniform(0.3, 0.9)
        return [[1-prob, prob]]

class MockEncoder:
    def transform(self, X):
        return np.random.randint(0, 10, size=X.shape)

# Use mock models
model = MockModel()
encoder = MockEncoder()

# App setup
st.set_page_config("Employee Salary Prediction", "💼", layout="wide")

if st.sidebar.button("🔄 Clear Cache & Reload"):
    st.cache_resource.clear()
    st.rerun()

st.title("💼 Employee Salary Predictor")
st.caption("Predict if a person's income is above or below $50K based on demographic and work features.")

# Sidebar
with st.sidebar:
    st.header("📊 Prediction Settings")
    mode = st.radio("Choose mode:", ["🔍 Predict", "📈 Feature Analysis"])
    st.success("✅ Model Loaded Successfully!")

    st.markdown("### 🎯 Quick Tips")
    st.markdown("""
- Fill all inputs for accurate results  
- Education and hours/week affect salary  
- Capital gains/losses are strong indicators  
- Marital status and occupation matter  
""")

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

        # Progress bar for confidence visualization
        st.write("**Prediction Confidence:**")
        st.progress(confidence / 100)
        
        # Color-coded confidence level
        if confidence >= 80:
            st.success(f"High Confidence: {confidence:.1f}%")
        elif confidence >= 60:
            st.warning(f"Medium Confidence: {confidence:.1f}%")
        else:
            st.error(f"Low Confidence: {confidence:.1f}%")

        # Simulated Feature Importance using bar chart
        st.subheader("📊 Feature Importance")
        imp = pd.DataFrame({
            "Factor": ['Education', 'Age', 'Hours', 'Occupation', 'Marital', 'Capital Gains'],
            "Importance": np.random.uniform(5, 25, 6)
        }).sort_values("Importance", ascending=True)
        
        st.bar_chart(imp.set_index('Factor')['Importance'])

# --- Feature Analysis ---
if mode == "📈 Feature Analysis":
    st.subheader("📈 Feature Impact Analysis")
    
    # Generate sample data for analysis
    ages = np.random.normal(40, 12, 1000)
    ages = ages[(ages >= 18) & (ages <= 90)]
    age_hist = pd.DataFrame({'Age': ages})
    
    col1, col2 = st.columns(2)

    with col1:
        # Age Distribution
        st.subheader("Age Distribution")
        if age_hist is not None and not age_hist.empty and 'Age' in age_hist.columns:
            fig_age, ax_age = plt.subplots(figsize=(10, 6))
            ax_age.hist(age_hist['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax_age.set_xlabel('Age')
            ax_age.set_ylabel('Frequency')
            ax_age.set_title('Age Distribution')
            ax_age.grid(True, alpha=0.3)
            st.pyplot(fig_age)
        else:
            st.warning("Age distribution data not available or 'Age' column not found")

    with col2:
        st.subheader("Education vs High Income Probability")
        # Simulate Education Impact
        edu_impact = pd.DataFrame({
            'Education': ['Doctorate', 'Masters', 'Bachelors', 'HS-grad'],
            'Prob_High_Income': [85, 72, 58, 23]
        })
        st.bar_chart(edu_impact.set_index('Education')['Prob_High_Income'])

    # Additional analysis sections
    st.subheader("📋 Summary Statistics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Average Age", "40.2 years", "2.1")
    with col4:
        st.metric("High Income Rate", "24.8%", "1.2%")
    with col5:
        st.metric("Avg Hours/Week", "40.4 hrs", "-0.8")

    # Correlation Analysis
    st.subheader("🔗 Feature Correlations")
    correlation_data = pd.DataFrame({
        'Feature': ['Age', 'Education Years', 'Hours/Week', 'Capital Gain', 'Capital Loss'],
        'Correlation with Income': [0.23, 0.34, 0.18, 0.22, -0.05]
    })
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    bars = ax_corr.barh(correlation_data['Feature'], correlation_data['Correlation with Income'])
    
    # Color bars based on correlation strength
    for i, bar in enumerate(bars):
        if correlation_data['Correlation with Income'].iloc[i] >= 0.2:
            bar.set_color('green')
        elif correlation_data['Correlation with Income'].iloc[i] >= 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax_corr.set_xlabel('Correlation Coefficient')
    ax_corr.set_title('Feature Correlation with High Income')
    ax_corr.grid(True, alpha=0.3)
    st.pyplot(fig_corr)

    # Income Distribution by Category
    st.subheader("💰 Income Distribution by Category")
    col6, col7 = st.columns(2)
    
    with col6:
        st.write("**By Education Level**")
        edu_income = pd.DataFrame({
            'Education': ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'],
            'High_Income_Percentage': [15, 22, 42, 65, 78]
        })
        st.bar_chart(edu_income.set_index('Education')['High_Income_Percentage'])
    
    with col7:
        st.write("**By Work Class**")
        work_income = pd.DataFrame({
            'Work_Class': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov'],
            'High_Income_Percentage': [22, 18, 45, 32, 28]
        })
        st.bar_chart(work_income.set_index('Work_Class')['High_Income_Percentage'])

# Footer
st.markdown("---")
st.caption("💼 Employee Salary Prediction")

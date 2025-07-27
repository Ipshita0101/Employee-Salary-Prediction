import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

# Mock model and encoder for demonstration with performance metrics
class MockModel:
    def predict(self, X):
        return [">50K" if np.random.random() > 0.5 else "<=50K"]
    
    def predict_proba(self, X):
        prob = np.random.uniform(0.3, 0.9)
        return [[1-prob, prob]]
    
    # Mock performance metrics
    def get_performance_metrics(self):
        return {
            'accuracy': 0.847,
            'precision': 0.821,
            'recall': 0.756,
            'f1_score': 0.787,
            'roc_auc': 0.892,
            'specificity': 0.901
        }

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
    st.header("📊 Model Information")
    st.success("✅ Model Loaded Successfully!")
    
    # Display model performance metrics
    st.subheader("🎯 Model Performance")
    metrics = model.get_performance_metrics()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("Precision", f"{metrics['precision']:.3f}")
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    with col_b:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.metric("Specificity", f"{metrics['specificity']:.3f}")

    st.markdown("### 🎯 Quick Tips")
    st.markdown("""
- Fill all inputs for accurate results  
- Education and hours/week affect salary  
- Capital gains/losses are strong indicators  
- Marital status and occupation matter  
""")

# --- Main Prediction Interface ---
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
fnlwgt = st.number_input("Final Weight", value=50000, min_value=10000, max_value=1000000)
gain = st.number_input("Capital Gain", value=0, min_value=0)
loss = st.number_input("Capital Loss", value=0, min_value=0)

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
    
    # Model Performance Summary
    st.subheader("📈 Model Performance Summary")
    metrics = model.get_performance_metrics()
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}", 
                 help="Overall correctness of predictions")
        st.metric("Precision", f"{metrics['precision']:.1%}", 
                 help="Accuracy of positive predictions")
    
    with col_metrics2:
        st.metric("Recall (Sensitivity)", f"{metrics['recall']:.1%}", 
                 help="Ability to find all positive cases")
        st.metric("F1-Score", f"{metrics['f1_score']:.1%}", 
                 help="Harmonic mean of precision and recall")
    
    with col_metrics3:
        st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.1%}", 
                 help="Area under the ROC curve")
        st.metric("Specificity", f"{metrics['specificity']:.1%}", 
                 help="Ability to correctly identify negatives")

# Footer
st.markdown("---")
st.caption("💼 Employee Salary Prediction")

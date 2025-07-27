# Employee Salary Prediction App

A machine learning application built with Streamlit that predicts employee salaries based on various features such as experience, education, location, and job role using a Random Forest algorithm.

## 🚀 Features

- **Random Forest Model**: Uses a trained Random Forest algorithm for accurate salary predictions
- **Streamlit Interface**: Clean, interactive web interface for easy use
- **Categorical Encoding**: Handles categorical variables with ordinal encoding
- **Real-time Predictions**: Instant salary predictions based on user input
- **Sample Data Support**: Test the app with provided sample data
- **Model Persistence**: Pre-trained model saved for quick predictions

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Interface**: Streamlit
- **Visualization**: matplotlib
- **Model Serialization**: pickle

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## 🔧 Installation

1. **Navigate to the project directory**
   ```bash
   cd employee-salary-prediction-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

### Making Predictions

1. Open the Streamlit app in your browser
2. Use the sidebar or main interface to enter employee details:
   - Years of experience
   - Education level
   - Job title/role
   - Location
   - Department
   - Other relevant features
3. The predicted salary will be displayed automatically
4. You can also upload the sample CSV file for batch predictions

### Testing with adult.csv Data

You can test the application using the provided input file:

1. **Prepare your sample data**:Use a CSV file with the same columns and format as used during model. training.
2. **Launch the app**: Open terminal and run streamlit run app.py to start the application.
3. **Upload the CSV**: Select your `adult.csv` file.
4. **View results**: The app will process the file and show predictions for all rows.

**Used CSV format:**
```csv
age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States
38,Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States
```

## 📊 Dataset

The model is trained on a comprehensive dataset containing:

- **Experience**: Years of professional experience
- **Education**: Degree level (Bachelor's, Master's, PhD)
- **Location**: Geographic location/city
- **Job Title**: Specific role or position
- **Company Size**: Small, Medium, Large enterprise
- **Industry**: Technology sector, Finance, Healthcare, etc.
- **Skills**: Technical and soft skills


## 🤖 Machine Learning Model

### Random Forest Algorithm

This application uses a **Random Forest** model for salary prediction:

- **Algorithm**: Random Forest Regressor from scikit-learn
- **Preprocessing**: Ordinal encoding for categorical variables
- **Model File**: `rf_model.pkl` (pre-trained model)
- **Encoder File**: `ordinal_encoder.pkl` (fitted encoder)

### Model Features

The model processes various employee attributes including:
- Years of experience
- Education level
- Job title/role
- Geographic location
- Department/industry

## 📁 Project Structure

```
EmployeeSalaryPrediction/
├── Dataset/
│   └── adult.csv                         # Raw dataset used for model training
├── Model/
│   ├── ordinal_encoder.pkl               # Encoder for categorical variables
│   └── rf_model.pkl                      # Trained Random Forest model
├── Training/
│   └── Employee Salary Prediction.ipynb  # Jupyter Notebook for training
├── app.py                                # Streamlit web application
├── License                               # License file (MIT recommended)
├── README.md                             # Project overview and instructions
└── requirements.txt                      # List of required Python packages             
```

## 📋 Requirements

The `requirements.txt` file should include:

```
streamlit
scikit-learn
pandas
numpy
matplotlib
pickle-mixin
```

*Note: Update this list based on your actual dependencies*

## 📈 Model Details

- **Algorithm**: Random Forest from scikit-learn
- **Preprocessing**: Ordinal encoding for categorical features
- **Model Persistence**: Saved using Python's pickle module
- **Input Validation**: Streamlit handles user input validation
- **Real-time Predictions**: Instant results through Streamlit interface

## 🔧 Model Files

- **`rf_model.pkl`**: Contains the trained Random Forest model.
- **`ordinal_encoder.pkl`**: Contains the fitted ordinal encoder for categorical variables.
- **`adult.csv`**: Dataset for testing the application.

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help improve this project:

Report bugs: If you find any issues, please let us know
Suggest features: Ideas for new functionality or improvements
Code improvements: Optimize existing code or add new features
Documentation: Help improve this README or add code comments
Testing: Test the app with different datasets and report results

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application provides salary predictions for informational purposes only. Actual salaries may vary based on numerous factors not captured in the model. Use predictions as guidance rather than definitive salary expectations.

## 🙏 Acknowledgments

- Dataset provided by [Data Source Name]
- Inspired by [Similar Projects or Research Papers]
- Thanks to the open-source community for the amazing ML libraries
---

**Star ⭐ this repository if you found it helpful!**
**Star ⭐ this repository if you found it helpful!**

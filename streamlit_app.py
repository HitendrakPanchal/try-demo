import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes, load_iris, load_boston

# Load sample datasets
def load_sample_dataset(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Diabetes":
        data = load_diabetes()
    elif dataset_name == "Boston Housing":
        data = load_boston()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

# Main app
def main():
    st.title("AutoML Web App")
    
    # Dataset selection
    st.header("1. Dataset Selection")
    dataset_option = st.radio("Choose dataset:", 
                            ("Sample Dataset", "Upload your own"))
    
    if dataset_option == "Sample Dataset":
        dataset_name = st.selectbox("Select sample dataset:", 
                                  ("Iris", "Diabetes", "Boston Housing"))
        df, target = load_sample_dataset(dataset_name)
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target = None
    
    if 'df' in locals():
        st.subheader("Dataset Preview:")
        st.write(df.head())
        
        # Column selection
        st.header("2. Column Selection")
        columns = df.columns.tolist()
        
        features = st.multiselect("Select feature columns:", columns)
        target_col = st.selectbox("Select target column:", columns)
        
        if features and target_col:
            X = df[features]
            y = df[target_col] if dataset_option == "Upload your own" else target
            
            # Train-test split
            st.header("3. Train-Test Split")
            test_size = st.selectbox("Select test size ratio:", 
                                    [0.2, 0.3, 0.1], index=0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            
            # Model selection
            st.header("4. Model Selection")
            model_name = st.selectbox("Choose a model:", 
                                    ["Linear Regression", 
                                     "Decision Tree", 
                                     "Random Forest"])
            
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_name == "Random Forest":
                model = RandomForestRegressor()
            
            # Training and evaluation
            if st.button("Train Model"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.header("5. Model Evaluation")
                st.subheader("Predictions:")
                st.write(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
                
                st.subheader("Evaluation Metrics:")
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics_df = pd.DataFrame({
                    "Metric": ["Mean Squared Error", 
                             "Mean Absolute Error", 
                             "R² Score"],
                    "Value": [mse, mae, r2]
                })
                st.write(metrics_df)

if __name__ == "__main__":
    main()
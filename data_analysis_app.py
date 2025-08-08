import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

# App configuration
st.set_page_config(
    page_title="Data Analysis Suite",
    page_icon="📊",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Main app
def main():
    st.title("📊 Comprehensive Data Analysis Suite")
    st.sidebar.header("Data Upload & Configuration")
    
    # Upload data
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV or Excel)", 
        type=["csv", "xlsx", "xls"]
    )
    
    if not uploaded_file:
        st.info("👈 Please upload a data file to begin")
        return
    
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # Basic info
    st.sidebar.divider()
    st.sidebar.subheader("Dataset Information")
    st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data preview
    st.header("Data Preview")
    st.write(df.head())
    
    # Data cleaning section
    st.header("🔍 Data Cleaning")
    
    # Missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found!")
    else:
        st.warning(f"Total missing values: {missing.sum()}")
        fig_missing = px.bar(
            missing[missing > 0], 
            title="Missing Values per Column",
            labels={'index':'Column', 'value':'Missing Count'}
        )
        st.plotly_chart(fig_missing)
        
        # Missing value treatment
        st.subheader("Treatment Options")
        treatment = st.selectbox(
            "Select treatment method for missing values:",
            ["Drop rows", "Drop columns", "Impute with mean", "Impute with median", "Impute with mode"]
        )
        
        if st.button("Apply Treatment"):
            if treatment == "Drop rows":
                df = df.dropna()
            elif treatment == "Drop columns":
                cols_to_drop = missing[missing > 0].index.tolist()
                df = df.drop(columns=cols_to_drop)
            else:
                imputer = SimpleImputer(
                    strategy=treatment.split()[-1].lower()
                )
                df = pd.DataFrame(
                    imputer.fit_transform(df),
                    columns=df.columns
                )
            st.success("Missing values treated successfully!")
            st.write("New shape:", df.shape)
    
    # Duplicate handling
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate rows")
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success(f"Removed {duplicates} duplicates")
            st.write("New shape:", df.shape)
    
    # EDA Section
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    # Column selection
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    # Distribution plots
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        num_feature = st.selectbox("Select numerical feature:", num_cols)
        fig = px.histogram(df, x=num_feature, marginal="box")
        st.plotly_chart(fig)
        
    with col2:
        cat_feature = st.selectbox("Select categorical feature:", cat_cols)
        fig = px.histogram(df, x=cat_feature)
        st.plotly_chart(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    if len(num_cols) > 1:
        corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
        corr_matrix = df[num_cols].corr(method=corr_method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r'
        ))
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig)
        
        # Show top correlations
        st.write("Top Correlations:")
        corr_unstacked = corr_matrix.unstack()
        corr_unstacked = corr_unstacked.sort_values(ascending=False)
        high_corr = corr_unstacked[corr_unstacked < 1].drop_duplicates().head(10)
        st.dataframe(high_corr.reset_index().rename(columns={0: "Correlation"}))
    else:
        st.warning("Not enough numerical columns for correlation analysis")
    
    # Outlier detection
    st.subheader("Outlier Detection")
    if num_cols.any():
        outlier_col = st.selectbox("Select column for outlier detection:", num_cols)
        fig = px.box(df, y=outlier_col)
        st.plotly_chart(fig)
    
    # Feature engineering
    st.header("⚙️ Feature Engineering")
    if len(num_cols) > 1:
        if st.button("Perform PCA"):
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[num_cols])
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            
            fig = px.scatter(
                df, 
                x='PCA1', 
                y='PCA2', 
                title="PCA Visualization"
            )
            st.plotly_chart(fig)
            st.success("PCA components added to dataset")
    
    # Model training section
    st.header("🤖 Machine Learning Modeling")
    
    # Problem type selection
    problem_type = st.radio(
        "Select problem type:", 
        ["Classification", "Regression"]
    )
    
    # Target selection
    target = st.selectbox("Select target variable:", df.columns)
    
    # Preprocessing
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encode target for classification
    if problem_type == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Feature selection
    features = st.multiselect("Select features:", X.columns, default=list(X.columns))
    X = X[features]
    
    # Split data
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Preprocessing pipeline
    # Handle numerical features
    num_transformer = StandardScaler()
    X_train_num = num_transformer.fit_transform(X_train.select_dtypes(include=np.number))
    X_test_num = num_transformer.transform(X_test.select_dtypes(include=np.number))
    
    # Handle categorical features
    cat_cols = X_train.select_dtypes(exclude=np.number).columns
    if not cat_cols.empty:
        X_train_cat = pd.get_dummies(X_train[cat_cols])
        X_test_cat = pd.get_dummies(X_test[cat_cols])
        
        # Align columns
        X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
        
        # Combine features
        X_train_processed = np.hstack([X_train_num, X_train_cat])
        X_test_processed = np.hstack([X_test_num, X_test_cat])
    else:
        X_train_processed = X_train_num
        X_test_processed = X_test_num
    
    # Model selection
    if problem_type == "Classification":
        model_choice = st.selectbox(
            "Select classifier:",
            ["Random Forest", "Logistic Regression"]
        )
    else:
        model_choice = st.selectbox(
            "Select regressor:",
            ["Random Forest", "Linear Regression"]
        )
    
    # Model training
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if problem_type == "Classification":
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100)
                else:
                    model = LogisticRegression(max_iter=1000)
            else:  # Regression
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100)
                else:
                    model = LinearRegression()
            
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Evaluation
            st.subheader("Model Evaluation")
            
            if problem_type == "Classification":
                st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm, 
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Class 0', 'Class 1'],
                    y=['Class 0', 'Class 1']
                )
                st.plotly_chart(fig)
                
                # ROC curve
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC curve (area = {roc_auc:.2f})'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Random'
                    ))
                    fig.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate'
                    )
                    st.plotly_chart(fig)
            else:  # Regression
                st.success(f"R² Score: {r2_score(y_test, y_pred):.2f}")
                st.info(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
                st.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                
                # Residual plot
                residuals = y_test - y_pred
                fig = px.scatter(
                    x=y_pred, 
                    y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title="Residual Plot"
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig)
                
                # Actual vs Predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, 
                    y=y_pred,
                    mode='markers',
                    name='Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=[min(y_test), max(y_test)], 
                    y=[min(y_test), max(y_test)],
                    mode='lines',
                    name='Ideal Fit',
                    line=dict(dash='dash')
                ))
                fig.update_layout(
                    title="Actual vs Predicted",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
                st.plotly_chart(fig)
    
    # Download processed data
    st.sidebar.divider()
    st.sidebar.subheader("Data Export")
    if st.sidebar.button("Export Processed Data"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
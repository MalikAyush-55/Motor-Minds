import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


# ------------------------ #
# **Helper Functions and Classes**
# ------------------------ #

# Helper Function: Split by Component
def split_by_component(df, feature_cols, test_size=0.3):
    components = df["ComponentID"].unique()
    np.random.shuffle(components)
    split_point = int(len(components) * (1 - test_size))
    train_components = components[:split_point]
    test_components = components[split_point:]

    train_df = df[df["ComponentID"].isin(train_components)]
    test_df = df[df["ComponentID"].isin(test_components)]

    X_train = train_df[feature_cols].values
    y_train = train_df["Failure"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["Failure"].values.astype(int)

    return X_train, X_test, y_train, y_test, train_components, test_components


# Helper Function: One-Hot Encode Regions
def one_hot_encode(regions, unique_regions):
    """One-hot encode the regions based on the unique regions list."""
    one_hot = np.zeros((len(regions), len(unique_regions)))
    for i, region in enumerate(regions):
        if region in unique_regions:
            index = np.where(unique_regions == region)[0][0]
            one_hot[i, index] = 1
    return one_hot


# Custom ARIMA Implementation
class CustomARIMA:
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.ar_coeffs = None
        self.mean = None

    def difference(self, data, d=1):
        """Apply d-th order differencing"""
        diff_data = data.copy()
        for _ in range(d):
            diff_data = np.diff(diff_data)
        return diff_data

    def inverse_difference(self, diff_data, original_data, d=1):
        """Inverse of differencing"""
        recovered = diff_data.copy()
        for _ in range(d):
            recovered = np.cumsum(recovered) + original_data[0]
        return recovered

    def fit(self, data):
        self.original_data = data
        diff_data = self.difference(data, self.d)

        X = np.zeros((len(diff_data) - self.p, self.p))
        for i in range(self.p):
            X[:, i] = diff_data[i:len(diff_data) - self.p + i]
        y = diff_data[self.p:]

        # Add a column of ones for intercept
        X = np.column_stack([X, np.ones(X.shape[0])])

        # Least squares estimation
        try:
            theta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            print("Linear algebra error during ARIMA fit:", e)
            theta = np.zeros(self.p + 1)

        self.ar_coeffs = theta[:-1]
        self.mean = theta[-1]

        return self

    def predict(self, steps):
        last_values = self.difference(self.original_data, self.d)[-self.p:]
        predictions = []
        for _ in range(steps):
            pred = self.mean
            for coeff, val in zip(self.ar_coeffs, last_values[::-1]):
                pred += coeff * val
            predictions.append(pred)
            last_values = np.append(last_values[1:], pred)
        return self.inverse_difference(np.array(predictions), self.original_data, self.d)


# Custom Linear Regression Implementation
class CustomLinearRegression:
    def __init__(self, alpha=1e-3):  # Adding regularization parameter
        self.coefficients = None
        self.intercept = None
        self.alpha = alpha  # Regularization strength

    def fit(self, X, y):
        X_b = np.column_stack([X, np.ones(X.shape[0])])  # Add intercept
        I = np.eye(X_b.shape[1])
        I[-1, -1] = 0

        try:
            theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        except np.linalg.LinAlgError as e:
            print("Matrix inversion error. Possible singular matrix:", e)
            theta = np.zeros(X_b.shape[1])

        self.coefficients = theta[:-1]
        self.intercept = theta[-1]
        return self

    def predict(self, X):
        return X @ self.coefficients + self.intercept


# Custom Logistic Regression Implementation
class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.class_weight = class_weight

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        # Calculate class weights if not provided
        if self.class_weight is None:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            self.class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]}

        sample_weights = np.array([self.class_weight[yi] for yi in y])

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)

            # Weighted gradient
            gradient = np.dot(X.T, (predictions - y) * sample_weights) / y.size
            self.weights -= self.lr * gradient

            if (epoch + 1) % 100 == 0:
                weighted_loss = -np.mean(sample_weights * (y * np.log(predictions + 1e-15) +
                                                           (1 - y) * np.log(1 - predictions + 1e-15)))
                print(f"Epoch {epoch + 1}, Weighted Loss: {weighted_loss:.4f}")

    def predict_prob(self, X):
        z = np.dot(X, self.weights)
        return self.sigmoid(z)


# Custom Decision Tree Implementation
class CustomDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = (group[:, -1] == class_val).sum() / size
                score += p ** 2
            gini += (1 - score) * (size / n_instances)
        return gini

    def split(self, index, value, dataset):
        left = dataset[dataset[:, index] < value]
        right = dataset[dataset[:, index] >= value]
        return left, right

    def weighted_gini_index(self, groups, classes, class_weights):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            weighted_size = sum([class_weights[val] * np.sum(group[:, -1] == val) for val in classes])
            for class_val in classes:
                weighted_p = (class_weights[class_val] * np.sum(group[:, -1] == class_val)) / weighted_size
                score += weighted_p ** 2
            gini += (1 - score) * (size / n_instances)
        return gini

    def get_split(self, dataset):
        class_values = np.unique(dataset[:, -1])
        b_index, b_value, b_score, b_groups = None, None, float("inf"), None

        # Calculate class weights
        class_weights = {val: 1.0 / (np.sum(dataset[:, -1] == val) + 1e-6) for val in class_values}

        for index in range(dataset.shape[1] - 1):
            for row in dataset:
                groups = self.split(index, row[index], dataset)
                # Apply weighted gini index
                gini = self.weighted_gini_index(groups, class_values, class_weights)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {"index": b_index, "value": b_value, "groups": b_groups}

    def to_terminal(self, group):
        if len(group) < self.min_samples_leaf:
            return 0  # Default to no failure for very small groups
        classes, counts = np.unique(group[:, -1], return_counts=True)
        weights = 1.0 / (counts + 1e-6)
        weighted_counts = counts * weights
        return classes[np.argmax(weighted_counts)]

    def split_node(self, node, depth):
        left, right = node["groups"]
        del node["groups"]

        if left.shape[0] == 0 or right.shape[0] == 0:
            node["left"] = node["right"] = self.to_terminal(np.vstack((left, right)))
            return

        if depth >= self.max_depth:
            node["left"], node["right"] = self.to_terminal(left), self.to_terminal(right)
            return

        if left.shape[0] < self.min_samples_split:
            node["left"] = self.to_terminal(left)
        else:
            node["left"] = self.get_split(left)
            self.split_node(node["left"], depth + 1)

        if right.shape[0] < self.min_samples_split:
            node["right"] = self.to_terminal(right)
        else:
            node["right"] = self.get_split(right)
            self.split_node(node["right"], depth + 1)

    def build_tree(self, train):
        root = self.get_split(train)
        self.split_node(root, 1)
        return root

    def fit(self, X, y):
        dataset = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self.build_tree(dataset)

    def predict_row(self, row, node):
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return self.predict_row(row, node["left"])
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.predict_row(row, node["right"])
            else:
                return node["right"]

    def predict(self, X):
        return np.array([self.predict_row(row, self.tree) for row in X])


# Car Sales Predictor Class
class CarSalesPredictor:
    def __init__(self):
        self.arima_models = {}
        self.lr_models = {}
        self.feature_scaler = None
        self.target_scaler = None

    def preprocess_data(self, data):
        """Preprocess the input data"""
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

        car_dummies = pd.get_dummies(data['car_model'], prefix='car')
        data = pd.concat([data, car_dummies], axis=1)

        return data

    def fit(self, data):
        processed_data = self.preprocess_data(data)

        for car in data['car_model'].unique():
            car_data = processed_data[processed_data['car_model'] == car]['sales'].values

            # Check if there are enough data points
            if len(car_data) <= 2:
                print(f"Not enough data to train ARIMA for car model: {car}")
                continue

            arima = CustomARIMA(p=2, d=1, q=0)
            arima.fit(car_data)
            self.arima_models[car] = arima

        features = ['price', 'marketing_spend', 'festival_season',
                    'competitor_launches', 'month']

        for car in data['car_model'].unique():
            car_data = processed_data[processed_data['car_model'] == car]
            X = car_data[features].values
            y = car_data['sales'].values

            # Avoid errors with small datasets
            if X.shape[0] <= X.shape[1]:
                print(f"Skipping model training for {car}: Not enough samples.")
                continue

            lr = CustomLinearRegression(alpha=1e-3)
            lr.fit(X, y)
            self.lr_models[car] = lr

        return self

    def predict(self, data, months_ahead=3):
        predictions = {}
        processed_data = self.preprocess_data(data)

        features = ['price', 'marketing_spend', 'festival_season',
                    'competitor_launches', 'month']

        for car in data['car_model'].unique():
            if car not in self.arima_models or car not in self.lr_models:
                print(f"Skipping prediction for {car}: Model not trained.")
                continue

            car_data = processed_data[processed_data['car_model'] == car]['sales'].values
            arima_pred = self.arima_models[car].predict(months_ahead)

            latest_features = processed_data[processed_data['car_model'] == car][features].iloc[-1].values
            future_months = []
            last_month = latest_features[-1]
            for i in range(months_ahead):
                future_months.append(((last_month + i) % 12) + 1)

            lr_predictions = []
            for future_month in future_months:
                features_copy = latest_features.copy()
                features_copy[-1] = future_month
                lr_pred = self.lr_models[car].predict(features_copy.reshape(1, -1))
                lr_predictions.append(lr_pred[0])

            final_predictions = (arima_pred + np.array(lr_predictions)) / 2
            predictions[car] = final_predictions

        return predictions


# ------------------------ #
# **Model Training and Caching**
# ------------------------ #

@st.cache_resource
def train_predictive_maintenance_model(df, selected_features, max_depth, min_samples_split, min_samples_leaf,
                                       learning_rate, epochs):
    # Prepare data
    X = df[selected_features].values
    y = df["Failure"].values.astype(int)

    # Train-test split
    if "ComponentID" in df.columns:
        X_train, X_test, y_train, y_test, train_components, test_components = split_by_component(df, selected_features)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Decision Tree
    tree = CustomDecisionTree(max_depth=max_depth,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf)
    tree.fit(X_train_scaled, y_train)

    # Predict regions
    train_regions = tree.predict(X_train_scaled)
    test_regions = tree.predict(X_test_scaled)

    # Get unique regions from training data
    unique_regions = np.unique(train_regions)

    # One-hot encode regions
    X_train_regions = one_hot_encode(train_regions, unique_regions)
    X_test_regions = one_hot_encode(test_regions, unique_regions)

    # Combine features
    X_train_hybrid = np.hstack((X_train_scaled, X_train_regions))
    X_test_hybrid = np.hstack((X_test_scaled, X_test_regions))

    # Train Logistic Regression
    class_weights = {0: 1.0, 1: np.sum(y_train == 0) / np.sum(y_train == 1)}
    lr = CustomLogisticRegression(lr=learning_rate, epochs=epochs, class_weight=class_weights)
    lr.fit(X_train_hybrid, y_train)

    # Make predictions
    y_pred_prob = lr.predict_prob(X_test_hybrid)

    # Find optimal threshold
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
        "Optimal Threshold": optimal_threshold
    }

    return {
        "scaler": scaler,
        "tree": tree,
        "lr": lr,
        "unique_regions": unique_regions,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred
    }


# ------------------------ #
# **Main Application Functions**
# ------------------------ #

# Sample Data Creation Function
def create_sample_dataset():
    # Creating dates for past 2 years of monthly data
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=30 * i) for i in range(24)]

    # Popular car models in India
    cars = ['Swift', 'Baleno', 'WagonR', 'Alto', 'Creta']

    data = []
    np.random.seed(42)

    for car in cars:
        # Base price (in lakhs)
        if car in ['Swift', 'Baleno', 'WagonR']:
            base_price = np.random.uniform(6, 8)
        elif car == 'Alto':
            base_price = np.random.uniform(4, 5)
        else:  # Creta
            base_price = np.random.uniform(12, 15)

        for date in dates:
            # Add seasonality
            season_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)

            # Add trend
            trend = date.month / 12 * 0.1

            # Base sales with randomness
            base_sales = np.random.normal(
                loc=1000 if car in ['Swift', 'Baleno'] else 800,
                scale=100
            )

            # Final sales calculation
            sales = int(base_sales * season_factor * (1 + trend))

            # Features
            marketing_spend = np.random.uniform(50, 100)  # in lakhs
            festival_season = 1 if date.month in [10, 11, 12] else 0
            competitor_launches = np.random.randint(0, 3)

            data.append({
                'date': date,
                'car_model': car,
                'price': base_price,
                'marketing_spend': marketing_spend,
                'festival_season': festival_season,
                'competitor_launches': competitor_launches,
                'sales': sales
            })

    return pd.DataFrame(data)


# Function to Show Home Page
def show_home_page():
    st.write("""
    ## Welcome to MotorMinds

    **MotorMinds** is an Automotive Smart Insights Platform that leverages advanced data analytics and machine learning to provide accurate forecasts and predictions for the automotive industry. Our platform helps automotive companies optimize inventory, improve customer satisfaction, and increase operational efficiency.

    ### Key Features

    - **Vehicle Sales Forecasting**: Predict future vehicle sales using time series analysis and machine learning models.
    - **Predictive Maintenance**: Anticipate maintenance needs and potential component failures before they occur.
    - **Spare Parts Demand Prediction**: Accurately forecast spare parts demand to optimize inventory levels.

    ### Benefits

    - **Optimize Inventory Management**: Reduce holding costs and prevent stockouts by maintaining optimal inventory levels.
    - **Enhance Vehicle Reliability**: Minimize unexpected breakdowns and extend vehicle lifespan through predictive maintenance.
    - **Adapt to Market Changes**: Stay ahead of market shifts by adapting quickly based on accurate sales forecasts.

    ### How to Use This App

    - Navigate through the app using the sidebar menu.
    - Upload your own data or adjust parameters in each section to see customized insights.
    - Visualize results through interactive graphs and charts.

    **Get started by selecting an analysis from the sidebar!**
    """)


# Predictive Maintenance Analysis
def run_predictive_maintenance():
    st.header("Predictive Maintenance Analysis")

    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    max_depth = st.sidebar.slider("Decision Tree Max Depth", 2, 10, 6)
    min_samples_split = st.sidebar.slider("Min Samples Split", 10, 100, 50)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 5, 50, 20)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
    epochs = st.sidebar.slider("Number of Epochs", 100, 2000, 1000)

    # File uploader
    uploaded_file = st.file_uploader("Upload your predictive maintenance data CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    else:
        st.info("Using default dataset. Upload your own CSV file to analyze custom data.")
        try:
            df = pd.read_csv("Datasets/predictive_maintenance_data.csv")
        except FileNotFoundError:
            st.error("Default dataset not found. Please upload a CSV file.")
            return
        except pd.errors.ParserError:
            st.error("Error parsing the default dataset. Please check the file format.")
            return

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Feature Selection")
    default_features = ["Temperature", "Pressure", "Vibration", "UsageHours", "DaysSinceMaintenance", "ComponentHealth"]
    available_features = df.columns.tolist()
    selected_features = st.multiselect(
        "Select features for analysis",
        available_features,
        default=default_features
    )

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    if "Failure" not in df.columns:
        st.error("Dataset must contain a 'Failure' column as the target variable.")
        return

    # Train the model using cached function
    with st.spinner("Training models..."):
        model_data = train_predictive_maintenance_model(
            df, selected_features, max_depth, min_samples_split, min_samples_leaf, learning_rate, epochs
        )

    # Display training status
    st.success("Models trained successfully!")

    # Display metrics
    st.subheader("Model Performance Metrics")
    metrics = model_data["metrics"]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        st.metric("Precision", f"{metrics['Precision']:.2%}")

    with col2:
        st.metric("Recall", f"{metrics['Recall']:.2%}")
        st.metric("F1-Score", f"{metrics['F1-Score']:.2%}")

    with col3:
        st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")
        st.metric("Optimal Threshold", f"{metrics['Optimal Threshold']:.3f}")

    # Create visualizations
    st.subheader("Model Visualization")

    # Retrieve prediction results
    y_test = model_data["y_test"]
    y_pred_prob = model_data["y_pred_prob"]
    y_pred = model_data["y_pred"]

    # Create tabs for different plots
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])

    with tab1:
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(conf_matrix,
                           labels=dict(x="Predicted", y="Actual"),
                           x=['No Failure', 'Failure'],
                           y=['No Failure', 'Failure'],
                           text_auto=True,  # Automatically display text on the heatmap
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title='Confusion Matrix')
        st.plotly_chart(fig_cm)

    with tab2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={metrics["ROC-AUC"]:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title='ROC Curve',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              showlegend=True)
        st.plotly_chart(fig_roc)

    with tab3:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall'))
        fig_pr.update_layout(title='Precision-Recall Curve',
                             xaxis_title='Recall',
                             yaxis_title='Precision',
                             showlegend=True)
        st.plotly_chart(fig_pr)

    # ------------------------ #
    # **New Section: Make a Prediction**
    # ------------------------ #
    st.subheader("Make a Prediction")

    st.write("### Enter the feature values to predict failure:")

    # Retrieve models and scaler from cached data
    scaler = model_data["scaler"]
    tree = model_data["tree"]
    lr = model_data["lr"]
    unique_regions = model_data["unique_regions"]
    optimal_threshold = metrics["Optimal Threshold"]

    # Create input fields dynamically based on selected features
    input_features = {}
    for feature in selected_features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            input_features[feature] = st.number_input(
                label=feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        else:
            # Handle non-numeric features if any
            unique_vals = df[feature].unique()
            input_features[feature] = st.selectbox(
                label=feature,
                options=unique_vals,
                index=0
            )

    if st.button("Predict Failure"):
        try:
            # Convert input_features to numpy array
            input_data = np.array(list(input_features.values())).reshape(1, -1)

            # Scale the input features using the same scaler
            input_scaled = scaler.transform(input_data)

            # Predict the region using the decision tree
            input_region = tree.predict(input_scaled)

            # One-hot encode the region using the training unique regions
            input_region_encoded = one_hot_encode(input_region, unique_regions)

            # Combine scaled features and encoded region
            input_hybrid = np.hstack((input_scaled, input_region_encoded))

            # Predict probability using logistic regression
            input_prob = lr.predict_prob(input_hybrid)[0]

            # Apply optimal threshold
            input_pred = (input_prob >= optimal_threshold).astype(int)

            # Display the prediction
            if input_pred == 1:
                st.success(f"**Prediction:** Failure (Probability: {input_prob:.2%})")
            else:
                st.info(f"**Prediction:** No Failure (Probability: {input_prob:.2%})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


# Spare Parts Demand Forecasting
def visualize_spare_part_demand(data):
    """
    Create an interactive plot for the top 20 spare parts demand.
    """
    spare_part_counts = data['spare_part'].value_counts().head(20)
    fig = px.bar(
        x=spare_part_counts.index,
        y=spare_part_counts.values,
        labels={'x': 'Spare Part', 'y': 'Count'},
        title="Top 20 Spare Parts Demand"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)


def visualize_weekly_demand(weekly_data):
    """
    Create an interactive line plot for weekly demand.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_data.index,
        y=weekly_data['spare_part'],
        mode='lines+markers',
        name='Weekly Demand',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title="Weekly Spare Parts Demand",
        xaxis_title="Date",
        yaxis_title="Demand",
        template="plotly_dark"
    )
    st.plotly_chart(fig)


def compare_monthly_weekly_demand(weekly_data, monthly_data):
    """
    Create an interactive line plot comparing weekly vs. monthly demand.
    """
    fig = go.Figure()

    # Weekly Demand
    fig.add_trace(go.Scatter(
        x=weekly_data.index,
        y=weekly_data['spare_part'],
        mode='lines+markers',
        name='Weekly Demand',
        line=dict(color='blue')
    ))

    # Monthly Demand
    fig.add_trace(go.Scatter(
        x=monthly_data.index,
        y=monthly_data['spare_part'],
        mode='lines+markers',
        name='Monthly Demand',
        line=dict(color='green')
    ))

    fig.update_layout(
        title="Weekly vs Monthly Spare Parts Demand",
        xaxis_title="Date",
        yaxis_title="Demand",
        template="plotly_dark"
    )
    st.plotly_chart(fig)


def run_spare_parts_forecasting():
    st.title("Spare Parts Demand Forecasting")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        return

    # Load and Preprocess Data
    data = load_and_clean_data(uploaded_file)
    data = clean_invoice_text(data)
    data = remove_services(data)
    data = prepare_time_series(data)

    # Display Data
    st.subheader("Cleaned Data Preview")
    st.dataframe(data.head(10))

    # Allow User Input for Spare Part
    spare_part_input = st.text_input("Enter Spare Part for Prediction", "")
    if spare_part_input:
        st.subheader(f"Predictions for '{spare_part_input}'")

        # Filter Data for the Selected Spare Part
        spare_part_data = data[data['spare_part'].str.contains(spare_part_input, case=False, na=False)]

        if spare_part_data.empty:
            st.warning(f"No data found for '{spare_part_input}'")
            return

        # Resample Data for the Selected Spare Part
        weekly_data, monthly_data = resample_data(spare_part_data)

        # Sidebar for Parameters
        st.sidebar.header("Holt-Winters Parameters")
        seasonal_periods = st.sidebar.slider("Seasonal Periods", min_value=4, max_value=52, value=26, step=1)
        test_split = st.sidebar.slider("Test Data Split (weeks)", min_value=4, max_value=20, value=16, step=1)

        st.sidebar.header("SARIMA Parameters")
        p = st.sidebar.number_input("AR Order (p)", min_value=0, max_value=10, value=5, step=1)
        d = st.sidebar.number_input("Differencing Order (d)", min_value=0, max_value=2, value=1, step=1)
        q = st.sidebar.number_input("MA Order (q)", min_value=0, max_value=10, value=1, step=1)
        P = st.sidebar.number_input("Seasonal AR Order (P)", min_value=0, max_value=10, value=1, step=1)
        D = st.sidebar.number_input("Seasonal Differencing Order (D)", min_value=0, max_value=2, value=0, step=1)
        Q = st.sidebar.number_input("Seasonal MA Order (Q)", min_value=0, max_value=10, value=0, step=1)
        seasonal_period = st.sidebar.slider("Seasonal Period (s)", min_value=4, max_value=52, value=12, step=1)

        # Holt-Winters Forecasting
        st.subheader(f"Holt-Winters Forecasting for {spare_part_input}")
        train_data, test_data, hw_predictions = forecast_holt_winters(weekly_data, test_split, seasonal_periods)
        hw_next_week = hw_predictions[-7:].sum()
        hw_next_month = hw_predictions.sum()
        st.write(f"Predicted demand for the next week (Holt-Winters): {hw_next_week:.2f}")
        st.write(f"Predicted demand for the next month (Holt-Winters): {hw_next_month:.2f}")
        plot_forecast(weekly_data[:-test_split], weekly_data[-test_split:], hw_predictions,
                      title=f'Holt-Winters Forecast for {spare_part_input}')

        # SARIMA Forecasting
        st.subheader(f"SARIMA Forecasting for {spare_part_input}")
        sarima_predictions, sarima_results = sarima_forecast(
            train_data,
            test_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonal_period)
        )
        sarima_next_week = sarima_predictions[-7:].sum()
        sarima_next_month = sarima_predictions.sum()
        st.write(f"Predicted demand for the next week (SARIMA): {sarima_next_week:.2f}")
        st.write(f"Predicted demand for the next month (SARIMA): {sarima_next_month:.2f}")
        plot_forecast(weekly_data[:-test_split], weekly_data[-test_split:], sarima_predictions,
                      title=f'SARIMA Forecast for {spare_part_input}')

        # Calculate Final Range for Next Week and Month
        st.subheader("Final Demand Range")
        final_week_range = (min(hw_next_week, sarima_next_week), max(hw_next_week, sarima_next_week))
        final_month_range = (min(hw_next_month, sarima_next_month), max(hw_next_month, sarima_next_month))

        st.write(f"**Next Week Demand Range**: {final_week_range[0]:.2f} to {final_week_range[1]:.2f}")
        st.write(f"**Next Month Demand Range**: {final_month_range[0]:.2f} to {final_month_range[1]:.2f}")

    # Visualization
    st.subheader("Top 20 Spare Parts Demand")
    visualize_spare_part_demand(data)

    # Resampling
    weekly_data, monthly_data = resample_data(data)

    # New Graphs
    st.subheader("Weekly Spare Parts Demand")
    visualize_weekly_demand(weekly_data)

    st.subheader("Weekly vs Monthly Spare Parts Demand")
    compare_monthly_weekly_demand(weekly_data, monthly_data)

    # Sidebar for Parameters
    st.sidebar.header("Holt-Winters Parameters")
    seasonal_periods = st.sidebar.slider("Seasonal Periods", min_value=4, max_value=52, value=26, step=1)
    test_split = st.sidebar.slider("Test Data Split (weeks)", min_value=4, max_value=20, value=16, step=1)

    st.sidebar.header("SARIMA Parameters")
    p = st.sidebar.number_input("AR Order (p)", min_value=0, max_value=10, value=5, step=1)
    d = st.sidebar.number_input("Differencing Order (d)", min_value=0, max_value=2, value=1, step=1)
    q = st.sidebar.number_input("MA Order (q)", min_value=0, max_value=10, value=1, step=1)
    P = st.sidebar.number_input("Seasonal AR Order (P)", min_value=0, max_value=10, value=1, step=1)
    D = st.sidebar.number_input("Seasonal Differencing Order (D)", min_value=0, max_value=2, value=0, step=1)
    Q = st.sidebar.number_input("Seasonal MA Order (Q)", min_value=0, max_value=10, value=0, step=1)
    seasonal_period = st.sidebar.slider("Seasonal Period (s)", min_value=4, max_value=52, value=12, step=1)

    # Holt-Winters Forecasting
    st.subheader("Holt-Winters Forecasting")
    train_data, test_data, hw_predictions = forecast_holt_winters(weekly_data, test_split, seasonal_periods)
    plot_forecast(train_data, test_data, hw_predictions, title='Holt-Winters Forecast')

    st.write("### Holt-Winters Forecast Evaluation Metrics")
    hw_mae = mean_absolute_error(test_data, hw_predictions)
    hw_rmse = np.sqrt(mean_squared_error(test_data, hw_predictions))
    st.write(f"Mean Absolute Error (MAE): {hw_mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {hw_rmse:.2f}")

    # SARIMA Forecasting
    st.subheader("SARIMA Forecasting")
    sarima_predictions, sarima_results = sarima_forecast(
        train_data,
        test_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, seasonal_period)
    )
    plot_forecast(train_data, test_data, sarima_predictions, title='SARIMA Forecast')

    st.write("### SARIMA Forecast Evaluation Metrics")
    sarima_mae = mean_absolute_error(test_data, sarima_predictions)
    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_predictions))
    st.write(f"Mean Absolute Error (MAE): {sarima_mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {sarima_rmse:.2f}")


# Supporting Functions
def load_and_clean_data(file):
    data = pd.read_csv(file)
    data = data[pd.notnull(data['invoice_line_text'])].reset_index(drop=True)
    data = data[data['current_km_reading'] <= 100000].reset_index(drop=True)
    data = data[['job_card_date', 'vehicle_model', 'invoice_line_text']]
    return data


def clean_invoice_text(data):
    replacements = {
        'BULB ': 'BULB',
        'OVERHUAL': 'OVERHAUL',
        'WIELDING': 'WELDING',
        'ENGINE OIL TOPUP': 'ENGINE OIL',
        'ASSEBLY': 'ASSEMBLY',
        'GRIP HANDLE': 'HANDLE GRIPPER',
        'HANDLEBAR': 'HANDLE BAR',
        'NUMBER PLATE WITH STICKERS': 'NUMBER PLATE',
    }
    for old, new in replacements.items():
        data['invoice_line_text'] = data['invoice_line_text'].str.replace(old, new)
    return data


def remove_services(data):
    service_related_tokens = [
        'OVERHAUL', 'WELDING', 'SERVICE', 'WORK', 'PUNCHER', 'DENT', 'CHECK',
        'LABOUR', 'CHARGE', 'PAYMENT', 'STICKERS', 'INSURANCE', 'CLEANING'
    ]
    services = [item for item in data['invoice_line_text'].unique()
                if any(token in item for token in service_related_tokens)]
    data = data[~data['invoice_line_text'].isin(services)].reset_index(drop=True)
    return data


def prepare_time_series(data):
    data.rename(columns={"job_card_date": "date", "invoice_line_text": "spare_part"}, inplace=True)
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%y')
    return data


def visualize_spare_part_demand(data):
    plt.figure(figsize=(15, 10))
    sns.countplot(
        data=data, x='spare_part',
        order=data['spare_part'].value_counts().index[:20]
    )
    plt.title('Top 20 Spare Parts', fontsize=20)
    plt.ylabel('Count', fontsize=15)
    plt.xlabel('Spare Part', fontsize=15)
    plt.xticks(rotation=90)
    st.pyplot(plt)


def resample_data(data):
    data.set_index('date', inplace=True)
    weekly_data = data[['spare_part']].resample('W').count()
    monthly_data = weekly_data.resample('M').sum()
    return weekly_data, monthly_data


def forecast_holt_winters(weekly_data, test_split=16, seasonal_periods=26):
    train_data = weekly_data[:-test_split]
    test_data = weekly_data[-test_split:]
    model = ExponentialSmoothing(train_data['spare_part'], trend='mul', seasonal='add',
                                 seasonal_periods=seasonal_periods).fit()
    predictions = model.forecast(len(test_data))
    return train_data, test_data, predictions


def sarima_forecast(train_data, test_data, order=(5, 1, 1), seasonal_order=(1, 0, 0, 12)):
    model = SARIMAX(train_data['spare_part'], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    start = len(train_data)
    end = len(train_data) + len(test_data) - 1
    predictions = results.predict(start=start, end=end)
    return predictions, results


def plot_forecast(train_data, test_data, predictions, title='Forecast'):
    """
    Create an interactive forecast plot comparing train, test, and predicted values.
    """
    fig = go.Figure()

    # Train Data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data['spare_part'],
        mode='lines',
        name='Train Data',
        line=dict(color='blue')
    ))

    # Test Data
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data['spare_part'],
        mode='lines',
        name='Test Data',
        line=dict(color='red', dash='dash')
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=predictions,
        mode='lines',
        name='Predictions',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Demand',
        template="plotly_dark",
        legend=dict(x=0, y=1.1, orientation="h")
    )

    st.plotly_chart(fig)


# Car Sales Prediction
def run_car_sales_prediction():
    st.header("Car Sales Prediction")

    # Sidebar Configuration
    with st.sidebar:
        st.title("Model Parameters")

        # ARIMA Parameters
        st.subheader("ARIMA Parameters")
        p = 5
        d = 2
        q = 0
        months_ahead = st.slider("Prediction Months", 1, 12, 3)

    # Create or upload data
    data = create_sample_dataset()
    uploaded_file = st.file_uploader("Upload your car sales data file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format! Please upload a CSV or Excel file.")
                st.stop()
            st.success("File uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Initialize predictor with selected parameters
    predictor = CarSalesPredictor()

    # Update ARIMA initialization in the fit method
    model_metrics = {}
    for car in data['car_model'].unique():
        car_data = data[data['car_model'] == car]['sales'].values
        if len(car_data) > 2:
            arima = CustomARIMA(p=p, d=d, q=q)
            arima.fit(car_data)
            predictor.arima_models[car] = arima

            # Calculate model metrics
            predictions = arima.predict(len(car_data))
            model_metrics[car] = {
                'MAE': mean_absolute_error(car_data, predictions),
            }

    predictor.fit(data)
    predictions = predictor.predict(data, months_ahead=months_ahead)

    # Real-time Prediction Interface
    st.subheader("Quick Prediction Tool")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_car = st.selectbox("Select Car Model", data['car_model'].unique())
        price = st.number_input("Price (in lakhs)", min_value=0.0, max_value=50.0, value=8.0)

    with col2:
        marketing_spend = st.number_input("Marketing Spend (in lakhs)", min_value=0.0, max_value=200.0, value=75.0)
        festival_season = st.selectbox("Festival Season", [0, 1])

    with col3:
        competitor_launches = st.number_input("Competitor Launches", min_value=0, max_value=10, value=1)
        prediction_month = st.slider("Prediction Month", 1, 12, datetime.now().month)

    if st.button("Generate Quick Prediction"):
        features = np.array([price, marketing_spend, festival_season, competitor_launches, prediction_month])
        if selected_car in predictor.lr_models:
            quick_pred = predictor.lr_models[selected_car].predict(features.reshape(1, -1))
            st.success(f"Predicted sales for {selected_car}: {int(quick_pred[0])} units")

    # Display model metrics for selected car
    st.subheader("Model Performance Metrics")
    if selected_car in model_metrics:
        metrics_df = pd.DataFrame({
            'Metric': list(model_metrics[selected_car].keys()),
            'Value': [f"{value:.4f}" for value in model_metrics[selected_car].values()]
        })
        st.dataframe(metrics_df)

    # Interactive Visualizations
    st.subheader("Sales Analysis and Predictions")

    # 1. Historical vs Predicted Sales (Interactive)
    for car, pred in predictions.items():
        car_data = data[data['car_model'] == car]
        if car_data.empty:
            continue

        future_dates = [car_data['date'].iloc[-1] + timedelta(days=30 * i) for i in range(1, len(pred) + 1)]

        # Create interactive plot using Plotly
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=car_data['date'],
            y=car_data['sales'],
            name='Historical',
            mode='lines+markers'
        ))

        # Predicted data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred,
            name='Predicted',
            mode='lines+markers',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title=f'{car} - Historical vs Predicted Sales',
            xaxis_title='Date',
            yaxis_title='Sales (units)',
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig)

    # 2. Sales Seasonality Analysis
    st.subheader("Seasonality Analysis")

    car_data_seasonal = data.copy()
    car_data_seasonal['month'] = pd.to_datetime(car_data_seasonal['date']).dt.month

    fig_seasonal = px.box(car_data_seasonal, x='month', y='sales', color='car_model',
                          title='Monthly Sales Distribution by Car Model')
    fig_seasonal.update_layout(xaxis_title='Month', yaxis_title='Sales (units)')
    st.plotly_chart(fig_seasonal)

    # 3. Sales Correlation Matrix
    st.subheader("Feature Correlation Analysis")

    correlation_data = data[['sales', 'price', 'marketing_spend', 'festival_season', 'competitor_launches']]
    correlation_matrix = correlation_data.corr()

    fig_corr = px.imshow(correlation_matrix,
                         labels=dict(color="Correlation"),
                         x=correlation_matrix.columns,
                         y=correlation_matrix.columns)
    fig_corr.update_layout(title='Feature Correlation Matrix')
    st.plotly_chart(fig_corr)

    # 4. Marketing Spend vs Sales Scatter Plot
    st.subheader("Marketing Spend vs Sales Analysis")

    fig_scatter = px.scatter(data, x='marketing_spend', y='sales', color='car_model',
                             trendline="ols", title='Marketing Spend vs Sales by Car Model')
    fig_scatter.update_layout(xaxis_title='Marketing Spend (lakhs)',
                              yaxis_title='Sales (units)')
    st.plotly_chart(fig_scatter)

    # 5. Year-over-Year Comparison
    st.subheader("Year-over-Year Comparison")

    yearly_data = data.copy()
    yearly_data['year'] = pd.to_datetime(yearly_data['date']).dt.year
    yearly_sales = yearly_data.groupby(['year', 'car_model'])['sales'].sum().reset_index()

    fig_yoy = px.bar(yearly_sales, x='year', y='sales', color='car_model',
                     title='Yearly Sales Comparison by Car Model',
                     barmode='group')
    fig_yoy.update_layout(xaxis_title='Year', yaxis_title='Total Sales (units)')
    st.plotly_chart(fig_yoy)


# ------------------------ #
# **Run the Main Application**
# ------------------------ #

def main():
    st.set_page_config(page_title="MotorMinds", layout="wide")
    st.title("MotorMinds - Automotive Smart Insights Platform")

    # Sidebar menu
    menu = ["Home", "Predictive Maintenance", "Spare Parts Demand Forecasting", "Car Sales Prediction"]
    choice = st.sidebar.selectbox("Select Analysis", menu)

    if choice == "Home":
        show_home_page()
    elif choice == "Predictive Maintenance":
        run_predictive_maintenance()
    elif choice == "Spare Parts Demand Forecasting":
        run_spare_parts_forecasting()
    elif choice == "Car Sales Prediction":
        run_car_sales_prediction()


# ------------------------ #
# **Run the Main Function**
# ------------------------ #

if __name__ == "__main__":
    main()
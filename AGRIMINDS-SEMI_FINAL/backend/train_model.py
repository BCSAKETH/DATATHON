"""
AgriMinds AI Training Pipeline
================================
Trains two Random Forest models:
1. Crop Recommendation (Classification)
2. Yield Prediction (Regression)

Mathematical Foundation:
------------------------
Random Forest: Ensemble of Decision Trees using Bootstrap Aggregating (Bagging)

For Classification (Crop Recommendation):
- Split Criterion: Gini Impurity
  Gini(D) = 1 - Σ(p_i)² where p_i = proportion of class i in dataset D
  
  At each node, we find feature f and threshold t that minimize:
  Gini_split = (N_left/N)*Gini(D_left) + (N_right/N)*Gini(D_right)
  
  Lower Gini = More pure split (better separation of crop classes)

For Regression (Yield Prediction):
- Split Criterion: Mean Squared Error (MSE)
  MSE(D) = (1/N) * Σ(y_i - ȳ)² where ȳ = mean yield in dataset D
  
  We minimize: MSE_split = (N_left/N)*MSE(D_left) + (N_right/N)*MSE(D_right)

Hyperparameters Rationale:
- n_estimators=100: Balance between accuracy and computation (diminishing returns after 100)
- max_depth=15: Prevent overfitting while capturing complex interactions
- min_samples_split=10: Ensure statistical significance of splits
- random_state=42: Reproducibility for production deployment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AgriMindsTrainer:
    """
    Orchestrates training of both crop recommendation and yield prediction models.
    Implements feature engineering, hyperparameter tuning, and model persistence.
    """
    
    def __init__(self, data_dir='.', model_dir='models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Feature importance tracking
        self.crop_feature_importance = {}
        self.yield_feature_importance = {}
        
    def load_and_prepare_crop_data(self):
        """
        Load crop recommendation dataset and prepare features.
        
        Returns:
            X: Feature matrix [N, P, K, temperature, humidity, ph, rainfall]
            y: Target labels (crop types)
            feature_names: List of feature column names
        """
        print("📊 Loading Crop Recommendation Dataset...")
        df = pd.read_csv(self.data_dir / 'Crop_recommendation.csv')
        
        # Data validation
        print(f"   Dataset shape: {df.shape}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Unique crops: {df['label'].nunique()}")
        
        # Feature engineering: NPK ratio (agronomically significant)
        df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1e-6)  # Avoid division by zero
        df['moisture_index'] = df['humidity'] * df['rainfall']  # Soil moisture proxy
        
        # Separate features and target
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                       'NPK_ratio', 'moisture_index']
        X = df[feature_cols].values
        y = df['label'].values
        
        # Encode crop labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save label mapping for API
        label_mapping = {idx: label for idx, label in enumerate(self.label_encoder.classes_)}
        with open(self.model_dir / 'crop_labels.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"   ✓ Features engineered: {feature_cols}")
        return X, y_encoded, feature_cols
    
    def load_and_prepare_yield_data(self):
        """
        Load yield prediction dataset and prepare features.
        
        Returns:
            X: Feature matrix [soil_moisture_%, NDVI_index, temperature, rainfall_mm]
            y: Target (yield_kg_per_hectare)
            feature_names: List of feature column names
        """
        print("\n📊 Loading Yield Prediction Dataset...")
        df = pd.read_csv(self.data_dir / 'Smart_Farming_Crop_Yield_2024.csv')
        
        print(f"   Dataset shape: {df.shape}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering: Vegetation health index
        df['VHI'] = df['NDVI_index'] * df['soil_moisture_%']  # Combined vegetation + water stress
        df['thermal_stress'] = np.where(df['temperature'] > 35, 1, 0)  # Heat stress indicator
        
        feature_cols = ['soil_moisture_%', 'NDVI_index', 'temperature', 'rainfall_mm', 
                       'VHI', 'thermal_stress']
        X = df[feature_cols].values
        y = df['yield_kg_per_hectare'].values
        
        print(f"   ✓ Features engineered: {feature_cols}")
        return X, y, feature_cols
    
    def train_crop_model(self, X, y, feature_names):
        """
        Train Random Forest Classifier for crop recommendation.
        
        Mathematical Process:
        1. Bootstrap sampling: Create B=100 datasets via sampling with replacement
        2. For each tree: Recursively split using Gini impurity minimization
        3. Prediction: Majority vote across all trees (ensemble democracy)
        
        Hyperparameter Tuning Strategy:
        - Grid search over {max_depth, min_samples_split}
        - 5-fold cross-validation to prevent overfitting
        """
        print("\n🌾 Training Crop Recommendation Model...")
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [12, 15, 18],
            'min_samples_split': [8, 10, 12],
            'min_samples_leaf': [3, 4]
        }
        
        print("   🔧 Performing Grid Search (this may take a few minutes)...")
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        print(f"   ✓ Best CV accuracy: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        self.crop_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.crop_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n   📈 Test Set Accuracy: {accuracy:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_,
                                   zero_division=0))
        
        # Feature importance (based on Gini decrease)
        importances = self.crop_model.feature_importances_
        self.crop_feature_importance = dict(zip(feature_names, importances))
        
        print("\n   🔍 Feature Importance (Gini Decrease):")
        for feat, imp in sorted(self.crop_feature_importance.items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"      {feat:20s}: {imp:.4f}")
        
        # Save model
        joblib.dump(self.crop_model, self.model_dir / 'crop_model.pkl')
        joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.pkl')
        print(f"\n   ✓ Model saved to {self.model_dir / 'crop_model.pkl'}")
        
        return accuracy
    
    def train_yield_model(self, X, y, feature_names):
        """
        Train Random Forest Regressor for yield prediction.
        
        Mathematical Process:
        1. Bootstrap sampling: Create B=100 datasets
        2. For each tree: Split using MSE minimization
        3. Prediction: Average predictions across all trees
        
        Evaluation Metrics:
        - RMSE: √(MSE) in kg/hectare (interpretable units)
        - R²: Proportion of variance explained (0-1, higher is better)
        - MAE: Mean absolute error (robust to outliers)
        """
        print("\n🌱 Training Yield Prediction Model...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling (important for yield regression)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        print("   🔧 Performing Grid Search...")
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='neg_mean_squared_error', 
            verbose=0, n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        
        # Train final model
        self.yield_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.yield_model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n   📈 Test Set Metrics:")
        print(f"      RMSE: {rmse:.2f} kg/hectare")
        print(f"      MAE:  {mae:.2f} kg/hectare")
        print(f"      R²:   {r2:.4f}")
        
        # Feature importance
        importances = self.yield_model.feature_importances_
        self.yield_feature_importance = dict(zip(feature_names, importances))
        
        print("\n   🔍 Feature Importance (MSE Decrease):")
        for feat, imp in sorted(self.yield_feature_importance.items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"      {feat:20s}: {imp:.4f}")
        
        # Save model and scaler
        joblib.dump(self.yield_model, self.model_dir / 'yield_model.pkl')
        joblib.dump(self.scaler, self.model_dir / 'yield_scaler.pkl')
        print(f"\n   ✓ Model saved to {self.model_dir / 'yield_model.pkl'}")
        
        return rmse, r2
    
    def generate_training_report(self, crop_accuracy, yield_rmse, yield_r2):
        """Generate comprehensive training report"""
        report = {
            'crop_model': {
                'accuracy': float(crop_accuracy),
                'feature_importance': self.crop_feature_importance
            },
            'yield_model': {
                'rmse': float(yield_rmse),
                'r2': float(yield_r2),
                'feature_importance': self.yield_feature_importance
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.model_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Training report saved to {self.model_dir / 'training_report.json'}")


def main():
    """Execute the complete training pipeline"""
    print("=" * 70)
    print("🚀 AgriMinds ML Training Pipeline")
    print("=" * 70)
    
    trainer = AgriMindsTrainer()
    
    # Train Crop Recommendation Model
    X_crop, y_crop, crop_features = trainer.load_and_prepare_crop_data()
    crop_accuracy = trainer.train_crop_model(X_crop, y_crop, crop_features)
    
    # Train Yield Prediction Model
    X_yield, y_yield, yield_features = trainer.load_and_prepare_yield_data()
    yield_rmse, yield_r2 = trainer.train_yield_model(X_yield, y_yield, yield_features)
    
    # Generate final report
    trainer.generate_training_report(crop_accuracy, yield_rmse, yield_r2)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete! Models ready for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
# src/ml_models/health_predictor.py
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
from src.utils.logger_config import logger

class HealthPredictor:
    """
    Production-ready ML model for predicting confidence scores for startup health assessments.
    Includes model training, prediction, logging, and retraining capabilities.
    """
    
    def __init__(self, model_dir: str = "/opt/airflow/ml_models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "health_predictor_model.pkl")
        self.scaler_path = os.path.join(model_dir, "health_predictor_scaler.pkl")
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
        mlflow.set_experiment("startup_health_predictor")
        
        # Try to load existing model
        self._load_model()
        
        logger.info("HealthPredictor initialized")
    
    def _load_model(self) -> bool:
        """Load existing model and scaler if available."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load feature names if available
                feature_names_path = os.path.join(self.model_dir, "feature_names.json")
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        self.feature_names = json.load(f)
                
                self.is_trained = True
                logger.info("Model and scaler loaded successfully")
                return True
            else:
                logger.info("No existing model found. Will use heuristic confidence until model is trained.")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_model(self):
        """Save model, scaler, and feature names."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            if self.feature_names:
                with open(os.path.join(self.model_dir, "feature_names.json"), 'w') as f:
                    json.dump(self.feature_names, f)
                    
            logger.info("Model and scaler saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _extract_features(self, normalized_kpis: Dict[str, float], 
                         category_scores: Dict[str, float],
                         total_score: float) -> np.ndarray:
        """
        Extract features from normalized KPIs and scores for ML model.
        
        Args:
            normalized_kpis: Dictionary of normalized KPI scores
            category_scores: Dictionary of category scores
            total_score: Total health score
            
        Returns:
            Numpy array of features
        """
        # Start with category scores
        features = []
        category_order = sorted(category_scores.keys())
        for category in category_order:
            features.append(category_scores.get(category, 0))
        
        # Add total score
        features.append(total_score)
        
        # Add statistics about normalized KPIs
        kpi_values = list(normalized_kpis.values())
        features.append(np.mean(kpi_values))
        features.append(np.std(kpi_values))
        features.append(np.min(kpi_values))
        features.append(np.max(kpi_values))
        features.append(len([v for v in kpi_values if v > 70]))  # Count of high-performing KPIs
        features.append(len([v for v in kpi_values if v < 30]))  # Count of low-performing KPIs
        
        # Add specific important KPIs if available
        important_kpis = [
            "Monthly Recurring Revenue (MRR)",
            "Burn Rate",
            "Runway (in Months)",
            "Net Revenue Retention (NRR)",
            "Customer (Logo) Churn Rate"
        ]
        
        for kpi in important_kpis:
            features.append(normalized_kpis.get(kpi, 0))
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_heuristic_confidence(self, normalized_kpis: Dict[str, float],
                                       category_scores: Dict[str, float],
                                       missing_mandatory_kpis: List[str],
                                       missing_non_mandatory_kpis: List[str]) -> float:
        """
        Calculate heuristic confidence score when ML model is not available.
        
        Args:
            normalized_kpis: Dictionary of normalized KPI scores
            category_scores: Dictionary of category scores
            missing_mandatory_kpis: List of missing mandatory KPIs
            missing_non_mandatory_kpis: List of missing non-mandatory KPIs
            
        Returns:
            Heuristic confidence score (0-100)
        """
        # Base confidence
        confidence = 100.0
        
        # Penalize for missing mandatory KPIs
        confidence -= len(missing_mandatory_kpis) * 15
        
        # Penalize for missing non-mandatory KPIs
        confidence -= len(missing_non_mandatory_kpis) * 5
        
        # Penalize for high variance in category scores
        if len(category_scores) > 1:
            scores_std = np.std(list(category_scores.values()))
            confidence -= min(20, scores_std * 2)  # Cap penalty at 20 points
        
        # Penalize for low-performing KPIs
        low_kpis = [v for v in normalized_kpis.values() if v < 30]
        if low_kpis:
            confidence -= min(15, len(low_kpis) * 2)
        
        # Ensure confidence is within 0-100 range
        return max(0, min(100, confidence))
    
    def predict_confidence(self, normalized_kpis: Dict[str, float],
                          category_scores: Dict[str, float],
                          total_score: float,
                          missing_mandatory_kpis: List[str],
                          missing_non_mandatory_kpis: List[str]) -> Tuple[float, str]:
        """
        Predict confidence score for the health assessment.
        
        Args:
            normalized_kpis: Dictionary of normalized KPI scores
            category_scores: Dictionary of category scores
            total_score: Total health score
            missing_mandatory_kpis: List of missing mandatory KPIs
            missing_non_mandatory_kpis: List of missing non-mandatory KPIs
            
        Returns:
            Tuple of (confidence_score, prediction_method)
        """
        if self.is_trained and self.model is not None:
            try:
                # Extract features
                features = self._extract_features(normalized_kpis, category_scores, total_score)
                
                # Scale features
                scaled_features = self.scaler.transform(features)
                
                # Predict confidence
                confidence = float(self.model.predict(scaled_features)[0])
                
                # Ensure confidence is within 0-100 range
                confidence = max(0, min(100, confidence))
                
                logger.info(f"ML model predicted confidence: {confidence:.2f}")
                return confidence, "ml_model"
            except Exception as e:
                logger.error(f"Error predicting with ML model: {str(e)}. Falling back to heuristic.")
        
        # Fall back to heuristic if model is not available or prediction fails
        confidence = self._calculate_heuristic_confidence(
            normalized_kpis, category_scores, missing_mandatory_kpis, missing_non_mandatory_kpis
        )
        logger.info(f"Heuristic confidence calculated: {confidence:.2f}")
        return confidence, "heuristic"
    
    def train_model(self, data_path: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the ML model using historical data.
        
        Args:
            data_path: Path to CSV file with historical data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting model training with data from {data_path}")
        
        # Load data
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} records for training")
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        # Prepare features and target
        try:
            # Parse JSON columns
            df['normalized_kpis'] = df['normalized_kpis'].apply(json.loads)
            df['category_scores'] = df['category_scores'].apply(json.loads)
            
            # Extract features
            features_list = []
            for _, row in df.iterrows():
                features = self._extract_features(
                    row['normalized_kpis'], 
                    row['category_scores'], 
                    row['total_score']
                )
                features_list.append(features.flatten())
            
            X = np.array(features_list)
            y = df['confidence_score'].values
            
            # Store feature names
            self.feature_names = [
                f"category_{i}" for i in range(len(df.iloc[0]['category_scores']))
            ] + [
                "total_score", "kpi_mean", "kpi_std", "kpi_min", "kpi_max", 
                "high_kpis_count", "low_kpis_count"
            ] + [
                f"kpi_{kpi.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')}" 
                for kpi in [
                    "Monthly Recurring Revenue (MRR)",
                    "Burn Rate",
                    "Runway (in Months)",
                    "Net Revenue Retention (NRR)",
                    "Customer (Logo) Churn Rate"
                ]
            ]
            
            logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", len(df))
            
            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_mse", test_mse)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Model trained. Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        # Save model and scaler
        self._save_model()
        self.is_trained = True
        
        return {
            "status": "success",
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "n_samples": len(df),
            "n_features": X.shape[1]
        }
    
    def log_prediction(self, document_name: str, normalized_kpis: Dict[str, float],
                      category_scores: Dict[str, float], total_score: float,
                      confidence_score: float, prediction_method: str,
                      missing_mandatory_kpis: List[str],
                      missing_non_mandatory_kpis: List[str]):
        """
        Log prediction data to CSV for future retraining.
        
        Args:
            document_name: Name of the processed document
            normalized_kpis: Dictionary of normalized KPI scores
            category_scores: Dictionary of category scores
            total_score: Total health score
            confidence_score: Predicted confidence score
            prediction_method: Method used for prediction (ml_model or heuristic)
            missing_mandatory_kpis: List of missing mandatory KPIs
            missing_non_mandatory_kpis: List of missing non-mandatory KPIs
        """
        log_file = os.path.join(self.model_dir, "health_scores_log.csv")
        
        # Create log file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,document_name,total_score,confidence_score,prediction_method," +
                        "normalized_kpis,category_scores,missing_mandatory_kpis,missing_non_mandatory_kpis\n")
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "document_name": document_name,
            "total_score": total_score,
            "confidence_score": confidence_score,
            "prediction_method": prediction_method,
            "normalized_kpis": json.dumps(normalized_kpis),
            "category_scores": json.dumps(category_scores),
            "missing_mandatory_kpis": json.dumps(missing_mandatory_kpis),
            "missing_non_mandatory_kpis": json.dumps(missing_non_mandatory_kpis)
        }
        
        # Write to log file
        try:
            df_log = pd.DataFrame([log_entry])
            df_log.to_csv(log_file, mode='a', header=False, index=False)
            logger.info(f"Logged prediction for {document_name} to {log_file}")
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
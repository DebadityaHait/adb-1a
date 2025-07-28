from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging

class MLClassifier:
    """Machine Learning classifier for heading detection using Random Forest."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10,
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.class_names = ["non_heading", "H1", "H2", "H3"]
        
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, labeled_blocks: List[Dict[str, Any]], 
                            feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from labeled blocks.
        
        Args:
            labeled_blocks: List of text blocks with labels and features
            feature_names: List of feature names to extract
            
        Returns:
            Tuple of (X, y) for training
        """
        # Extract features
        X_list = []
        y_list = []
        
        for block in labeled_blocks:
            # Extract feature values
            features = []
            for feature_name in feature_names:
                value = block.get(feature_name, 0)
                # Handle boolean features
                if isinstance(value, bool):
                    value = float(value)
                features.append(value)
            
            X_list.append(features)
            
            # Extract label
            if block.get("is_heading", False):
                y_list.append(block.get("heading_level", "H2"))
            else:
                y_list.append("non_heading")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        self.feature_names = feature_names
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              balance_classes: bool = True) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            balance_classes: Whether to balance class weights
            
        Returns:
            Training metrics
        """
        # Handle class imbalance
        class_weight = None
        if balance_classes:
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight = dict(zip(classes, class_weights))
            self.logger.info(f"Using class weights: {class_weight}")
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='f1_weighted')
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        metrics = {
            "cv_mean_score": cv_scores.mean(),
            "cv_std_score": cv_scores.std(),
            "feature_importance": feature_importance,
            "top_features": top_features
        }
        
        self.logger.info(f"Training completed. CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        self.logger.info(f"Top features: {[f[0] for f in top_features[:5]]}")
        
        return metrics
    
    def predict(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict headings for text blocks.
        
        Args:
            text_blocks: List of text blocks with features
            
        Returns:
            List of text blocks with ML predictions
        """
        if not self.model or not self.scaler:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X_list = []
        for block in text_blocks:
            features = []
            for feature_name in self.feature_names:
                value = block.get(feature_name, 0)
                if isinstance(value, bool):
                    value = float(value)
                features.append(value)
            X_list.append(features)
        
        X = np.array(X_list, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add predictions to blocks
        results = []
        for i, block in enumerate(text_blocks):
            result = block.copy()
            
            pred_label = predictions[i]
            pred_probs = probabilities[i]
            max_prob = max(pred_probs)
            
            result.update({
                "ml_predicted_level": pred_label if pred_label != "non_heading" else None,
                "ml_is_heading": pred_label != "non_heading",
                "ml_confidence": max_prob,
                "ml_probabilities": dict(zip(self.model.classes_, pred_probs))
            })
            
            results.append(result)
        
        self.logger.info(f"Made ML predictions for {len(results)} blocks")
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained.")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        }
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """Save the trained model and scaler."""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "model_params": self.model.get_params()
        }
        
        metadata_path = model_path.replace(".joblib", "_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """Load a trained model and scaler."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = model_path.replace(".joblib", "_metadata.joblib")
        try:
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata["feature_names"]
            self.class_names = metadata["class_names"]
        except FileNotFoundError:
            self.logger.warning("Metadata file not found")
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.model:
            raise ValueError("Model not trained.")
        
        return dict(zip(self.feature_names, self.model.feature_importances_)) 
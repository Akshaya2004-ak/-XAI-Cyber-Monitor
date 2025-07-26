import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from utils.feature_extraction import extract_features
from utils.dummy_data import generate_large_dataset

def train_enhanced_model():
    """Train enhanced model with better performance and larger dataset"""
    
    print("🚀 Starting Enhanced Model Training...")
    
    # Created model directory
    os.makedirs("model", exist_ok=True)
    
    # Generated larger dataset for better training
    print("📊 Generating enhanced training dataset...")
    generate_large_dataset()
    
    # Loaded training data
    df = pd.read_csv("enhanced_training_data.csv")
    print(f"✅ Loaded {len(df)} training samples")
    
    # Extracted features
    print("🔍 Extracting features...")
    X = []
    for i, url in enumerate(df["url"]):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(df)} URLs...")
        features = extract_features(url)
        X.append(features)
    
    X = np.array(X)
    y = df["label"].values
    
    print(f"📏 Feature matrix shape: {X.shape}")
    print(f"🎯 Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning
    print("🔧 Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    print(f"🎯 Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Training final model
    best_model = grid_search.best_estimator_
    
    # Evaluating model
    print("📊 Evaluating model performance...")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    print(f"\n🔄 Cross-validation scores: {cv_scores}")
    print(f"🔄 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_names = [
        'Token Count', 'Token Length Sum', 'Avg Token Length', 'Max Token Length', 'URL Length',
        'Special Chars', 'Encoded Chars', 'Numeric Chars', 'Query Length',
        'SQL Keywords', 'SQL Operators', 'SQL Functions', 'SQL Comments', 'SQL Quotes',
        'SQL Equals', 'SQL Union', 'SQL OR Pattern',
        'XSS Keywords', 'XSS Tags', 'XSS Events', 'XSS JavaScript', 'XSS Encoded', 'XSS Entities',
        'Query Entropy', 'Path Entropy', 'URL Entropy',
        'Param Count', 'Avg Param Length', 'Suspicious Params',
        'Directory Traversal', 'File Inclusion', 'Command Injection'
    ]
    
    importance_pairs = list(zip(feature_names, best_model.feature_importances_))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(importance_pairs[:10], 1):
        print(f"{i:2d}. {feature:20s}: {importance:.4f}")
    
    # Save model
    model_path = "model/rf_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save training metadata
    metadata = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_,
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import json
    with open("model/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Enhanced model training completed!")
    return best_model

if __name__ == "__main__":
    train_enhanced_model()
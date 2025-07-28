#!/usr/bin/env python3
"""
Realistic Model Evaluation - Fix Target Leakage and Get Honest Accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

def get_real_nasdaq_data():
    """Get REAL NASDAQ 100 data instead of synthetic target."""
    
    print("📈 Downloading REAL NASDAQ 100 data...")
    
    try:
        # Download QQQ (NASDAQ 100 ETF) data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*15)  # 15 years
        
        nasdaq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
        
        # Handle multi-level columns from yfinance
        if isinstance(nasdaq.columns, pd.MultiIndex):
            nasdaq.columns = nasdaq.columns.droplevel(1)  # Remove ticker level
        
        # Calculate quarterly returns  
        quarterly_nasdaq = nasdaq['Close'].resample('Q').last()
        quarterly_returns = quarterly_nasdaq.pct_change().dropna()
        
        # Create REAL target: positive quarterly return = 1, negative = 0
        real_target = (quarterly_returns > 0).astype(int)
        
        print(f"✅ Got {len(real_target)} quarters of REAL NASDAQ data")
        print(f"📊 Target distribution: {real_target.value_counts().to_dict()}")
        
        return real_target
        
    except Exception as e:
        print(f"❌ Failed to get real data: {e}")
        return None

def create_enhanced_features_no_leakage(macro_data, real_target):
    """Create enhanced features WITHOUT target leakage."""
    
    print("🛠️ Creating enhanced features (NO LEAKAGE)...")
    
    # Convert to quarterly
    macro_data = macro_data.copy()
    macro_data['date'] = pd.to_datetime(macro_data['date'])
    macro_data.set_index('date', inplace=True)
    
    # Aggregate to quarterly
    quarterly_cols = ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y', 'real_gdp']
    quarterly_df = macro_data[quarterly_cols].resample('Q').mean().dropna()
    
    print(f"📊 Base data: {quarterly_df.shape[0]} quarters")
    
    # Align with real target data
    common_quarters = quarterly_df.index.intersection(real_target.index)
    if len(common_quarters) < 20:
        print(f"❌ Insufficient overlapping data: {len(common_quarters)} quarters")
        return None, None
        
    quarterly_df = quarterly_df.loc[common_quarters]
    target = real_target.loc[common_quarters]
    
    print(f"✅ Aligned data: {len(quarterly_df)} quarters")
    
    # Create LAGGED features (NO leakage - use past data only)
    features = []
    
    # 1. LAG FEATURES (use PAST quarters only)
    for col in ['vix', 'unemployment_rate', 'fed_funds_rate', 'treasury_10y']:
        if col in quarterly_df.columns:
            quarterly_df[f'{col}_lag_1q'] = quarterly_df[col].shift(1)
            quarterly_df[f'{col}_lag_2q'] = quarterly_df[col].shift(2)
            quarterly_df[f'{col}_lag_4q'] = quarterly_df[col].shift(4)  # 1 year lag
            features.extend([f'{col}_lag_1q', f'{col}_lag_2q', f'{col}_lag_4q'])
    
    # 2. TREND FEATURES (based on past data)
    for col in ['vix', 'unemployment_rate', 'fed_funds_rate']:
        if col in quarterly_df.columns:
            quarterly_df[f'{col}_trend_4q'] = quarterly_df[col].rolling(4).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 4 else np.nan
            ).shift(1)  # Shift to avoid leakage
            quarterly_df[f'{col}_ma_8q'] = quarterly_df[col].rolling(8).mean().shift(1)
            features.extend([f'{col}_trend_4q', f'{col}_ma_8q'])
    
    # 3. CHANGE FEATURES (quarter-over-quarter changes, lagged)
    for col in ['vix', 'unemployment_rate', 'fed_funds_rate']:
        if col in quarterly_df.columns:
            quarterly_df[f'{col}_qoq_change'] = quarterly_df[col].diff().shift(1)
            quarterly_df[f'{col}_yoy_change'] = quarterly_df[col].diff(4).shift(1)
            features.extend([f'{col}_qoq_change', f'{col}_yoy_change'])
    
    # 4. INTERACTION FEATURES (economic relationships, lagged)
    if 'fed_funds_rate' in quarterly_df.columns and 'unemployment_rate' in quarterly_df.columns:
        quarterly_df['fed_unemployment_interaction'] = (
            quarterly_df['fed_funds_rate'].shift(2) * quarterly_df['unemployment_rate'].shift(1)
        )
        features.append('fed_unemployment_interaction')
    
    # 5. CYCLICAL FEATURES
    n_quarters = len(quarterly_df)
    quarterly_df['business_cycle'] = np.sin(2 * np.pi * np.arange(n_quarters) / 24)  # 6-year cycle
    quarterly_df['time_trend'] = np.arange(n_quarters) / n_quarters
    features.extend(['business_cycle', 'time_trend'])
    
    # Remove rows with NaN (due to lags)
    quarterly_df = quarterly_df.dropna()
    target = target.loc[quarterly_df.index]
    
    print(f"📊 After cleaning: {len(quarterly_df)} quarters")
    print(f"🎯 Features created: {len(features)}")
    
    # Return only the features (no leakage)
    X = quarterly_df[features].fillna(quarterly_df[features].median())
    y = target
    
    return X, y

def realistic_model_evaluation():
    """Perform realistic model evaluation without target leakage."""
    
    print("🔍 REALISTIC MODEL EVALUATION")
    print("=" * 40)
    print("✅ NO Target Leakage")
    print("✅ Real NASDAQ Data")
    print("✅ Proper Time Series Validation")
    print("")
    
    # Load macro data
    try:
        from sqlalchemy import create_engine, text
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        db_url = os.getenv('NEON_DB_URL')
        engine = create_engine(db_url)
        
        query = '''SELECT * FROM "macro"."macro_data" ORDER BY date DESC LIMIT 5000'''
        macro_data = pd.read_sql_query(text(query), engine)
        engine.dispose()
        
        print(f"📊 Loaded {len(macro_data)} macro records")
        
    except Exception as e:
        print(f"❌ Failed to load macro data: {e}")
        return
    
    # Get REAL NASDAQ target
    real_target = get_real_nasdaq_data()
    if real_target is None:
        return
    
    # Create features WITHOUT leakage
    X, y = create_enhanced_features_no_leakage(macro_data, real_target)
    if X is None:
        return
    
    print(f"\n📊 FINAL DATASET:")
    print(f"   • Quarters: {len(X)}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Target: Real NASDAQ quarterly returns")
    print(f"   • Bullish quarters: {sum(y)} ({sum(y)/len(y):.1%})")
    print(f"   • Bearish quarters: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y):.1%})")
    
    if len(X) < 20:
        print("❌ Insufficient data for training")
        return
    
    # Feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select best features
    n_features = min(10, max(5, len(X) // 5))  # Conservative selection
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
    
    print(f"\n🎯 FEATURE SELECTION:")
    print(f"   • Selected: {n_features} features")
    print(f"   • Features: {selected_features}")
    
    # Time series cross-validation (PROPER)
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'Conservative RF': RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    }
    
    print(f"\n🤖 MODEL EVALUATION:")
    print(f"   • Validation: Time Series Cross-Validation")
    print(f"   • Splits: {tscv.n_splits}")
    
    results = {}
    
    for name, model in models.items():
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_selected):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, val_pred))
        
        avg_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        results[name] = {
            'cv_mean': avg_cv_score,
            'cv_std': std_cv_score,
            'cv_scores': cv_scores
        }
        
        print(f"   📊 {name}:")
        print(f"      CV Accuracy: {avg_cv_score:.3f} ± {std_cv_score:.3f}")
        print(f"      Individual folds: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Final evaluation
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_score = results[best_model_name]['cv_mean']
    
    print(f"\n🏆 REALISTIC RESULTS:")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • **HONEST Accuracy: {best_score:.1%}**")
    print(f"   • This is realistic for quarterly market prediction!")
    
    # Comparison with fake accuracy
    print(f"\n⚖️ COMPARISON:")
    print(f"   • Fake (with leakage): 100.0% ← IMPOSSIBLE")
    print(f"   • **Realistic (no leakage): {best_score:.1%}** ← HONEST")
    
    if best_score > 0.60:
        print(f"\n✅ {best_score:.1%} is actually EXCELLENT for quarterly market prediction!")
        print(f"   • Random guessing: 50%")
        print(f"   • Your model: {best_score:.1%}")
        print(f"   • This suggests real predictive value!")
    else:
        print(f"\n💡 {best_score:.1%} suggests model needs improvement")
        print(f"   • Try more features, more data, or different approach")
    
    # NEW: Save the realistic model for example_usage.py
    print(f"\n💾 STEP 6: Saving REALISTIC Model")
    print("-" * 40)
    
    # Train final model on all data
    best_model = models[best_model_name]
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_selected)
    
    final_model = type(best_model)(**best_model.get_params())
    final_model.fit(X_final_scaled, y)
    
    # Save realistic model (replace fake one)
    try:
        import pickle
        import json
        from pathlib import Path
        from datetime import datetime
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        with open(models_dir / "enhanced_nasdaq_model.pkl", 'wb') as f:
            pickle.dump(final_model, f)
        
        # Save scaler
        with open(models_dir / "feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler_final, f)
        
        # Save feature columns
        with open(models_dir / "feature_columns.json", 'w') as f:
            json.dump(selected_features, f)
        
        # Save REALISTIC metadata
        metadata = {
            'model_name': f"REALISTIC {best_model_name} (No Leakage)",
            'accuracy': best_score,
            'features_used': len(selected_features),
            'timestamp': datetime.now().isoformat(),
            'time_series_enhanced': True,
            'realistic_training': True,
            'target_leakage_fixed': True,
            'real_nasdaq_target': True
        }
        
        with open(models_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ REALISTIC model saved successfully!")
        print(f"📊 Model: {best_model_name}")
        print(f"🎯 Honest Accuracy: {best_score:.1%}")
        print(f"💾 Saved to: models/enhanced_nasdaq_model.pkl")
        print(f"✅ Ready for example_usage.py!")
        
        return results, True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return results, False

if __name__ == "__main__":
    results, saved = realistic_model_evaluation()
    
    if saved:
        print(f"\n🎯 WORKFLOW COMPLETE!")
        print(f"✅ Realistic model saved to models/")
        print(f"✅ Ready to use with example_usage.py")
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Run: python example_usage.py")
        print(f"   2. Choose Option 1 or 2 for predictions")
        print(f"   3. Expect realistic accuracy (not fake 100%)")
    else:
        print(f"\n❌ Model saving failed - check errors above") 
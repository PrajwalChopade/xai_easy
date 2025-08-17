import pandas as pd
import warnings
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xai_easy import explain_model, explain_instance, save_html_report

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Generate sample dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_df, y)

# 1. Global Explanation
global_exp = explain_model(model, X_df, y, task="classification", top_n=5)
print("🔍 Global Feature Importance:")
print(global_exp[['rank', 'feature', 'importance']])

print("\n" + "="*60)

# 2. Local Explanation (single instance) - Fix: pass DataFrame row instead of numpy array
local_exp = explain_instance(model, X_df, X_df.iloc[0], feature_names=feature_names)
print("📋 Local Instance Explanation:")
print(local_exp[['feature', 'contribution']].head())

print("\n" + "="*60)

# 3. Save an interactive HTML report
save_html_report(global_exp, local_exp, title="Professional ML Model Analysis Report", filename="report.html")

# Success message
print("\n🎉 Professional HTML report generated successfully!")
print("📁 File: report.html")
print("👨‍💻 Created with XAI Easy by Prajwal")
print("🔍 Features include:")
print("   ✓ Professional styling and layout")
print("   ✓ Interactive charts and visualizations")
print("   ✓ Comprehensive model explanations")
print("   ✓ Author attribution and branding")

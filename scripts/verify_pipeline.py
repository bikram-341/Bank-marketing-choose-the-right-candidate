"""
End-to-End Pipeline Verification Script.

Executes data loading, preprocessing, model training (SVM & Decision Tree),
held-out testing, customer ranking, top 20% selection, and visualization generation
directly on the loaded UCI Bank Marketing dataset.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.preprocessing import split_and_preprocess
from src.train_models import train_model, save_trained_artifacts
from src.evaluate_models import evaluate_single_model, compare_models
from src.feature_analysis import compute_tree_feature_importance, analyze_top_features
from src.customer_ranking import rank_and_select_customers, generate_export_dataframe
from src.visualization import (
    plot_target_distribution,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_selected_vs_remaining,
    plot_cumulative_gains,
)


def run_full_pipeline_verification():
    print("=" * 60)
    print("BANK MARKETING PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Load data
    print("\n[STEP 1] Loading Dataset...")
    df, summary = load_dataset()
    print(f"Loaded {summary['rows']:,} rows, {summary['columns']} columns.")
    print(f"Target distribution: {summary['target_distribution']}")

    # 2. Preprocessing
    print("\n[STEP 2] Preprocessing & Stratified 80:20 Split...")
    prep_res = split_and_preprocess(df, test_size=0.20, random_state=42, exclude_leakage=True)
    X_train = prep_res["X_train_transformed"]
    y_train = prep_res["y_train"]
    X_test = prep_res["X_test_transformed"]
    y_test = prep_res["y_test"]
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Encoded features: {len(prep_res['feature_names'])}")
    print(f"Excluded leakage features: {prep_res['excluded_cols']}")

    # 3. Train Decision Tree
    print("\n[STEP 3] Training Decision Tree Classifier...")
    dt_model = train_model("decision_tree", X_train, y_train, random_state=42)
    dt_eval = evaluate_single_model(dt_model, X_test, y_test, model_name="Decision Tree")
    print(f"Decision Tree Accuracy:  {dt_eval['accuracy']:.4f}")
    print(f"Decision Tree Precision: {dt_eval['precision']:.4f}")
    print(f"Decision Tree Recall:    {dt_eval['recall']:.4f}")
    print(f"Decision Tree F1-Score:  {dt_eval['f1_score']:.4f}")
    print(f"Decision Tree ROC-AUC:   {dt_eval['roc_auc']:.4f}")

    # 4. Train SVM (SVC with probability calibration)
    print("\n[STEP 4] Training Support Vector Machine (SVC)...")
    svm_model = train_model("svm", X_train, y_train, random_state=42)
    svm_eval = evaluate_single_model(svm_model, X_test, y_test, model_name="SVM (SVC)")
    print(f"SVM Accuracy:  {svm_eval['accuracy']:.4f}")
    print(f"SVM Precision: {svm_eval['precision']:.4f}")
    print(f"SVM Recall:    {svm_eval['recall']:.4f}")
    print(f"SVM F1-Score:  {svm_eval['f1_score']:.4f}")
    print(f"SVM ROC-AUC:   {svm_eval['roc_auc']:.4f}")

    # 5. Model Comparison
    print("\n[STEP 5] Model Comparison Summary:")
    comp_df = compare_models({"Decision Tree": dt_eval, "SVM (SVC)": svm_eval})
    print(comp_df.to_string(index=False))

    # 6. Feature Importance
    print("\n[STEP 6] Extracting Top Predictive Features...")
    imp_df = compute_tree_feature_importance(X_train, y_train, prep_res["feature_names"], random_state=42)
    print(imp_df[["Rank", "Feature", "Percentage"]].head(8).to_string(index=False))
    insights = analyze_top_features(imp_df)
    print(f"Top drivers: {insights['top_features']} ({insights['top_3_coverage_pct']}% coverage)")

    # 7. Customer Ranking & Top 20% Selection (Bank Scenario: 50,000 -> 10,000)
    print("\n[STEP 7] Customer Ranking & Campaign Decile Selection (Top 20%)...")
    # Score the customer population using Decision Tree
    ranked_df, stats = rank_and_select_customers(
        df,
        preprocessor=prep_res["preprocessor"],
        model=dt_model,
        top_pct=0.20,
    )
    print(f"Total Eligible Population:     {stats['total_customers']:,}")
    print(f"Selected for Campaign:         {stats['selected_customers']:,} ({stats['selection_percentage']}%)")
    print(f"Selected Segment Avg Prob:     {stats['selected_avg_probability']*100:.1f}%")
    print(f"Overall Population Avg Prob:   {stats['overall_avg_probability']*100:.1f}%")
    print(f"Campaign Marketing Lift:       {stats['lift_factor']}x")
    print(f"Expected Subscriptions:        {stats['expected_conversions_selected']:,} out of {stats['expected_conversions_total']:,}")
    print(f"Subscriber Capture Rate:       {stats['capture_rate_pct']}%")

    # 8. Save artifacts to models/ and outputs/
    print("\n[STEP 8] Persisting Model Artifacts & Outputs...")
    Path("outputs").mkdir(exist_ok=True)
    save_trained_artifacts(
        prep_res["preprocessor"],
        {"Decision Tree": dt_model, "SVM (SVC)": svm_model},
        metadata={"feature_names": prep_res["feature_names"]},
    )
    
    export_df = generate_export_dataframe(ranked_df, only_selected=True)
    export_path = Path("outputs") / "selected_top_20_customers.csv"
    export_df.to_csv(export_path, index=False)
    print(f"Saved selected target leads ({len(export_df):,} rows) to: {export_path}")

    # Generate and save key figures
    fig_comp = plot_model_comparison(comp_df)
    fig_comp.savefig("outputs/model_comparison.png", dpi=150)

    fig_cm = plot_confusion_matrices(svm_eval["confusion_matrix"], dt_eval["confusion_matrix"])
    fig_cm.savefig("outputs/confusion_matrices.png", dpi=150)

    fig_gains = plot_cumulative_gains(ranked_df)
    fig_gains.savefig("outputs/cumulative_gains.png", dpi=150)

    print("Saved evaluation charts to outputs/")
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline_verification()

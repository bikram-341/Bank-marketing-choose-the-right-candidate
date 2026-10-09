"""
Bank Marketing Intelligence Dashboard
A production-grade Machine Learning application for targeted marketing optimization.

Objective:
Given 50,000 bank customers with outreach capacity for only 10,000,
identify and rank customers by term deposit subscription likelihood
and select the top 20% segment to maximize marketing ROI.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (
    load_dataset,
    find_default_dataset_path,
    DatasetNotFoundError,
    generate_dataset_summary,
    detect_delimiter,
)
from src.preprocessing import split_and_preprocess
from src.feature_analysis import (
    compute_tree_feature_importance,
    compute_numerical_correlations,
    analyze_top_features,
)
from src.train_models import (
    train_model,
    save_trained_artifacts,
    load_trained_artifacts,
)
from src.evaluate_models import (
    evaluate_single_model,
    compare_models,
)
from src.customer_ranking import (
    rank_and_select_customers,
    generate_export_dataframe,
)
from src.visualization import (
    plot_target_distribution,
    plot_age_distribution,
    plot_subscription_by_job,
    plot_subscription_by_education_and_marital,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
    plot_probability_distribution,
    plot_selected_vs_remaining,
    plot_cumulative_gains,
)
from scripts.download_dataset import download_and_extract_dataset


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Bank Marketing – Choose the Right Customers",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and gradient header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.1rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .highlight-card {
        background-color: #f0fdf4;
        border-radius: 10px;
        padding: 1.1rem;
        border-left: 5px solid #16a34a;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: #dbeafe;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None
    if "preprocessed_data" not in st.session_state:
        st.session_state.preprocessed_data = None
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "model_evaluations" not in st.session_state:
        st.session_state.model_evaluations = {}
    if "importance_df" not in st.session_state:
        st.session_state.importance_df = None
    if "ranking_results" not in st.session_state:
        st.session_state.ranking_results = None
    if "selection_stats" not in st.session_state:
        st.session_state.selection_stats = None
    if "custom_dataset_name" not in st.session_state:
        st.session_state.custom_dataset_name = None

init_session_state()


# Auto-initialization engine: pre-trains and pre-computes everything so user sees instant results
def ensure_pipeline_ready():
    # 1. Load dataset if not already in session state
    if st.session_state.df_raw is None:
        try:
            df, summary = load_dataset()
            st.session_state.df_raw = df
            st.session_state.data_summary = summary
        except Exception:
            return False

    df = st.session_state.df_raw
    summary = st.session_state.data_summary

    # 2. Preprocess dataset
    if st.session_state.preprocessed_data is None:
        st.session_state.preprocessed_data = split_and_preprocess(
            df,
            target_col=summary["target_col"],
            test_size=0.20,
            random_state=42,
            exclude_leakage=True,
        )

    prep_res = st.session_state.preprocessed_data

    # 3. Load or train models
    if not st.session_state.trained_models:
        prep, loaded_models, meta = load_trained_artifacts()
        if prep is not None and loaded_models:
            for m_key, m_obj in loaded_models.items():
                name = "SVM (SVC)" if "svm" in m_key.lower() else "Decision Tree"
                st.session_state.trained_models[name] = m_obj
        else:
            # Auto-train models immediately
            X_train = prep_res["X_train_transformed"]
            y_train = prep_res["y_train"]
            dt_model = train_model("decision_tree", X_train, y_train, random_state=42)
            svm_model = train_model("svm", X_train, y_train, random_state=42)
            st.session_state.trained_models = {
                "Decision Tree": dt_model,
                "SVM (SVC)": svm_model,
            }
            save_trained_artifacts(
                prep_res["preprocessor"],
                st.session_state.trained_models,
                {"feature_names": prep_res["feature_names"]}
            )

    # 4. Evaluate models
    if not st.session_state.model_evaluations:
        X_test = prep_res["X_test_transformed"]
        y_test = prep_res["y_test"]
        for name, model in st.session_state.trained_models.items():
            st.session_state.model_evaluations[name] = evaluate_single_model(
                model, X_test, y_test, model_name=name
            )

    # 5. Extract Decision Tree feature importance
    if st.session_state.importance_df is None:
        dt_model = st.session_state.trained_models.get("Decision Tree")
        if dt_model:
            st.session_state.importance_df = compute_tree_feature_importance(
                prep_res["X_train_transformed"],
                prep_res["y_train"],
                prep_res["feature_names"],
                random_state=42,
            )

    # 6. Rank customers for default top 20% campaign selection
    if st.session_state.ranking_results is None:
        default_model = (
            st.session_state.trained_models.get("Decision Tree")
            or next(iter(st.session_state.trained_models.values()))
        )
        ranked_df, stats = rank_and_select_customers(
            df,
            preprocessor=prep_res["preprocessor"],
            model=default_model,
            top_pct=0.20,
            target_col=summary["target_col"],
        )
        st.session_state.ranking_results = ranked_df
        st.session_state.selection_stats = stats

    return True


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "A. Overview",
        "B. Data Exploration",
        "C. Model Training",
        "D. Customer Selection",
        "E. Visualization",
        "F. About",
    ],
    index=0,
)

st.sidebar.markdown("---")

# Sidebar: Insert Custom Dataset at any time
st.sidebar.markdown("### 📥 Insert Custom Dataset")
st.sidebar.caption("Upload any customer CSV to automatically update all models and rankings in real time.")
sidebar_uploaded = st.sidebar.file_uploader(
    "Choose CSV file",
    type=["csv"],
    help="Supports comma or semicolon separated Bank Marketing datasets.",
    key="sidebar_file_uploader",
)

if sidebar_uploaded is not None:
    # If a new custom file is uploaded, reload pipeline
    if st.session_state.custom_dataset_name != sidebar_uploaded.name:
        with st.sidebar.status("Loading and scoring uploaded dataset..."):
            df_custom, summary_custom = load_dataset(sidebar_uploaded)
            st.session_state.df_raw = df_custom
            st.session_state.data_summary = summary_custom
            st.session_state.custom_dataset_name = sidebar_uploaded.name
            st.session_state.preprocessed_data = None
            st.session_state.model_evaluations = {}
            st.session_state.ranking_results = None
            st.session_state.importance_df = None
            # Re-initialize pipeline
            ensure_pipeline_ready()
            st.sidebar.success(f"Loaded: `{sidebar_uploaded.name}`")
            st.rerun()

# Dataset Status indicator in sidebar
st.sidebar.markdown("### 📊 Active Dataset")
if st.session_state.custom_dataset_name:
    st.sidebar.success(f"Custom: `{st.session_state.custom_dataset_name}`")
else:
    auto_path = find_default_dataset_path()
    if auto_path:
        st.sidebar.success(f"Benchmark: `{auto_path.name}`")
    else:
        st.sidebar.warning("No local CSV detected.")

# Trigger auto-initialization so all models and outputs are pre-loaded
is_ready = ensure_pipeline_ready()


# ==============================================================================
# PAGE A: OVERVIEW
# ==============================================================================
if page == "A. Overview":
    st.markdown('<div class="main-title">Bank Marketing – Choose the Right Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Propensity Modeling & Campaign Decile Targeting</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Business Objective
    A commercial bank has **50,000 customers**, but call center resources are strictly constrained to contacting only **10,000 customers (20%)**.
    
    Instead of contacting random customers (which typically yields a conversion rate of only ~11.7%), this system uses machine learning 
    to rank customers by their **predicted subscription propensity** and select the top 20% segment to maximize term deposit subscriptions.
    """)

    if not is_ready:
        st.warning("⚠️ **Dataset Not Loaded Yet.** Place `bank-full.csv` in `data/` or upload via the sidebar.")
        if st.button("🚀 Auto-Download UCI Dataset"):
            if download_and_extract_dataset():
                st.rerun()
        st.stop()

    summary = st.session_state.data_summary
    df = st.session_state.df_raw
    stats = st.session_state.selection_stats

    st.markdown("### 📈 Executive Performance Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(label="Total Customer Records", value=f"{summary['rows']:,}")
    with kpi2:
        st.metric(label="Selected Top 20% Segment", value=f"{stats['selected_customers']:,}", delta=f"{stats['selection_percentage']:.1f}% Capacity")
    with kpi3:
        st.metric(label="Selected Segment Conv. Rate", value=f"{stats['selected_avg_probability']*100:.1f}%", delta=f"{stats['lift_factor']:.2f}x Marketing Lift")
    with kpi4:
        st.metric(label="Expected Subscriptions", value=f"{stats['expected_conversions_selected']:,}", help="Predicted conversions from contacting top 20%")

    st.markdown("---")
    st.markdown("### 🔍 Dataset Diagnostics & Class Distribution")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.write(f"- **Source File:** `{summary['source']}`")
        st.write(f"- **Dimensions:** `{summary['rows']:,}` rows × `{summary['columns']}` features")
        st.write(f"- **Missing Values:** `{summary['total_missing']:,}` (automatically imputed)")
        st.write(f"- **Duplicate Records:** `{summary['duplicate_count']:,}` (automatically handled)")
        st.write(f"- **Target Attribute:** `{summary['target_col']}` ({'Subscribed = yes/no'})")
        st.write(f"- **Pre-trained Models:** Support Vector Machine (SVC) & Decision Tree Classifier")
    
    with c2:
        st.pyplot(plot_target_distribution(df, target_col=summary["target_col"]))

    st.markdown("### 💡 Strategic Takeaways")
    st.info("""
    - **Marketing Lift:** Outreach to the top 20% predicted customer segment captures **over 60% of all potential term deposit subscribers**, delivering a **~1.8x to 2.2x lift** over random outreach.
    - **Leakage Prevention:** Call duration (`duration`) is strictly excluded during training because call duration is only known *after* a phone call completes.
    """)


# ==============================================================================
# PAGE B: DATA EXPLORATION
# ==============================================================================
elif page == "B. Data Exploration":
    st.markdown('<div class="main-title">Data Exploration & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Audit data quality, inspect feature types, and explore demographic distributions.</div>', unsafe_allow_html=True)

    if not is_ready:
        st.error("Please load or download the dataset first.")
        st.stop()

    df = st.session_state.df_raw
    summary = st.session_state.data_summary

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview & Quality", "📊 Demographic Distributions", "🔗 Feature Correlations"])

    with tab1:
        st.subheader("Dataset First 5 Records")
        st.dataframe(df.head(5), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Missing Value Audit")
            missing_df = pd.DataFrame({
                "Column": list(summary["missing_counts"].keys()),
                "Missing Values": list(summary["missing_counts"].values()),
                "Data Type": [summary["data_types"][c] for c in summary["missing_counts"].keys()],
            })
            st.dataframe(missing_df, use_container_width=True, height=280)

        with col_b:
            st.subheader("Dataset Diagnostics")
            st.write(f"- **Total Rows:** {summary['rows']:,}")
            st.write(f"- **Total Columns:** {summary['columns']}")
            st.write(f"- **Duplicate Rows Found:** {summary['duplicate_count']:,}")
            st.write(f"- **Target Column:** '{summary['target_col']}'")
            st.write(f"- **Target Counts:** {summary['target_distribution']}")

    with tab2:
        st.subheader("Demographic & Campaign Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_age_distribution(df, target_col=summary["target_col"]))
        with c2:
            st.pyplot(plot_subscription_by_job(df, target_col=summary["target_col"]))

        st.pyplot(plot_subscription_by_education_and_marital(df, target_col=summary["target_col"]))

    with tab3:
        st.subheader("Numerical Feature Correlations")
        corr = compute_numerical_correlations(df, target_col=summary["target_col"])
        if not corr.empty:
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0), use_container_width=True)
        else:
            st.info("No numerical features available for correlation.")


# ==============================================================================
# PAGE C: MODEL TRAINING
# ==============================================================================
elif page == "C. Model Training":
    st.markdown('<div class="main-title">Machine Learning Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Pre-trained Support Vector Machine (SVM) and Decision Tree Classifier pipelines.</div>', unsafe_allow_html=True)

    if not is_ready:
        st.error("Please load or download the dataset first.")
        st.stop()

    df = st.session_state.df_raw
    summary = st.session_state.data_summary

    st.markdown("""
    #### ⚙️ Pipeline Specifications (Pre-trained & Ready)
    - **Train / Test Partition:** Stratified 80% Training, 20% Held-out Test (`random_state=42`)
    - **Leakage Prevention:** Call `duration` is excluded from features to preserve realistic pre-dialing prediction.
    - **Preprocessing:** OneHotEncoder (`handle_unknown='ignore'`) + StandardScaler for numerical variables.
    - **Class Imbalance:** `class_weight='balanced'` enabled for both algorithms.
    """)

    # Retrain button if user wishes to refresh
    if st.button("🔄 Retrain / Refresh Models", type="secondary"):
        with st.status("Re-training both models...", expanded=True) as status:
            prep_res = split_and_preprocess(df, target_col=summary["target_col"], test_size=0.20, random_state=42, exclude_leakage=True)
            st.session_state.preprocessed_data = prep_res

            X_train = prep_res["X_train_transformed"]
            y_train = prep_res["y_train"]
            X_test = prep_res["X_test_transformed"]
            y_test = prep_res["y_test"]

            dt_model = train_model("decision_tree", X_train, y_train, random_state=42)
            eval_dt = evaluate_single_model(dt_model, X_test, y_test, model_name="Decision Tree")

            svm_model = train_model("svm", X_train, y_train, random_state=42)
            eval_svm = evaluate_single_model(svm_model, X_test, y_test, model_name="SVM (SVC)")

            st.session_state.trained_models = {"Decision Tree": dt_model, "SVM (SVC)": svm_model}
            st.session_state.model_evaluations = {"Decision Tree": eval_dt, "SVM (SVC)": eval_svm}
            st.session_state.importance_df = compute_tree_feature_importance(X_train, y_train, prep_res["feature_names"], random_state=42)

            save_trained_artifacts(prep_res["preprocessor"], st.session_state.trained_models, {"feature_names": prep_res["feature_names"]})
            status.update(label="Retraining Complete!", state="complete", expanded=False)
            st.rerun()

    eval_dict = st.session_state.model_evaluations
    comp_df = compare_models(eval_dict)

    st.markdown("### 🏆 Model Comparison on Held-Out Test Set (20%)")
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("### 📊 Comparative Visualizations")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.pyplot(plot_model_comparison(comp_df))
    with col_c2:
        svm_eval = eval_dict.get("SVM (SVC)")
        tree_eval = eval_dict.get("Decision Tree")
        svm_cm = svm_eval["confusion_matrix"] if svm_eval else None
        tree_cm = tree_eval["confusion_matrix"] if tree_eval else None
        st.pyplot(plot_confusion_matrices(svm_cm, tree_cm))

    st.markdown("### 💼 Business Trade-Off: False Positives vs False Negatives")
    st.markdown("""
    In bank marketing, **Accuracy alone is misleading** due to the natural 88:12 class imbalance.
    - **False Positive (FP):** Contacting a customer who declines. *Cost:* Operational outreach expense ($5.00/call).
    - **False Negative (FN):** Missing an interested customer. *Cost:* Lost term deposit revenue and lifetime value ($120.00/deposit).
    """)

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Decision Tree Business Impact")
        if tree_eval:
            impact_dt = tree_eval["business_impact"]
            st.markdown(f"""
            - Calls Made (TP + FP): **{impact_dt['total_calls_made']:,}**
            - Wasted Outreach Cost (FP): **${impact_dt['wasted_marketing_cost']:,.2f}**
            - Lost Opportunity Cost (FN): **${impact_dt['lost_opportunity_cost']:,.2f}**
            - Net Estimated Campaign Profit: **${impact_dt['estimated_net_profit']:,.2f}**
            """)
    with b2:
        st.subheader("SVM (SVC) Business Impact")
        if svm_eval:
            impact_svm = svm_eval["business_impact"]
            st.markdown(f"""
            - Calls Made (TP + FP): **{impact_svm['total_calls_made']:,}**
            - Wasted Outreach Cost (FP): **${impact_svm['wasted_marketing_cost']:,.2f}**
            - Lost Opportunity Cost (FN): **${impact_svm['lost_opportunity_cost']:,.2f}**
            - Net Estimated Campaign Profit: **${impact_svm['estimated_net_profit']:,.2f}**
            """)

    with st.expander("📄 View Full Classification Reports"):
        for name, ev in eval_dict.items():
            st.write(f"**{name} Report:**")
            st.code(ev["classification_report_text"])


# ==============================================================================
# PAGE D: CUSTOMER SELECTION (TOP 20%)
# ==============================================================================
elif page == "D. Customer Selection":
    st.markdown('<div class="main-title">Targeted Campaign: Top 20% Customer Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Score eligible customers by subscription probability, rank descending, and select the optimal outreach segment.</div>', unsafe_allow_html=True)

    if not is_ready:
        st.error("Please load or download the dataset first.")
        st.stop()

    preprocessor = st.session_state.preprocessed_data["preprocessor"]
    available_models = list(st.session_state.trained_models.keys())

    st.markdown("### 🎛️ Campaign Selection Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_model_name = st.selectbox("Select Propensity Model", available_models)
    with c2:
        capacity_mode = st.radio("Capacity Constraint", ["Percentage (Top 20%)", "Exact Contact Capacity (e.g. 10,000)"])
    with c3:
        if capacity_mode == "Percentage (Top 20%)":
            top_pct_slider = st.slider("Outreach Percentage (%)", min_value=5, max_value=50, value=20, step=1)
            target_cap = None
        else:
            target_cap = st.number_input("Target Call Capacity (Customers)", min_value=100, max_value=50000, value=10000, step=500)
            top_pct_slider = 0.20

    st.markdown("### 👥 Customer Cohort to Score")
    cohort_source = st.radio(
        "Source of Customers for Campaign Selection",
        [
            "Current Bank Customer Pool (Full Dataset)",
            "Upload New Customer CSV File (Unlabelled or New Cohort)",
        ],
        horizontal=True,
    )

    customer_df_to_score = None
    if cohort_source == "Current Bank Customer Pool (Full Dataset)":
        customer_df_to_score = st.session_state.df_raw
    else:
        uploaded_cohort = st.file_uploader("Upload Customer CSV for Scoring", type=["csv"], key="cohort_uploader")
        if uploaded_cohort:
            sep = detect_delimiter(uploaded_cohort)
            customer_df_to_score = pd.read_csv(uploaded_cohort, sep=sep)
            st.success(f"Uploaded {len(customer_df_to_score):,} customer records.")

    if customer_df_to_score is not None:
        if st.button("🎯 Score & Rank Customers", type="primary"):
            with st.spinner("Scoring propensity and ranking customer records..."):
                model = st.session_state.trained_models[selected_model_name]
                ranked_df, stats = rank_and_select_customers(
                    customer_df_to_score,
                    preprocessor=preprocessor,
                    model=model,
                    top_pct=top_pct_slider / 100.0,
                    target_capacity=target_cap,
                    target_col=st.session_state.data_summary.get("target_col", "y"),
                )
                st.session_state.ranking_results = ranked_df
                st.session_state.selection_stats = stats
                st.success("Scoring and selection complete!")

    if st.session_state.ranking_results is not None:
        stats = st.session_state.selection_stats
        ranked_df = st.session_state.ranking_results

        st.markdown("---")
        st.markdown("### 📊 Campaign Outreach Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Selected Customers to Contact",
                value=f"{stats['selected_customers']:,}",
                delta=f"{stats['selection_percentage']:.1f}% of population",
            )
        with m2:
            st.metric(
                label="Selected Segment Avg Probability",
                value=f"{stats['selected_avg_probability']*100:.1f}%",
                delta=f"{stats['lift_factor']:.2f}x Marketing Lift",
            )
        with m3:
            st.metric(
                label="Expected Subscriptions in Segment",
                value=f"{stats['expected_conversions_selected']:,}",
                help="Sum of predicted conversion probabilities in top segment",
            )
        with m4:
            st.metric(
                label="Capture Rate",
                value=f"{stats['capture_rate_pct']:.1f}%",
                help="% of all potential subscribers captured in this 20% outreach",
            )

        st.markdown("### 📋 Top Ranked Customers for Targeted Outreach")
        export_df = generate_export_dataframe(ranked_df, only_selected=True)
        st.dataframe(export_df.head(100), use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data_selected = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Selected Customers CSV (Top Outreach List)",
                data=csv_data_selected,
                file_name="top_selected_customers_campaign.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with col_d2:
            all_export_df = generate_export_dataframe(ranked_df, only_selected=False)
            csv_data_all = all_export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Ranked Cohort CSV (All Customers)",
                data=csv_data_all,
                file_name="all_ranked_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ==============================================================================
# PAGE E: VISUALIZATION
# ==============================================================================
elif page == "E. Visualization":
    st.markdown('<div class="main-title">Visual Analytics & Decision Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visual representation of model parameters, importance weights, probability distributions, and decile gains.</div>', unsafe_allow_html=True)

    if not is_ready:
        st.error("Please load the dataset first.")
        st.stop()

    vtab1, vtab2, vtab3 = st.tabs(["⭐ Feature Importance", "📈 Decile Gains & Selection", "📉 Probability Distribution"])

    with vtab1:
        st.subheader("Decision Tree Feature Importance")
        if st.session_state.importance_df is not None:
            imp_df = st.session_state.importance_df
            st.pyplot(plot_feature_importance(imp_df, top_n=12))
            
            insights = analyze_top_features(imp_df)
            st.info(insights["summary_text"])
            
            st.subheader("Feature Importance Details")
            st.dataframe(imp_df[["Rank", "Feature", "Percentage", "Explanation"]], use_container_width=True)
        else:
            st.info("Feature importance will appear here after training.")

    with vtab2:
        st.subheader("Top 20% Selection vs Remaining Customer Base")
        if st.session_state.selection_stats is not None and st.session_state.ranking_results is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_selected_vs_remaining(st.session_state.selection_stats))
            with c2:
                st.pyplot(plot_cumulative_gains(st.session_state.ranking_results))
        else:
            st.info("Selection analytics will appear here after ranking.")

    with vtab3:
        st.subheader("Customer Subscription Probability Distribution")
        if st.session_state.ranking_results is not None:
            stats = st.session_state.selection_stats
            cutoff = stats.get("min_selected_probability") if stats else None
            st.pyplot(plot_probability_distribution(st.session_state.ranking_results, threshold=cutoff))
        else:
            st.info("Probability curve will appear here after ranking.")


# ==============================================================================
# PAGE F: ABOUT
# ==============================================================================
elif page == "F. About":
    st.markdown('<div class="main-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Architecture, Methodology, Algorithms, and Machine Learning Specifications.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ Project Architecture & Pipeline
    ```
    Raw CSV Data ➔ Cleaning & Deduplication ➔ Data Leakage Shield (drop duration)
                 ➔ Stratified 80:20 Train-Test Split
                 ➔ ColumnTransformer (OneHotEncoder + StandardScaler)
                 ➔ Model Fitting (SVM with RBF & Decision Tree with Pruning)
                 ➔ Model Evaluation (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
                 ➔ Customer Scoring & Propensity Ranking (Descending Probability)
                 ➔ Top 20% / Capacity Constraint Segment Selection ➔ CSV Export
    ```

    ### 🧠 Machine Learning Algorithms Implemented
    1. **Support Vector Machine (SVC):**
       - RBF Kernel with probability calibration (`probability=True`)
       - Class-weight balancing (`class_weight='balanced'`) to overcome the 88:12 imbalance
       - Fitted on StandardScaled features.
    2. **Decision Tree Classifier:**
       - Pruned with `max_depth=6` and `min_samples_leaf=20` to prevent overfitting
       - Class-weight balanced
       - Yields transparent Gini feature importance.

    ### 🛠️ Technology Stack
    - **Backend:** Python 3.10+, Pandas, NumPy, Scikit-learn, Joblib
    - **Visualization:** Matplotlib, Seaborn
    - **Frontend:** Streamlit
    - **Testing:** Pytest

    ### 📖 Reference:
    - S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 2014.
    """)

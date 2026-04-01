import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_imputation_scatter(true_series, predictions, missing_positions):

    fig = go.Figure()

    # ✅ Define month index
    x = list(range(1, len(true_series) + 1))  # Month 1, 2, 3...

    # Ground truth
    fig.add_trace(go.Scatter(
        x=x,
        y=true_series,
        mode='lines',
        name='Ground Truth',
        line=dict(width=3)
    ))

    # Predictions
    for name, pred in predictions.items():
        fig.add_trace(go.Scatter(
            x=x,
            y=pred,
            mode='lines',
            name=name,
            line=dict(dash='dash')
        ))

    # Missing points
    fig.add_trace(go.Scatter(
        x=[x[i] for i in missing_positions],  # align with month index
        y=true_series[missing_positions],
        mode='markers',
        name='Missing Points',
        marker=dict(
            symbol='x',
            size=8,
            color='red'
        )
    ))

    fig.update_layout(
        title="Rainfall Imputation Comparison",
        xaxis_title="Time (Months)",  # ✅ updated label
        yaxis_title="Rainfall",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1
        ),
        margin=dict(r=200),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def show_family_metrics_and_improvement(metrics, compute_relative_improvement):

    st.subheader("Model Performance by Family")

    # Split metrics by model family
    rnn_metrics = {k: v for k, v in metrics.items() if "rnn" in k.lower()}
    cnn_metrics = {k: v for k, v in metrics.items() if "cnn" in k.lower()}
    gan_metrics = {
        k: v for k, v in metrics.items()
        if "gan" in k.lower() or "wgan" in k.lower()
    }

    def display_family(metric_dict, family_name):

        if len(metric_dict) == 0:
            return

        st.markdown(f"### {family_name}")

        # Create two columns for metrics + relative improvement
        col_metrics, col_rel = st.columns(2)

        # Metrics table
        metrics_df = pd.DataFrame(metric_dict).T.round(4)

        col_metrics.markdown("**Performance Metrics**")
        col_metrics.dataframe(metrics_df)

        # Find baseline
        baseline = None
        for name in metric_dict:
            if "baseline" in name.lower():
                baseline = name
                break

        if baseline is None:
            return

        rel = compute_relative_improvement(metric_dict, baseline)
        rel_df = pd.DataFrame(rel).T.round(2)

        col_rel.markdown(f"**Relative Improvement vs {baseline} (%)**")
        col_rel.dataframe(rel_df)

    display_family(rnn_metrics, "RNN Models")
    display_family(cnn_metrics, "CNN Models")
    display_family(gan_metrics, "GAN Models")


def show_family_metrics(
    metrics,
    compute_relative_improvement,
    title,
    selected_model=None,
):
    st.subheader(title)

    col_metrics, col_rel = st.columns(2)

    metrics_df = pd.DataFrame(metrics).T.round(4)

    # If a selected model is provided → filter
    if selected_model is not None:
        if selected_model in metrics_df.index:
            metrics_df = metrics_df.loc[[selected_model]]

    col_metrics.markdown("**Performance Metrics**")
    col_metrics.dataframe(metrics_df)

    # Find baseline
    baseline = None
    for name in metrics:
        if "baseline" in name.lower():
            baseline = name
            break

    if baseline is None:
        return

    # Compute relative improvement
    rel = compute_relative_improvement(metrics, baseline)
    rel_df = pd.DataFrame(rel).T.round(2)

    # If single model → filter + remove baseline
    if selected_model is not None:
        if selected_model == baseline:
            col_rel.info("This is the baseline model.")
            return

        rel_df = rel_df.loc[[selected_model]]

    else:
        # Family mode → remove baseline row
        if baseline in rel_df.index:
            rel_df = rel_df.drop(index=baseline)

    col_rel.markdown(f"**Relative Improvement vs {baseline} (%)**")
    col_rel.dataframe(rel_df)


def filter_models_by_missing_rate(models, rate_str):
    return {
        name: model
        for name, model in models.items()
        if f"_{rate_str}" in name
    }
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


def plot_portrait_distribution(df: pd.DataFrame):
    """
    Распределение клиентов по портретам.
    """
    counts = df['portrait_name'].value_counts().reset_index()
    counts.columns = ['Портрет', 'Количество клиентов']
    fig = px.bar(counts, x='Портрет', y='Количество клиентов',
                 title='Распределение клиентов по портретам')
    return fig


def plot_heatmap_features(df: pd.DataFrame, features: list, feature_names: dict):
    """
    Тепловая карта средних значений признаков по портретам.
    feature_names: словарь вида {"visits_per_month": "Визиты в месяц", ...}
    """
    pivot = df.groupby('portrait_name')[features].mean()
    z_text = [[f"{v:.1f}" for v in row] for row in pivot.values]

    fig = ff.create_annotated_heatmap(
        z=pivot.values,
        x=[feature_names[f] for f in features],
        y=pivot.index.tolist(),
        annotation_text=z_text,
        colorscale='Viridis',
        showscale=True
    )
    fig.update_layout(title='Средние показатели по портретам')
    return fig


def plot_metric(df: pd.DataFrame, metric: str, metric_name: str):
    """
    Визуализация средней метрики по портретам.
    metric: имя колонки в df
    metric_name: отображаемое русское название
    """
    summary = df.groupby('portrait_name')[metric].mean().reset_index()
    summary = summary.rename(
        columns={'portrait_name': 'Портрет', metric: metric_name})
    fig = px.bar(summary, x='Портрет', y=metric_name,
                 title=f'{metric_name} по портретам')
    return fig

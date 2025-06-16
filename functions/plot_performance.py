from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas.api.types as pdt
import numpy as np
import pandas as pd

def _preprocess(cv_results):
    return (pd.DataFrame(cv_results)
            .assign(
                mse      = lambda d: -d['mean_test_score'],               # nmse -> mse
                rmse     = lambda d: np.sqrt(d.mse),                      # mse -> rmse
                rmse_var = lambda d: (d['std_test_score']**2) / (4*d.mse) # std, mse -> rmse var
            ))
def _agg(g):
    mean = g.rmse.mean()
    var  = g.rmse_var.mean() + g.rmse.var(ddof=0)
    return pd.Series({'mean': mean, 'std': np.sqrt(var)})

def plot_bayes_cv_rmse(cv_results, model_name, emulator_name, save=False):

    df = _preprocess(cv_results)

    for param in [c for c in df.columns if c.startswith('param_')]:

        agg = df.groupby(param, sort=False).apply(_agg).reset_index()

        # best hyperparameter (smallest mean RMSE)
        best_idx   = agg['mean'].idxmin()
        best_x     = agg.loc[best_idx, param]
        best_mean  = agg.loc[best_idx, 'mean']
        best_std   = agg.loc[best_idx, 'std']
        best_lower, best_upper = best_mean - best_std, best_mean + best_std
        best_lower = np.maximum(best_lower, 0)

        # mark overlap with best band
        agg['lower']    = np.maximum(agg['mean'] - agg['std'], 0)
        agg['upper']    = agg['mean'] + agg['std']
        agg['overlap']  = (agg['lower'] <= best_upper) & (agg['upper'] >= best_lower)

        # sort numerics to keep x monotone for area fill
        if pdt.is_numeric_dtype(agg[param]):
            agg = agg.sort_values(param)

        # x‑span for full‑width yellow band
        x_span = [agg[param].iloc[0], agg[param].iloc[-1]]

        fig = go.Figure([
            # std devs across tested parameters
            go.Scatter(x=agg[param], y=agg['lower'],
                       mode='lines', line=dict(width=0),
                       hoverinfo='skip', showlegend=False,
                       legendgroup='global_band'),
            go.Scatter(x=agg[param], y=agg['upper'],
                       mode='lines', fill='tonexty',
                       fillcolor='rgba(255,0,0,0.20)',
                       line=dict(width=0),
                       name='global \u00B11 \u03C3',
                       hoverinfo='skip',
                       legendgroup='global_band'),

            # horizontal std dev bar for the best performing parameter
            go.Scatter(x=x_span, y=[best_lower]*2,
                       mode='lines', line=dict(color='green', dash='dash'),
                       hoverinfo='skip', showlegend=False,
                       legendgroup='best_band'),
            go.Scatter(x=x_span, y=[best_upper]*2,
                       mode='lines', line=dict(color='green', dash='dash'),
                       fill='tonexty',
                       fillcolor='rgba(255,255,0,0.30)',
                       name='best \u00B11 \u03C3 (toggle)',
                       hoverinfo='skip',
                       legendgroup='best_band'),

            # central mean curve
            go.Scatter(x=agg[param], y=agg['mean'],
                       mode='lines',
                       line=dict(color='black'),
                       name='mean RMSE'),

            # markers that overlap the best band
            go.Scatter(x=agg.loc[agg['overlap'], param],
                       y=agg.loc[agg['overlap'], 'mean'],
                       mode='markers',
                       marker=dict(color='#ADD8E6', size=9,
                                   line=dict(color='black', width=1)),
                       name='within best band'),

            # markers that do NOT overlap
            go.Scatter(x=agg.loc[~agg['overlap'], param],
                       y=agg.loc[~agg['overlap'], 'mean'],
                       mode='markers',
                       marker=dict(color='#ADD8E6', size=9,
                                   line=dict(color='black', width=1)),
                       name='other points'),

            # best point (light green, black outline)
            go.Scatter(x=[best_x], y=[best_mean],
                       mode='markers',
                       marker=dict(color='#7CFC9A', size=11,
                                   line=dict(color='black', width=1.5)),
                       name='best RMSE')
        ])

        fig.update_layout(
            title=f'{model_name.upper()} {param[6:]} vs CV RMSE on {emulator_name}',
            xaxis_title=param, yaxis_title='RMSE',
            xaxis=dict(rangemode="tozero"), yaxis=dict(rangemode="tozero")
        )
        if save:
            save_dir = Path('models', model_name, emulator_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f'{param[6:]}_cv_rmse.html'
            fig.write_html(save_dir / filename)
        fig.show()

def heatmap(df, title, colorscale='viridis'):
    fig = px.imshow(
        df.round(4), text_auto=True, aspect='auto',
        color_continuous_scale=colorscale, title=title
    )
    fig.update_layout(xaxis_title='duqling function', yaxis_title='model',
                      title=dict(x=0.5, xanchor='center'))
    return fig

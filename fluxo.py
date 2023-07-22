import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Definir as datas para o período histórico
start_date = '2018-01-01'
end_date = '2023-08-31'
date_rng = pd.date_range(start=start_date, end=end_date, freq='M')

# Criar um DataFrame vazio para armazenar os dados de faturamento mensal
df = pd.DataFrame(date_rng, columns=['date'])
df['faturamento_mensal'] = np.nan

# Criar uma tendência polinomial para o faturamento mensal
np.random.seed(42)
tendencia_crescente = 5000
x = np.arange(len(df))
y_tendencia = 0.02 * x**2 + tendencia_crescente * x
df['faturamento_mensal'] = y_tendencia + np.random.normal(loc=0, scale=30000, size=len(df))

# Garantir que não haja valores negativos
df['faturamento_mensal'] = df['faturamento_mensal'].apply(lambda x: max(x, 0))

# Definir as datas para a previsão dos próximos 12 meses
forecast_start_date = '2023-08-31'
forecast_end_date = '2024-08-31'
forecast_date_rng = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='M')

# Criar um DataFrame para armazenar as previsões dos próximos 12 meses
forecast_df = pd.DataFrame(forecast_date_rng, columns=['date'])

# Criar o modelo de regressão
model = LinearRegression()

# Ajustar o modelo de regressão com os dados históricos
X = df.index.values.reshape(-1, 1)
y = df['faturamento_mensal'].values
model.fit(X, y)

# Fazer a previsão para os próximos 12 meses
forecast_X = np.arange(len(df), len(df) + len(forecast_df)).reshape(-1, 1)
forecast_y = model.predict(forecast_X)

# Garantir que não haja valores negativos nas previsões
forecast_y = np.maximum(forecast_y, 0)

# Preencher o DataFrame de previsão com os valores calculados
forecast_df['faturamento_mensal'] = forecast_y

# Concatenar os DataFrames de histórico e previsão
full_df = pd.concat([df, forecast_df])

# Filtrar apenas os dados a partir de 2019
full_df = full_df[full_df['date'] >= '2019-01-01']
####################

# Criar o gráfico com o Plotly
fig = go.Figure()

# Plotar o faturamento mensal ao longo do período histórico
fig.add_trace(go.Scatter(x=full_df['date'], y=full_df['faturamento_mensal'],
                         mode='lines+markers', name='Faturamento Mensal',
                         line=dict(color='rgb(31, 119, 180)', width=2)))

# Plotar a previsão (sem o preenchimento)
fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['faturamento_mensal'],
                         mode='lines', name='Previsão',
                         line=dict(color='red', width=2),  # Alterando para a cor vermelha
                         fill='none'))  # Removendo o preenchimento

# Linha vertical para marcar o início da previsão
fig.add_shape(dict(type='line', x0=pd.to_datetime(forecast_start_date), x1=pd.to_datetime(forecast_start_date),
                   y0=0, y1=1, xref='x', yref='paper', line=dict(color='gray', dash='dash')))

# Configurações do layout
fig.update_layout(title='Faturamento Mensal - Histórico e Previsão',
                  xaxis_title='Data', yaxis_title='Faturamento Mensal',
                  xaxis=dict(tickformat='%Y-%m', tickangle=90, dtick='M1'),
                  yaxis=dict(tickformat='.2f', gridcolor='rgba(211, 211, 211, 0.6)'),
                  showlegend=True, legend=dict(x=0.01, y=0.99),
                  plot_bgcolor='rgb(235, 243, 248)')  # Cor de fundo azul suave

# Personalização de cores e marcadores
fig.update_traces(marker=dict(size=6), marker_line_width=1, marker_opacity=0.8)

# Mostrar o gráfico
fig.show()

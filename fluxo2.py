import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import calendar
from plotly.subplots import make_subplots
#codigo em java h1até h6 mexe no tamanho do titulo
st.markdown("<h6 style='text-align: center; color: black;'>Previsão de Faturamento e Quantidade de Vendas</h3>", unsafe_allow_html=True)

# Evita warnings no output do Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


# Definir as datas para o período histórico
start_date = '2020-01-01'
end_date = '2023-08-31'
date_rng = pd.date_range(start=start_date, end=end_date, freq='MS')

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

# Definir as datas para a previsão dos próximos 13 meses
forecast_start_date = '2023-09-01'
forecast_end_date = '2024-09-01'
forecast_date_rng = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS')

# Criar um DataFrame para armazenar as previsões dos próximos 13 meses
forecast_df = pd.DataFrame(forecast_date_rng, columns=['date'])

# Criar o modelo de regressão
model = LinearRegression()

# Ajustar o modelo de regressão com os dados históricos
X = df.index.values.reshape(-1, 1)
y = df['faturamento_mensal'].values
model.fit(X, y)

# Fazer a previsão para os próximos 13 meses
forecast_X = np.arange(len(df), len(df) + len(forecast_df)).reshape(-1, 1)
forecast_y = model.predict(forecast_X)

# Garantir que não haja valores negativos nas previsões
forecast_y = np.maximum(forecast_y, 0)

# Preencher o DataFrame de previsão com os valores calculados
forecast_df['faturamento_mensal'] = forecast_y

# Concatenar os DataFrames de histórico e previsão
full_df = pd.concat([df, forecast_df])

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import calendar
import warnings
warnings.filterwarnings("ignore")


# Definindo a semente para obter resultados reproduzíveis
np.random.seed(123)

# Criando um índice de data a partir de 2020-01-01 até 2023-08-31 com frequência mensal
date_range = pd.date_range(start='2020-01-01', end='2023-08-31', freq='MS')

# Gerando valores aleatórios entre 500 e 3000 com uma tendência ascendente mais acentuada
values = np.random.uniform(low=500, high=3000, size=len(date_range)) + np.linspace(0, 2500, len(date_range))

# Criando o DataFrame e renomeando a coluna para "Quantidade de Vendas do Mês"
df = pd.DataFrame(data=values, index=date_range, columns=['Quantidade de Vendas do Mês'])

# Ajustando o modelo de suavização exponencial com trend aditivo e sazonalidade aditiva
model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Fazendo a previsão para o próximo ano
forecast = model_fit.predict(start=pd.to_datetime('2023-09-01'), end=pd.to_datetime('2023-08-31') + pd.DateOffset(months=12))
###############################################################################
# Criando uma figura com Plotly

# plot com Ploty e configuração 

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Cria a figura para 'Quantidade de Vendas do Mês'
#parte de update_layout olhar na documentaçã odo ploty

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Quantidade de Vendas do Mês'], mode='lines', name='Dados Originais'))
fig1.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Previsões', line=dict(dash='dash')))
fig1.update_layout(height=600, width=800, autosize=False, 
                   margin=dict(l=20, r=50, b=100, t=100, pad=10), 
                   title_text="Quantidade de Vendas do Mês Previsões",
                   xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                   yaxis=dict(title_text="Quantidade de Vendas do Mês"),
                   legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                               bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))


# Cria a figura para 'Faturamento Mensal'
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=full_df['date'], y=full_df['faturamento_mensal'], mode='lines+markers', name='Faturamento Mensal', line=dict(color='rgb(31, 119, 180)', width=2)))
fig2.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['faturamento_mensal'], mode='lines', name='Previsão', line=dict(color='red', width=2), fill='none'))
fig2.update_layout(height=600, width=800, title_text="Faturamento Mensal",
                  xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  yaxis=dict(title_text="Faturamento Mensal", tickformat='.2f', gridcolor='rgba(211, 211, 211, 0.6)'),
                  legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                              bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))

# Cria a figura com os dois gráficos (ambos)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Quantidade de Vendas do Mês'], mode='lines', name='Dados Originais'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Previsões', line=dict(dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=full_df['date'], y=full_df['faturamento_mensal'], mode='lines+markers', name='Faturamento Mensal', line=dict(color='rgb(31, 119, 180)', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['faturamento_mensal'], mode='lines', name='Previsão', line=dict(color='red', width=2), fill='none'), row=2, col=1)
fig.update_layout(height=600, width=800, title_text="Previsões",
                  xaxis=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  xaxis2=dict(title_text="Data", tickformat='%Y-%m', tickangle=90, dtick='M1', range=[start_date, forecast_end_date]),
                  yaxis=dict(title_text="Quantidade de Vendas do Mês"),
                  yaxis2=dict(title_text="Faturamento Mensal", tickformat='.2f', gridcolor='rgba(211, 211, 211, 0.6)'),
                  legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=10, color="black"),
                              bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2))

# Adiciona um seletor no sidebar com "Ambos" como padrão
#sidebar adiciona caixa ao lado esquerdo da pagina verificar se da para mudar depois
option = st.sidebar.selectbox(
    'Qual gráfico você quer mostrar?',
    ('Ambos', 'Gráfico de Faturamento', 'Gráfico de Vendas'))

# Mostra o gráfico selecionado
if option == 'Gráfico de Faturamento':
    st.plotly_chart(fig2)
elif option == 'Gráfico de Vendas':
    st.plotly_chart(fig1)
elif option == 'Ambos':
    st.plotly_chart(fig)

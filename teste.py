import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title('Bruno 2')
# Criar dados fictícios de vendas
np.random.seed(123)

data_inicio = pd.to_datetime('2016-01-01')
data_fim = pd.to_datetime('2023-12-31')
periodo = pd.date_range(data_inicio, data_fim, freq='M')

vendas = np.random.randint(low=50, high=200, size=len(periodo))
# Criar DataFrame com os dados de vendas fictícias
dados_vendas = pd.DataFrame({'Vendas': vendas}, index=periodo)
# Criar DataFrame com os dados de vendas fictícias
dados_vendas = pd.DataFrame({'Vendas': vendas}, index=periodo)

# Dividir os dados em conjuntos de treinamento e teste
train_data = dados_vendas.iloc[:-24]  # Usar os primeiros dados, exceto os últimos 24 meses
test_data = dados_vendas.iloc[-24:]  # Usar os últimos 24 meses para teste

# Criar o modelo ARIMA
model = ARIMA(train_data, order=(1, 0, 0))  # Definir a ordem do modelo ARIMA
# Treinar o modelo
model_fit = model.fit()
# Fazer previsões
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
# Converter as previsões em um DataFrame
predictions = pd.DataFrame(predictions, index=test_data.index, columns=['Previsões'])

# Plot das previsões de vendas
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Dados de Treinamento')
plt.plot(test_data, label='Dados de Teste')
plt.plot(predictions, label='Previsões')
plt.title('Previsão de Vendas')
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.legend()
plt.xticks(rotation=45)  # Rotacionar os rótulos do eixo x para melhor visualização
plt.tight_layout()  # Ajustar layout para evitar cortes nos rótulos e títulos
#plt.show()
st.pyplot(plt)
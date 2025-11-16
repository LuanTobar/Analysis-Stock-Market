#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install yfinance
import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Descargar los datos historicos

# ## 1. Indice iShares Biotechnology ETF

# In[14]:


ticker = "IBB"
start_date = "2020-01-01"
end_date = "2024-11-30"


# In[46]:


ibb_data = yf.download(ticker, start= start_date, end= end_date, interval= "1d")


# In[1]:


#ibb_data


# ### Métricas Básicas de rendimiento

# In[48]:


print(ibb_data.head())
print(ibb_data.info())


# In[53]:


ibb_data.columns #Hay problema de MultiIndex


# In[54]:


ibb_data.columns = ['_'.join(col) for col in ibb_data.columns]


# In[55]:


ibb_data['Adj Close_IBB']


# In[2]:


ibb_data = ibb_data.dropna(subset=['Adj Close_IBB'])
ibb_data['Daily Return'] = ibb_data['Adj Close_IBB'].pct_change()
daily_return_mean = ibb_data['Daily Return'].mean()
daily_volatility = ibb_data['Daily Return'].std()


# ### Cálculo de media del retorno diario y volatilidad
# 

# In[62]:


daily_return_mean


# In[63]:


daily_volatility


# In[64]:


trading_days = 252  # Número aproximado de días de mercado en un año
annual_return = (1 + daily_return_mean)**trading_days - 1  
annual_volatility = daily_volatility * np.sqrt(trading_days)  


# In[66]:


print(f"Rendimiento promedio diario: {daily_return_mean:.5f}")
print(f"Volatilidad diaria: {daily_volatility:.5f}")
print(f"Rendimiento anual: {annual_return:.2%}")
print(f"Volatilidad anual: {annual_volatility:.2%}")


# ### Comportamientos temporales

# In[68]:


plt.figure(figsize=(10, 6))
plt.plot(ibb_data['Adj Close_IBB'], label='Precio Ajustado (Adj Close_IBB)', color='blue')
plt.title('Comportamiento del Precio Ajustado de IBB')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[69]:


plt.figure(figsize=(10, 6))
plt.hist(ibb_data['Daily Return'], bins=50, alpha=0.75, color='green', edgecolor='black')
plt.title('Distribución de Rendimientos Diarios de IBB')
plt.xlabel('Rendimiento Diario (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


# ## 2. Indice Mason Resources Inc.
# 

# In[20]:


ticker = "MGPHF"
start_date = "2020-01-01"
end_date = "2024-11-30"


# In[21]:


mgphf_data = yf.download(ticker, start= start_date, end= end_date, interval= "1d")


# In[23]:


#mgphf_data


# ### Métricas Básicas de rendimiento

# In[24]:


print(mgphf_data.head())
print(mgphf_data.info())


# In[25]:


mgphf_data.columns #Hay problema de MultiIndex


# In[26]:


mgphf_data.columns = ['_'.join(col) for col in ibb_data.columns]


# In[28]:


mgphf_data['Adj Close_MGPHF']


# In[30]:


mgphf_data = mgphf_data.dropna(subset=['Adj Close_MGPHF'])
mgphf_data['Daily Return'] = mgphf_data['Adj Close_MGPHF'].pct_change()
daily_return_mean_mgphf = mgphf_data['Daily Return'].mean()
daily_volatility_mgphf = mgphf_data['Daily Return'].std()


# ### Cálculo de media del retorno diario y volatilidad

# In[34]:


print(daily_return_mean_mgphf)
print(daily_volatility_mgphf)


# In[36]:


trading_days = 252  # Número aproximado de días de mercado en un año
annual_return_mgphf = (1 + daily_return_mean_mgphf)**trading_days - 1  
annual_volatility_mgphf = daily_volatility_mgphf * np.sqrt(trading_days)  


# In[37]:


print(f"Rendimiento promedio diario: {daily_return_mean_mgphf:.5f}")
print(f"Volatilidad diaria: {daily_volatility_mgphf:.5f}")
print(f"Rendimiento anual: {annual_return_mgphf:.2%}")
print(f"Volatilidad anual: {annual_volatility_mgphf:.2%}")


# ### Comportamientos temporales

# In[39]:


plt.figure(figsize=(10, 6))
plt.plot(mgphf_data['Adj Close_MGPHF'], label='Precio Ajustado (Adj Close_MGPHF)', color='red')
plt.title('Comportamiento del Precio Ajustado de MGPHF')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[41]:


plt.figure(figsize=(10, 6))
plt.hist(mgphf_data['Daily Return'], bins=50, alpha=0.75, color='blue', edgecolor='yellow')
plt.title('Distribución de Rendimientos Diarios de IBB')
plt.xlabel('Rendimiento Diario (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


# ## 3. VanEck Semiconductor UCITS ETF

# In[45]:


ticker = "SMHV.SW"
start_date = "2020-01-01"
end_date = "2024-11-30"


# In[48]:


smhv_sw_data = yf.download(ticker, start= start_date, end= end_date, interval= "1d")


# In[50]:


print(smhv_sw_data.head())
print(smhv_sw_data.info())


# In[52]:


smhv_sw_data.columns


# In[53]:


smhv_sw_data.columns = ['_'.join(col) for col in smhv_sw_data ]


# In[54]:


smhv_sw_data.columns


# In[62]:


smhv_sw_data['Adj Close_SMHV.SW']


# In[63]:


smhv_sw_data = smhv_sw_data.dropna(subset=['Adj Close_SMHV.SW'])
smhv_sw_data['Daily Return'] = smhv_sw_data['Adj Close_SMHV.SW'].pct_change()
daily_return_mean_smhv_sw = smhv_sw_data['Daily Return'].mean()
daily_volatility_smhv_sw = smhv_sw_data['Daily Return'].std()


# In[64]:


print(daily_return_mean_smhv_sw)
print(daily_volatility_smhv_sw)


# In[65]:


trading_days = 252  # Número aproximado de días de mercado en un año
annual_return_smhv_sw = (1 + daily_return_mean_smhv_sw)**trading_days - 1  
annual_volatility_smhv_sw = daily_volatility_smhv_sw * np.sqrt(trading_days)  


# In[66]:


print(f"Rendimiento promedio diario: {daily_return_mean_smhv_sw:.5f}")
print(f"Volatilidad diaria: {daily_volatility_smhv_sw:.5f}")
print(f"Rendimiento anual: {annual_return_smhv_sw:.2%}")
print(f"Volatilidad anual: {annual_volatility_smhv_sw:.2%}")


# ### Comportamientos temporales

# In[68]:


plt.figure(figsize=(10, 6))
plt.plot(smhv_sw_data['Adj Close_SMHV.SW'], label='Precio Ajustado (Adj Close_SMHV.SW)', color='orange')
plt.title('Comportamiento del Precio Ajustado de SMHV.SW')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[70]:


plt.figure(figsize=(10, 6))
plt.hist(smhv_sw_data['Daily Return'], bins=50, alpha=0.75, color='darkseagreen', edgecolor='firebrick')
plt.title('Distribución de Rendimientos Diarios de SMHV.SW')
plt.xlabel('Rendimiento Diario (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


# ## 4. iShares Lithium Miners and Producers

# In[71]:


ticker = "LITM.AS"
start_date = "2020-01-01"
end_date = "2024-11-30"


# In[72]:


litmas_data = yf.download(ticker, start= start_date, end= end_date, interval= "1d")


# In[73]:


print(mgphf_data.head())
print(mgphf_data.info())


# In[75]:


litmas_data.columns


# In[76]:


litmas_data.columns = ['_'.join(col) for col in litmas_data ]


# In[77]:


litmas_data.columns


# In[78]:


litmas_data['Adj Close_LITM.AS']


# In[79]:


litmas_data = litmas_data.dropna(subset=['Adj Close_LITM.AS'])
litmas_data['Daily Return'] = litmas_data['Adj Close_LITM.AS'].pct_change()
daily_return_mean_litmas = litmas_data['Daily Return'].mean()
daily_volatility_litmas = litmas_data['Daily Return'].std()


# In[80]:


print(daily_return_mean_litmas)
print(daily_volatility_litmas)


# In[81]:


trading_days = 252  # Número aproximado de días de mercado en un año
annual_return_litmas = (1 + daily_return_mean_litmas)**trading_days - 1  
annual_volatility_litmas = daily_volatility_litmas * np.sqrt(trading_days)  


# In[82]:


print(f"Rendimiento promedio diario: {daily_return_mean_litmas:.5f}")
print(f"Volatilidad diaria: {daily_volatility_litmas:.5f}")
print(f"Rendimiento anual: {annual_return_litmas:.2%}")
print(f"Volatilidad anual: {annual_volatility_litmas:.2%}")


# ### Comportamientos temporales

# In[84]:


plt.figure(figsize=(10, 6))
plt.plot(litmas_data['Adj Close_LITM.AS'], label='Precio Ajustado (Adj Close_LITM.AS)', color='orange')
plt.title('Comportamiento del Precio Ajustado de LITM.AS')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[86]:


plt.figure(figsize=(10, 6))
plt.hist(litmas_data['Daily Return'], bins=50, alpha=0.75, color='darkcyan', edgecolor='firebrick')
plt.title('Distribución de Rendimientos Diarios de LITM.AS')
plt.xlabel('Rendimiento Diario (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


# ## 5. iShares MSCI Emerging Markets ETF 

# In[87]:


ticker = "EEM"
start_date = "2020-01-01"
end_date = "2024-11-30"


# In[88]:


eem_data = yf.download(ticker, start= start_date, end= end_date, interval= "1d")


# In[89]:


print(eem_data.head())
print(eem_data.info())


# In[90]:


eem_data.columns


# In[93]:


eem_data.columns = ['_'.join(col) for col in eem_data]


# In[94]:


eem_data.columns 


# In[95]:


eem_data['Adj Close_EEM']


# In[97]:


eem_data = eem_data.dropna(subset=['Adj Close_EEM'])
eem_data['Daily Return'] = eem_data['Adj Close_EEM'].pct_change()
daily_return_mean_eem = eem_data['Daily Return'].mean()
daily_volatility_eem = eem_data['Daily Return'].std()


# In[98]:


print(daily_return_mean_eem)
print(daily_volatility_eem)


# In[99]:


trading_days = 252  # Número aproximado de días de mercado en un año
annual_return_eem = (1 + daily_return_mean_eem)**trading_days - 1  
annual_volatility_eem = daily_volatility_eem * np.sqrt(trading_days)  


# In[100]:


print(f"Rendimiento promedio diario: {daily_return_mean_eem:.5f}")
print(f"Volatilidad diaria: {daily_volatility_eem:.5f}")
print(f"Rendimiento anual: {annual_return_eem:.2%}")
print(f"Volatilidad anual: {annual_volatility_eem:.2%}")


# ### Comportamientos temporales

# In[105]:


plt.figure(figsize=(10, 6))
plt.plot(eem_data['Adj Close_EEM'], label='Precio Ajustado (Adj Close_EEM)', color='#CD5C5C')
plt.title('Comportamiento del Precio Ajustado de EEM')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.grid(True)
plt.show()


# In[112]:


plt.figure(figsize=(10, 6))
plt.hist(eem_data['Daily Return'], bins=50, alpha=0.75, color='#D8BFD8', edgecolor='#000000')
plt.title('Distribución de Rendimientos Diarios de EEM')
plt.xlabel('Rendimiento Diario (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()


# ## Dataframe stock portfolio

# In[121]:


import yfinance as yf

# Descargar datos para cada activo
activos = {
    "IBB": "IBB",
    "Mason Resources": "MNR",
    "SMH": "SMH",
    "LIT": "LIT",
    "EEM": "EEM"
}

# Diccionario para guardar los datos de precios históricos
datos_historicos = {}

# Descargar datos de cada ticker
for nombre, ticker in activos.items():
    datos_historicos[nombre] = yf.download(ticker, start="2020-01-01", end="2024-11-30", interval="1d")["Adj Close"]



# In[131]:


df_precios = pd.concat(datos_historicos.values(), axis=1)


# In[148]:


df_precios.columns = datos_historicos.keys()


# In[150]:


df_precios


# In[138]:


rendimientos_diarios = df_precios.pct_change().dropna()
volatilidad_diaria = rendimientos_diarios.std()
volatilidad_anual = volatilidad_diaria * np.sqrt(252)


# In[139]:


volatilidad_anual


# In[140]:


matriz_correlacion = df_precios.pct_change().corr()


# In[141]:


matriz_correlacion


# In[142]:


import seaborn as sns
import matplotlib.pyplot as plt

# Crear un heatmap de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlación de los Activos")
plt.show()


# ## Optimización de la cartera usando la Frontera Eficiente
# 

# In[143]:


import numpy as np
import scipy.optimize as sco

# 1. Calcular los rendimientos esperados de los activos (promedio de los rendimientos diarios)
rendimientos_esperados = df_precios.pct_change().mean()

# 2. Calcular la matriz de covarianza de los rendimientos diarios
covarianza = df_precios.pct_change().cov()

# 3. Función objetivo para minimizar el riesgo de la cartera dado un vector de pesos
def riesgo_cartera(pesos, covarianza):
    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

# 4. Restricciones: los pesos deben sumar 1 (100% de la cartera)
def restriccion_pesos(pesos):
    return np.sum(pesos) - 1

# 5. Condición inicial: suponer una cartera con pesos iguales
pesos_iniciales = np.ones(len(df_precios.columns)) / len(df_precios.columns)

# 6. Limitar los pesos entre 0 y 1 (sin posiciones cortas)
restricciones = ({'type': 'eq', 'fun': restriccion_pesos})

# 7. Optimización para minimizar el riesgo (frontera eficiente)
resultado = sco.minimize(riesgo_cartera, pesos_iniciales, args=(covarianza), method='SLSQP', bounds=[(0, 1)]*len(df_precios.columns), constraints=restricciones)

# Mostrar los pesos optimizados
pesos_optimos = resultado.x
print(f"Pesos optimizados: {pesos_optimos}")

# Calcular el rendimiento esperado de la cartera
rendimiento_cartera = np.sum(rendimientos_esperados * pesos_optimos)
print(f"Rendimiento esperado de la cartera: {rendimiento_cartera}")

# Calcular el riesgo de la cartera (volatilidad)
riesgo_cartera_final = riesgo_cartera(pesos_optimos, covarianza)
print(f"Riesgo de la cartera (volatilidad): {riesgo_cartera_final}")


# ## Rentabilidad Anual

# In[144]:


dias_trading = 252

# Rentabilidad anual
rendimiento_anual = (1 + rendimiento_cartera) ** dias_trading - 1
print(f"Rentabilidad anual: {rendimiento_anual:.4f}")


# ## Cálculo del CAGR

# In[145]:


valor_inicial = df_precios.iloc[0].sum()  # Valor total al inicio
valor_final = df_precios.iloc[-1].sum()   # Valor total al final


# In[152]:


num_anos = (df_precios.index[-1] - df_precios.index[0]).days / 365
cagr = (valor_final / valor_inicial) ** (1 / num_anos) - 1
print(f"CAGR de la cartera: {cagr:.4f}")


# ## Matriz de correlación

# In[153]:


matriz_correlacion = df_precios.corr()

# Mostrar la matriz de correlación
print("Matriz de correlación entre activos:")
print(matriz_correlacion)


# In[ ]:





# In[156]:


# Calcular rendimientos diarios logarítmicos de los precios ajustados
rendimientos_activos = np.log(df_precios / df_precios.shift(1))

# Eliminar filas con valores nulos generados por el shift
rendimientos_activos = rendimientos_activos.dropna()


# In[157]:


# Calcular la matriz de covarianza de los rendimientos diarios
matriz_covarianza = rendimientos_activos.cov()


# In[158]:


# Número de simulaciones para la frontera eficiente
num_simulaciones = 10000

# Crear arrays para los rendimientos y riesgos de las carteras simuladas
rendimientos_simulados = []
riesgos_simulados = []

# Realizar simulaciones de carteras con diferentes combinaciones de pesos
for _ in range(num_simulaciones):
    # Generar pesos aleatorios
    pesos = np.random.random(len(df_precios.columns))
    pesos /= np.sum(pesos)  # Normalizar para que la suma sea 1

    # Calcular el rendimiento esperado y la volatilidad de la cartera
    rendimiento_simulado = np.sum(pesos * rendimientos_esperados)
    riesgo_simulado = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))

    # Guardar los resultados
    rendimientos_simulados.append(rendimiento_simulado)
    riesgos_simulados.append(riesgo_simulado)

# Convertir los resultados en arrays
rendimientos_simulados = np.array(rendimientos_simulados)
riesgos_simulados = np.array(riesgos_simulados)

# Graficar la frontera eficiente
plt.figure(figsize=(10, 6))
plt.scatter(riesgos_simulados, rendimientos_simulados, c=rendimientos_simulados / riesgos_simulados, cmap='viridis', marker='o')
plt.title('Frontera Eficiente de la Cartera')
plt.xlabel('Riesgo (Volatilidad)')
plt.ylabel('Rendimiento Esperado')
plt.colorbar(label='Rentabilidad ajustada al riesgo')
plt.show()


# ## Simulación de pesos para una volatilidad del riesgo al 0.016 

# In[159]:


from scipy.optimize import minimize
import numpy as np

# Supongamos que ya tienes estos datos:
# matriz_covarianza: Matriz de covarianza de los activos.
# rendimientos_esperados: Rendimientos diarios esperados de cada activo.

nivel_deseado = 0.016  # Riesgo (volatilidad) deseado

# Función objetivo: Minimizar la diferencia entre el riesgo de la cartera y el nivel deseado
def objetivo_riesgo(pesos, matriz_covarianza, nivel_deseado):
    riesgo = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
    return abs(riesgo - nivel_deseado)

# Restricción: Los pesos deben sumar 1
restriccion_suma = {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}

# Restricción: No permitir pesos negativos (no posiciones cortas)
bounds = [(0, 1) for _ in range(len(rendimientos_esperados))]

# Pesos iniciales (distribución uniforme)
pesos_iniciales = np.ones(len(rendimientos_esperados)) / len(rendimientos_esperados)

# Optimización
resultado = minimize(
    objetivo_riesgo,
    pesos_iniciales,
    args=(matriz_covarianza, nivel_deseado),
    method='SLSQP',
    bounds=bounds,
    constraints=[restriccion_suma]
)

# Pesos optimizados
pesos_optimizados = resultado.x
print("Pesos optimizados:", pesos_optimizados)

# Calcular el rendimiento y riesgo de la cartera para verificar
riesgo_final = np.sqrt(np.dot(pesos_optimizados.T, np.dot(matriz_covarianza, pesos_optimizados)))
rendimiento_final = np.sum(pesos_optimizados * rendimientos_esperados)
print(f"Riesgo de la cartera: {riesgo_final}")
print(f"Rendimiento de la cartera: {rendimiento_final}")


# ## Rentabilidad Anual

# In[161]:


dias_trading = 252

# Rentabilidad anual
rendimiento_anual = (1 + rendimiento_final) ** dias_trading - 1
print(f"Rentabilidad anual: {rendimiento_anual:.4f}")


# ## Frontera eficiente - Comparación gráfica

# In[160]:


import matplotlib.pyplot as plt

# Datos de la frontera eficiente (rendimientos y riesgos simulados previamente)
# Asegúrate de que 'riesgos_simulados' y 'rendimientos_simulados' ya estén definidos.
# Estas listas provienen del proceso de simulación de pesos aleatorios.
# Si no los tienes, revisa las celdas anteriores donde los definimos.
riesgos_simulados = riesgos_simulados  # Volatilidades simuladas
rendimientos_simulados = rendimientos_simulados  # Rendimientos simulados

# Datos de la cartera anterior (punto base)
riesgo_anterior = 0.011183516612297145  # Riesgo (volatilidad)
rendimiento_anterior = 0.0001881380828084073  # Rendimiento esperado

# Datos de la cartera optimizada (con un riesgo de 0.016)
riesgo_optimizado = 0.01599999244691525  # Riesgo (volatilidad optimizada)
rendimiento_optimizado = 0.0009133576640537821  # Rendimiento esperado optimizado

# Generar el gráfico
plt.figure(figsize=(10, 6))
plt.scatter(
    riesgos_simulados,
    rendimientos_simulados,
    c=rendimientos_simulados / riesgos_simulados,
    cmap="viridis",
    label="Frontera Eficiente"
)
plt.colorbar(label="Rentabilidad ajustada al riesgo")
plt.scatter(riesgo_anterior, rendimiento_anterior, color="red", label="Cartera Anterior", s=100)
plt.scatter(riesgo_optimizado, rendimiento_optimizado, color="blue", label="Cartera Optimizada", s=100)
plt.title("Comparación de Carteras dentro de la Frontera Eficiente")
plt.xlabel("Riesgo (Volatilidad)")
plt.ylabel("Rendimiento Esperado")
plt.legend()
plt.show()


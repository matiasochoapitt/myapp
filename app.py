import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import webbrowser

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Lista de tickers disponibles para seleccionar
tickers_disponibles = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NVDA', 'SPY', 'V', 'NFLX']

# Función para descargar los datos históricos de los tickers
def obtener_datos(tickers, start_date, end_date):
    datos = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return datos

# Función para calcular los rendimientos diarios
def calcular_rendimientos(datos):
    rendimientos = datos.pct_change().dropna()
    return rendimientos

# Función para realizar la simulación de portafolios
def simular_portafolios(rendimientos, num_simulaciones=10000):
    num_activos = len(rendimientos.columns)
    resultados = np.zeros((num_simulaciones, 3))  # columnas: [rendimiento, volatilidad, ratio de Sharpe]
    for i in range(num_simulaciones):
        # Generar pesos aleatorios que sumen 1
        pesos = np.random.random(num_activos)
        pesos /= np.sum(pesos)
        
        # Calcular el rendimiento y volatilidad del portafolio
        rendimiento_portafolio = np.sum(pesos * rendimientos.mean()) * 252  # anualizado
        volatilidad_portafolio = np.sqrt(np.dot(pesos.T, np.dot(rendimientos.cov() * 252, pesos)))  # anualizado
        
        # Calcular el ratio de Sharpe (suponiendo tasa libre de riesgo = 0)
        ratio_sharpe = rendimiento_portafolio / volatilidad_portafolio
        
        # Guardar resultados
        resultados[i] = [rendimiento_portafolio, volatilidad_portafolio, ratio_sharpe]
    
    return resultados

# Layout del Dashboard
app.layout = html.Div([
    html.H1("Simulación de Portafolios de Inversión", style={'text-align': 'center'}),
    
    # Selección de tickers
    dcc.Dropdown(
        id='tickers-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers_disponibles],
        value=['AAPL', 'GOOG', 'MSFT'],  # Valor por defecto
        multi=True,
        style={'width': '50%', 'margin': 'auto'}
    ),
    
    # Selector de rango de fechas
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date='2018-01-01',
        end_date='2023-01-01',
        display_format='YYYY-MM-DD',  # Formato de fecha
        style={'width': '50%', 'margin': 'auto'}
    ),
    
    # Gráfico interactivo
    dcc.Graph(id='portafolio-graph'),
    
    # Mostrar el resultado del portafolio óptimo
    html.Div(id='optimal-portfolio-info', style={'text-align': 'center', 'margin-top': '20px'})
])

# Callback para actualizar el gráfico y la información del portafolio óptimo
@app.callback(
    [Output('portafolio-graph', 'figure'),
     Output('optimal-portfolio-info', 'children')],
    [Input('tickers-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def actualizar_dashboard(tickers_seleccionados, start_date, end_date):
    # Obtener los datos históricos
    datos = obtener_datos(tickers_seleccionados, start_date, end_date)
    
    # Calcular los rendimientos
    rendimientos = calcular_rendimientos(datos)
    
    # Realizar la simulación de portafolios
    resultados = simular_portafolios(rendimientos)
    
    # Convertir resultados a un DataFrame
    sim_out_df = pd.DataFrame(resultados, columns=['Portfolio_Return', 'Volatility', 'Sharpe_Ratio'])
    
    # Extraer el portafolio con el mayor ratio de Sharpe
    idx_max_sharpe = np.argmax(resultados[:, 2])
    optimal_portfolio_return = resultados[idx_max_sharpe, 0]
    optimal_volatility = resultados[idx_max_sharpe, 1]
    optimal_sharpe_ratio = resultados[idx_max_sharpe, 2]
    
    # Crear el gráfico interactivo con Plotly
    fig = px.scatter(sim_out_df, 
                     x='Portfolio_Return', 
                     y='Volatility', 
                     color='Sharpe_Ratio', 
                     hover_data=['Sharpe_Ratio'], 
                     color_continuous_scale='Viridis')
    
    # Resaltar el punto con el mayor ratio de Sharpe
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio_return], 
        y=[optimal_volatility], 
        mode='markers', 
        name='Optimal Point', 
        marker=dict(size=20, color='red')
    ))

    fig.update_layout(
        title='Simulación de Portafolios',
        xaxis_title='Rendimiento Anualizado',
        yaxis_title='Volatilidad Anualizada',
        plot_bgcolor="white",
        coloraxis_colorbar=dict(y=0.7, dtick=5)
    )
    
    # Información del portafolio óptimo
    optimal_info = (
        f"Portafolio con el mayor ratio de Sharpe:\n"
        f"Rendimiento anualizado: {optimal_portfolio_return:.2f}\n"
        f"Volatilidad anualizada: {optimal_volatility:.2f}\n"
        f"Ratio de Sharpe: {optimal_sharpe_ratio:.2f}"
    )
    
    return fig, optimal_info


# Función para abrir el navegador automáticamente
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
server = app.server
# Ejecutar la aplicación
if __name__ == '_main_':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
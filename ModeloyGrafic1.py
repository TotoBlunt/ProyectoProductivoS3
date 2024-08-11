import pandas as pd 
import openpyxl
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st 


#Configuracion de streamlit
st.title('Modelo de IA para Granja de Pollo con Comparacion de Predicciones')
st.write('''
        Esta aplicacion permite realizar las predicciones del peso final del pollo incluyendo el consumo de alimentos en la fase 'Acabado -Finalizador' asi como tamboien con la estacion del corral (Veran, Otoño, Invierno,Primavera) y finalizando con el sexo del pollo en mencion tabien se puede visualizar la comparacion de peso real vs el peso predicho por el algoritmo''')
    
#CARGAR DATOS
uploaded_file = st.file_uploader("Cargar Archivo Excel" ,type=['xlsx'])
    
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        # Convertir la columna objetivo a valores numéricos
        data['EstaciónCorral'] = data['EstaciónCorral'].map({"Verano":1,"Otoño":2,"Invierno":3,"Primavera":4})
        data['Nsexo'] = data["Nsexo"].map({"Macho":0,"Hembra":1})
        
        #DIVIR EL DATA FRAME EN CARACTERISTICAS Y ETIQUETAS PARA ENTRENAR EL MODELO
        x = data[['Consumo Acabado',' Consumo Finalizador','EstaciónCorral','Nsexo']] #Cuando es mas de una columna se utiliza dos corchetes para que lo lea correctamente
        y = data['Peso Prom. Final'] 
        
        #Conjuntpo de Pruebas
        # Dividir los datos en conjunto de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        
        #Crear una lista de modelos
        models = [ ("decision_tree", DecisionTreeRegressor()), ("linear_regression",LinearRegression()), ("k_neighbors",KNeighborsRegressor(n_neighbors=5))] #de los 5 vecinos sacara deciciones estadisticas
        # Crear un modelo de ensamble con los modelos anteriores el metodo combina las predicciones de varios modelos 
        model = VotingRegressor(models)
        
        #Entrenar el modelo con los datos
        model.fit(x,y)
        
        # Hacer predicciones con el modelo usando datos de prueba
        y_pred = model.predict(x_test)
        
        #Crear una nueva colmuna en el archivo Excel
        data['Peso Prom. Final Predicho'] = model.predict(x)
        
        # Calcular los errores (residuales)
        data['Error'] = data['Peso Prom. Final'] - data['Peso Prom. Final Predicho']

        # Grafico de Comparacion
        fig,ax = plt.subplots()

        ax.plot(data['Peso Prom. Final'], label='Peso Prom. Final (Estático)', color='blue')
        ax.plot(data['Peso Prom. Final Predicho'], label='Peso Prom. Final Predicho', color='red')
        ax.xlabel('Índice')
        ax.ylabel('Peso Prom. Final')
        ax.title('Comparación entre Peso Prom. Final Estático y Predicho')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        #varianza
        varianza = ((data['Peso Prom. Final'] - data['Peso Prom. Final Predicho']) **2).mean()
        st.write(f"La varianza de los valores es:  {varianza:.4f}")
        
        # Calcular métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"El error cuadratico medio es: {r2}")
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

# Crear una ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados en formato JSON
    datos = request.get_json()

    # Convertir los datos en un DataFrame
    datos_df = pd.DataFrame([datos])

    # Realizar predicciones
    prediccion = modelo.predict(datos_df)

    # Devolver la predicción en formato JSON
    return jsonify({'prediccion': prediccion.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

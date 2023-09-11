from flask import Flask, render_template, request, url_for, flash, redirect, jsonify

import pickle
from datetime import datetime
import pandas as pd
from sklearn import datasets
from sqlalchemy import create_engine

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from flask import Flask, jsonify, request

import pickle
import pandas as pd



from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import joblib

from datetime import datetime
from flask import Flask, jsonify, request

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


import sqlite3

from sqlalchemy import create_engine

app = Flask(__name__)

#Importo los datos del iris dataset
iris = datasets.load_iris()

#Las credenciales de la base de datos
DATABASE_CONFIG = {"user": "postgres",
                   "password": "admin1234",
                   "host": "database-1.ciwxv2n0vdmm.us-east-1.rds.amazonaws.com",
                   "port": "5432",
                   "database": "predictions"}


# Creo la conexion
engine = create_engine(f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")





app = Flask(__name__)

iris = datasets.load_iris()

@app.route("/", methods=["GET"])
def welcome():

    
    msg = """Bienvenido a nuestra app

    
    
        """
    return msg




@app.route("/v0/predictor", methods =["GET"])
def predictor_v0():
    with open ("iris_model.pkl", "rb") as file:
        model = pickle.load(file)


    s_length = float(request.args.get("s_length", None))
    s_width = float(request.args.get("s_width", None))
    p_length = float(request.args.get("p_length", None))
    p_width = float(request.args.get("p_width", None))
    time = datetime.now()
    formatted_dt = time.strftime('%Y-%m-%d %H:%M')




    if s_length is None or s_width is None or p_width is None or p_length is None:
        print("<Faltan datos, asegurate de que está todo")
    else:
        prediction = model.predict([[s_length,s_width,p_length,p_width]])
        class_name = iris.target_names[prediction]
        prediction = str(class_name[0])
        upload = {"p_length": p_length, "p_width": p_width, "s_length": s_length, "s_width": s_width, "prediction":prediction, "timestamp":formatted_dt}


        upload = pd.DataFrame([upload])
        upload.to_sql('predictions', con=engine, if_exists='append', index=False)


        return  jsonify({"prediction": prediction})


@app.route("/v1/predictor", methods =["GET"])
def predictor_v1():
    with open ("iris_model.pkl", "rb") as file:
        model = pickle.load(file)

    payload = request.get_json()
    s_length = float(payload["s_length"])
    s_width = float(payload["s_width"])
    p_length = float(payload["p_length"])
    p_width = float(payload["p_width"])
    time = datetime.now()
    formatted_dt = time.strftime('%Y-%m-%d %H:%M')


    if s_length is None or s_width is None or p_width is None or p_length is None:
        print("<Faltan datos, asegurate de que está todo")
    else:
        prediction = model.predict([[s_length,s_width,p_length,p_width]])
        class_name = iris.target_names[prediction]
        prediction = str(class_name[0])
        upload = {"p_length": p_length, "p_width": p_width, "s_length": s_length, "s_width": s_width, "prediction":prediction, "timestamp":formatted_dt}


        upload = pd.DataFrame([upload])
        upload.to_sql('predictions', con=engine, if_exists='append', index=False)


        return  jsonify({"prediction": prediction})
    


@app.route("/api/v0/predictions/get_all", methods=["GET"])
def return_all():

    todo = pd.read_sql_query(f"select * from predictions", con=engine).to_dict("records")
    
    return jsonify(todo)

@app.route('/api/v0/resources/predictions/eliminar', methods=['DELETE'])
def delete_flower():
    # Obtener los datos del libro en formato JSON desde la petición
    flower_data = request.get_json()
    # Obtener el título del libro que se quiere eliminar
    _id = flower_data['id']
    # Conectar con la base de datos sqlite
    conn = sqlite3.connect('iris.db')
    # Crear un cursor para ejecutar sentencias SQL
    cursor = conn.cursor()
    # Ejecutar la sentencia SQL para eliminar el libro con el título dado
    cursor.execute("DELETE FROM predictions WHERE id = ?", (_id,))
    # Guardar los cambios en la base de datos
    conn.commit()
    # Cerrar la conexión con la base de datos
    conn.close()
    # Devolver una respuesta en formato JSON con un mensaje de éxito
    return jsonify({'message': 'Predicción eliminado correctamente'})


@app.route("/v1/predictor_norm", methods =["GET"])
def predictor_v1_norma():
    with open ("iris_model_sc.pkl", "rb") as file:
        model_sc = pickle.load(file)
    

    payload = request.get_json()
    escalador = joblib.load('std_scaler.bin')

    s_length_u = float(payload["s_length"])
    s_width_u = float(payload["s_width"])
    p_length_u = float(payload["p_length"])
    p_width_u = float(payload["p_width"])

    payload = request.get_json()
    s_length = float(payload["s_length"])
    s_width = float(payload["s_width"])
    p_length = float(payload["p_length"])
    p_width = float(payload["p_width"])
    time = datetime.now()
    formatted_dt = time.strftime('%Y-%m-%d %H:%M')

    b = [[s_length, s_width, p_length, p_width]]

    X_train_SC = escalador.transform(b)


    if s_length is None or s_width is None or p_width is None or p_length is None:
        print("<Faltan datos, asegurate de que está todo")
    else:
        prediction = model_sc.predict(X_train_SC)
        class_name = iris.target_names[prediction]
        prediction = str(class_name[0])
        upload = {"p_length": p_length_u, "p_width": p_width_u, "s_length": s_length_u, "s_width": s_width_u, "prediction":prediction, "timestamp":formatted_dt}


        upload = pd.DataFrame([upload])
        upload.to_sql('predictions', con=engine, if_exists='append', index=False)


        return  jsonify({"prediction": prediction})
    

@app.route("/v1/predictor_bin", methods= ["GET"])
def predictor_bin():
    with open ("iris_model_bin.pkl", "rb") as file:
        model = pickle.load(file)

    payload = request.get_json()

    if payload["p_length"] < 3:
        p_length = 0
    else:
        p_length = 1

    s_length = float(payload["s_length"])
    s_width = float(payload["s_width"])
    p_length_u = float(payload["p_length"])
    p_width = float(payload["p_width"])
    time = datetime.now()
    formatted_dt = time.strftime('%Y-%m-%d %H:%M')


    if s_length is None or s_width is None or p_width is None or p_length is None:
        print("<Faltan datos, asegurate de que está todo")
    else:
        prediction = model.predict([[s_length,s_width,p_length,p_width]])
        class_name = iris.target_names[prediction]
        prediction = str(class_name[0])

        upload = {"p_length": p_length_u, "p_width": p_width, "s_length": s_length, "s_width": s_width, "prediction":prediction, "timestamp":formatted_dt}


        upload = pd.DataFrame([upload])
        upload.to_sql('predictions', con=engine, if_exists='append', index=False)
        
        return  jsonify({"prediction": prediction})


@app.route("/api/v0/retrain", methods=["GET"])
def retrain():

    #Descargamos los datos de la base de datos para añadir al retrain
    datos_retrain = pd.read_sql_query(f"select * from predictions", con=engine)


    comparador = pd.read_sql_query(f"select * from compare", con=engine)
    if len(comparador) == len(datos_retrain):
        return "Tu modelo ya esta entrenado con los datos más recientes"
    else:
    
        #Hacemos un merge de todo y comparador, juntamos todo en un solo dataframe donde se han eliminado los duplicados al ser un outer merge, e indicator= True nos crea una columna que nos dice que filas estaban en que dataframe originalmente.
        df = pd.merge(datos_retrain, comparador,  how="outer", indicator=True)

        # Seleccionamos las filas que son nuevas.
        df = df[df['_merge'] == 'left_only']

        #Creamos el dataframe que subiremos a la tabla de comparaciones
        upload = df.drop(["_merge"],axis=1) 

        #Actualizamos la tabla de comparaciones
        upload.to_sql('compare', con=engine, if_exists='append', index=False)

        # Hacemos el drop de las columnas que no nos sirven para el train
        df2 = df.drop(["id","timestamp",	"_merge"],axis=1)

        # Creamos diccionario para reemplazar los nombres por numeros
        flowers = {"setosa":0, "versicolor":1, "virginica":2}


        # Reemplazamos las flores por su numero correspondiente
        df2['prediction'].replace(flowers, inplace=True)

        # Cogemos los datos de entrenamiento que ya tenemos
        train_original = pd.read_sql_query(f"select * from datos_train", con=engine)

        #Cambiamos el tipo de la columna prediction para que coincida con el de df2
        train_original["prediction"] = train_original["prediction"].astype("int64")

        # Juntamos ambos dataframes
        retrain = pd.concat([train_original, df2])


        # Separamos en X e y
        X = retrain.drop("prediction", axis=1)
        y = retrain["prediction"]


        # Hacemos el train test split
        X_train, X_test, y_train, y_test = train_test_split(X,           
                                                            y,
                                                            test_size = 0.20,
                                                            random_state=42,
                                                            shuffle=True)
        
        #Entrenamos el modelo
        model = LogisticRegression(max_iter= 1000)
        model.fit(X_train, y_train)

        #Guardamos el modelo
        pickle.dump(model, open('iris_model_retrain.pkl', 'wb'))

        #Actualizamos la tabla con los datos de train
        df2.to_sql('datos_train', con=engine, if_exists='append', index=False)

        return "Modelo actualizado"

if __name__ == "__main__":
    app.run(debug=True, port=5002)
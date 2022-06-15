"""
Run a rest API exposing the yolov5s object detection mode
"""
import argparse
import io                                     #librerias para manejar carpetas del sistema operativo
from urllib import response                   #para el manejo de URL
from PIL import Image                         #Transformacion de imagenes
import base64
from io import BytesIO
import shutil                                 #Eliminacion de carpetas del sistema operativo
from shutil import rmtree
import os
import requests                                     #Para controlar sistema operativo

import torch                                  
from flask import Flask, render_template, request, send_file, make_response      #lib para crear el servidor web
from flask_ngrok import run_with_ngrok        #lib para crear la URL publica 

application = Flask(__name__)
#run_with_ngrok(application)                   #linea para indicar que se arrancara el servidor con Ngrok

DETECTION_URL = "/detect"                                 #Direccion que contiene la fucionalidad de detección

@application.route(DETECTION_URL, methods=["POST"])       #Se asigna la direccion y indica que admite el metodo POST
def predict():
    if not request.method == "POST":                      #cConsulta si lo recibido es un POST
        return

    if request.files.get("image"):         #Captura los Datos recibidos y obtiene el dato que tiene la llave "image"
        image_file = request.files["image"]
        image_bytes = image_file.read()    #Lee el archivo

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)     #pasa la imagen al modelo con un tamaño de imagen de 640px de ancho
        results.save()                     #Guarda la imagen con la deteccion en la carpeta run/detect/exp
        ###
        contenido = os.listdir('.\\runs\\detect\\exp')  #Almacena el nombre de la imagen en contenido, posicion 0
        shutil.copy(".\\runs\\detect\\exp\\"+contenido[0], ".\\images\\foto_detectada.jpg") # copia la imagen a la carpeta images con el nombre "foto_detectada.jpg"
        rmtree(".\\runs\\detect\\exp")     #Se elimina la carpeta runs con sus respectivas subcarpetas
        ###
        data = results.pandas().xyxy[0]     # Se almacenan los parametros de deteccion

        results.imgs # array of original images (as np array) passed to model for inference
        results.render()  # updates results.imgs with boxes and labels

        #for img in results.imgs:   
        # buffered = BytesIO()
        # im_base64 = Image.fromarray(img)
        # im_base64.save(buffered, format="JPEG")
        # print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # imagen base 64, si se requiere asi      

        print(str(data.values[0][6])) # Muestra por consola la Clasificacion de sonrisa detectada
        response= make_response(send_file('.\\images\\foto_detectada.jpg', download_name="foto_detectada.jpg")) 
        # Se agrega la clasificacion detectada a un Header de la foto : DetectionVal = (ALTA, MEDIA o BAJA)
        response.headers['DetectionVal'] = str(data.values[0][6]) 
        return response  
        # Se envia la imagen con la detección y con el valor de la Clasfición en el Campo DetectionVal
        

@application.route("/send-image/<path:url>")
def image_check(url):
    # ----- SECTION 1 -----  
    #File naming process for nameless base64 data.
    #We are using the timestamp as a file_name.
    from datetime import datetime
    dateTimeObj = datetime.now()
    file_name_for_base64_data = dateTimeObj.strftime("%d-%b-%Y--(%H-%M-%S)")
    
    #File naming process for directory form <file_name.jpg> data.
    #We are taken the last 8 characters from the url string.
    file_name_for_regular_data = url[-10:-4]
    
    # ----- SECTION 2 -----
    try:
        # Base64 DATA
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))

            file_name = file_name_for_base64_data + ".jpg"
            img.save(file_name, "jpeg")

        # Base64 DATA
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))

            file_name = file_name_for_base64_data + ".png"
            img.save(file_name, "png")

        # Regular URL Form DATA
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            file_name = file_name_for_regular_data + ".jpg"
            img.save(file_name, "jpeg")
        
    # ----- SECTION 3 -----    
        status = "Image has been succesfully sent to the server."
    except Exception as e:
        status = "Error! = " + str(e)

    return status

@application.route('/none') # Ruta para prueba de funcionamiento,  Solo muestra el memsaje de hola en el navegador
def none():
    return render_template('index.html') # se debe llamar con GET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # Carga el detector con el modelo COCO
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) # Carga el detector con el modelo Sonrisas
    model.conf = 0.7 # Indica el nivel de confianza minimo en la detección
    model.eval()

    application.run(host="0.0.0.0", port=4000, debug=True)  # Inicia en servidor Local
    #application.run() # inicia en Servidor Remoto,  Tener en cuenta que en cada inicio de servidor esta direccion cambia
                      # debido a que se esta usando una libreria gratuita de tunelamiento.

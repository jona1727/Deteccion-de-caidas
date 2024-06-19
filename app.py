import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import telebot
from geopy.geocoders import Nominatim
from flask import Flask, render_template, Response
import os
import time
import datetime 

# Flask app setup
app = Flask(__name__)

# YOLO model and SORT tracker setup
model = YOLO("best.pt")
tracker = Sort()

# Telegram Bot setup
TOKEN = '6645351405:AAHfjWddfRVoSeTnNuV7Erly_D-zZ9b_1FY'
bot = telebot.TeleBot(TOKEN)

# Global variables for alert timing
ultima_ejecucion = 0
ultima_notificacion_ausente = 0
persona_ausente = False

def obtener_direccion(latitud, longitud):
    geolocalizador = Nominatim(user_agent="mi_aplicacion")
    ubicacion = geolocalizador.reverse((latitud, longitud))
    if ubicacion:
        print(f"Dirección encontrada: {ubicacion.address}")
        return ubicacion.address
    else:
        print("Dirección no encontrada.")
        return "Dirección no encontrada."

# Geolocation setup
latitud = -0.905863
longitud = -78.621728
direccion = obtener_direccion(latitud, longitud)

def enviar_alerta(chat_id, mensaje):
    global ultima_ejecucion
    tiempo_actual = time.time()

    if tiempo_actual - ultima_ejecucion >= 10:
        bot.send_message(chat_id, mensaje)
        ultima_ejecucion = tiempo_actual
        if "Caida detectada" in mensaje:  # Solo enviar foto para alertas de caída
            with open("images/persona_detectada.jpg", 'rb') as photo:
                bot.send_photo(chat_id, photo)
    else:
        print("La función enviar_alerta no puede ejecutarse más de una vez en 10 segundos.")

def gen_frames():
    global ultima_notificacion_ausente, persona_ausente

    prev_y0 = None
    prev_time = time.time()
    jona = 0
    cap = cv2.VideoCapture(0)
    
    y_line = 0
    y_line2 = 0

    def is_time_to_activate(start_hour, end_hour):
        current_time = datetime.datetime.now().time()
        start_time = datetime.datetime.strptime(start_hour, "%H:%M").time()
        end_time = datetime.datetime.strptime(end_hour, "%H:%M").time()
        return start_time <= current_time or current_time <= end_time
    
    def main():
        start_hour = "20:00"
        end_hour = "7:00"
        return is_time_to_activate(start_hour, end_hour)
    
    main()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        results = model(frame, show=False, show_boxes=False, show_labels=False, stream=True)
        keypoints_detected = False

        for res in results:
            annotated_frame = res.plot(labels=False, conf=False, boxes=False)
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            kpts = res.keypoints

            y_line = frame.shape[0] * 1 // 2
            y_line2 = frame.shape[0] * 6 // 8

            for track_id, (xmin, ymin, xmax, ymax) in zip(tracks[:, 4], tracks[:, :4]):
                w = xmax - xmin
                h = ymax - ymin
                carpeta_regiones = "images"
                ruta_archivo = os.path.join(carpeta_regiones, f"persona_detectada.jpg")
                region_persona = frame[ymin:ymax, xmin:xmax]

                keypoints_detected = True

                if prev_y0 is not None:
                    current_y0 = kpts.xy[0, 0][1]
                    delta_y0 = abs(current_y0 - prev_y0)
                    current_time = time.time()
                    delta_time = current_time - prev_time
                    jona = delta_y0 / delta_time
                    if delta_time > 0 and delta_y0 / delta_time > 100:
                        print("Cambio brusco detectado en el punto 0.")

                    aspect_ratio = w / h if w != 0 and h != 0 else 0
                    depth_image = 1500
                    distance_threshold = 1.5
                    read_proximity_sensor = lambda x, y, z, w, v: 2.0
                    color_image = frame
                    if 0.3 < aspect_ratio < 0.6 and any(kpts.xy[0, idx][1] > y_line2 for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]):
                        min_distance = read_proximity_sensor(depth_image, xmin, xmax, ymin, ymax)
                        if min_distance < distance_threshold:
                            cv2.putText(color_image, f"Caída {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                            cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                            cv2.imwrite(ruta_archivo, region_persona)

                if any(kpts.xy[0, idx][1] > y_line for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]) and jona > 120 and w / h > 0.7:
                    cv2.putText(annotated_frame, f"Inestable : {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
                
                elif w / h > 0.7 and any(kpts.xy[0, idx][1] > y_line2 for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]):
                    cv2.putText(annotated_frame, f"Caida: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    cv2.imwrite(ruta_archivo, region_persona)
                    enviar_alerta('5003167951', "Caida detectada adulto mayor. " + "Localizacion: " + str(direccion))

                elif any(kpts.xy[0, idx][1] < y_line2 for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]) and w / h > 0.7:
                    cv2.putText(annotated_frame, f"Acostado: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, f"Estable: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                prev_y0 = kpts.xy[0, 0][1]
                prev_time = time.time()
                
            # cv2.line(annotated_frame, (0, y_line), (frame.shape[1], y_line), (255, 0, 0), 2)
            cv2.line(annotated_frame, (0, y_line2), (frame.shape[1], y_line2), (255, 0, 0), 2)

        if not keypoints_detected:
            current_time = time.time()
            if not persona_ausente:
                if current_time - ultima_notificacion_ausente >= 300:  # 5 minutos
                    enviar_alerta('5003167951', "El adulto mayor ha salido del lugar de monitoreo. " + "Localizacion: " + str(direccion))
                    ultima_notificacion_ausente = current_time
                persona_ausente = True
        else:
            if persona_ausente:
                enviar_alerta('5003167951', "El adulto mayor ha vuelto al lugar de monitoreo. " + "Localizacion: " + str(direccion))
                persona_ausente = False

        annotated_frame = cv2.resize(annotated_frame, (800, 600))
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video2', methods=['GET', 'POST'])
def index2():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)


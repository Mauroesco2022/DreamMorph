import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock

# Rutas de los modelos .tflite
cartoon_model_path = "1.tflite"
selfie2anime_model_path = "selfie2anime.tflite"
van_gogh_style_image_path = "van_gogh_style.jpg"

# Cargar el modelo de estilo de TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Función para preprocesar la imagen
def preprocess_image(image, target_size):
    image = cv2.resize(image, (target_size, target_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    return image

# Clase para la pantalla de detección de color en tiempo real
class RealTimeColorDetectionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Widget de imagen para mostrar el video en tiempo real
        self.video_widget = Image()
        layout.add_widget(self.video_widget)

        # Etiqueta para mostrar el color detectado
        self.color_label = Label(text="Color Detectado: Ninguno", size_hint_y=0.1)
        layout.add_widget(self.color_label)

        # Botón para ir a la pantalla de captura de imagen
        go_to_capture_button = Button(text="Ir a Capturar Imagen", size_hint_y=0.1)
        go_to_capture_button.bind(on_press=self.go_to_capture_screen)
        layout.add_widget(go_to_capture_button)

        self.add_widget(layout)

        # Inicializar captura de video
        self.capture = None
        self.detected_color = None
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Actualizar a 30 FPS

    def on_enter(self):
        # Iniciar la captura de video cada vez que se entra en esta pantalla
        if self.capture is None or not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)

    def update(self, dt):
        # Leer un frame de la cámara
        if self.capture is None or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        # Convertir la imagen de BGR a HSV para la detección de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definir los rangos de color en HSV
        blue_range = ((100, 150, 0), (140, 255, 255))
        red_range = ((0, 70, 50), (10, 255, 255))
        green_range = ((35, 100, 50), (85, 255, 255))

        # Crear máscaras y contar píxeles para cada color
        blue_mask = cv2.inRange(hsv, *blue_range)
        red_mask = cv2.inRange(hsv, *red_range)
        green_mask = cv2.inRange(hsv, *green_range)

        blue_pixels = cv2.countNonZero(blue_mask)
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        # Determinar el color predominante
        if blue_pixels > red_pixels and blue_pixels > green_pixels:
            self.manager.detected_color = "blue"
            self.color_label.text = "Color Detectado: Azul - CartoonGAN"
        elif red_pixels > blue_pixels and red_pixels > green_pixels:
            self.manager.detected_color = "red"
            self.color_label.text = "Color Detectado: Rojo - Selfie2Anime"
        elif green_pixels > blue_pixels and green_pixels > red_pixels:
            self.manager.detected_color = "green"
            self.color_label.text = "Color Detectado: Verde - Van Gogh"
        else:
            self.manager.detected_color = None
            self.color_label.text = "No se detectó un color predominante."

        # Mostrar el video en tiempo real
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.video_widget.texture = texture

    def go_to_capture_screen(self, instance):
        # Ir a la pantalla de captura de imagen
        self.manager.current = "capture_screen"

    def on_leave(self):
        # Liberar la cámara cuando se sale de la pantalla
        if self.capture is not None:
            self.capture.release()
            self.capture = None

# Clase para la pantalla de captura de imagen y aplicación de estilo
class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Etiqueta para mostrar el color detectado y el modelo de estilo
        self.color_label = Label(text="Color Detectado: Ninguno", size_hint_y=0.1)
        layout.add_widget(self.color_label)

        # Botón para capturar la imagen
        capture_image_button = Button(text="Capturar Imagen", size_hint_y=0.1)
        capture_image_button.bind(on_press=self.capture_image)
        layout.add_widget(capture_image_button)

        # Botón para regresar a la pantalla de detección de color
        go_back_button = Button(text="Regresar a Detección de Color", size_hint_y=0.1)
        go_back_button.bind(on_press=self.go_back_to_color_detection)
        layout.add_widget(go_back_button)

        # Botón para detectar otro color
        detect_another_color_button = Button(text="Detectar otro color", size_hint_y=0.1)
        detect_another_color_button.bind(on_press=self.detect_another_color)
        layout.add_widget(detect_another_color_button)

        # Widget de imagen para mostrar el resultado
        self.image_widget = Image()
        layout.add_widget(self.image_widget)

        self.add_widget(layout)
        self.captured_image = None

    def on_pre_enter(self):
        # Actualizar el texto del color detectado
        detected_color = self.manager.detected_color
        color_text = {
            "blue": "Azul - CartoonGAN",
            "red": "Rojo - Selfie2Anime",
            "green": "Verde - Van Gogh"
        }.get(detected_color, "Ninguno")
        self.color_label.text = f"Color Detectado: {color_text}"

    def capture_image(self, instance):
        # Capturar imagen
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.color_label.text = "Error al capturar la imagen"
            return

        self.captured_image = frame
        self.apply_style()

    def apply_style(self):
        detected_color = self.manager.detected_color
        if detected_color is None or self.captured_image is None:
            self.color_label.text = "Debe detectar un color y capturar una imagen primero."
            return

        if detected_color == "blue":
            styled_image = apply_style(self.captured_image, model_path=cartoon_model_path)
        elif detected_color == "red":
            styled_image = apply_style(self.captured_image, model_path=selfie2anime_model_path)
        elif detected_color == "green":
            style_image = cv2.imread(van_gogh_style_image_path)
            styled_image = apply_style(self.captured_image, use_hub_model=True, style_image=style_image)
        else:
            self.color_label.text = "No se pudo aplicar el estilo."
            return

        # Mostrar el resultado
        buf = cv2.flip(styled_image, 0).tobytes()
        texture = Texture.create(size=(styled_image.shape[1], styled_image.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image_widget.texture = texture

    def go_back_to_color_detection(self, instance):
        # Regresar a la pantalla de detección de color
        self.manager.current = "color_detection_screen"

    def detect_another_color(self, instance):
        # Limpiar la imagen capturada y el widget de imagen
        self.captured_image = None
        self.image_widget.texture = None
        # Reiniciar la detección de color y volver a la pantalla de detección
        self.manager.detected_color = None
        self.manager.get_screen("color_detection_screen").on_enter()
        self.manager.current = "color_detection_screen"

# Función para aplicar el estilo
def apply_style(image, model_path=None, use_hub_model=False, style_image=None):
    if use_hub_model:
        content_image = preprocess_image(image, target_size=512)
        style_image = preprocess_image(style_image, target_size=512)
        outputs = hub_model(tf.constant(content_image), tf.constant(style_image))
        styled_image = outputs[0][0].numpy()
        return np.clip((styled_image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    else:
        target_size = 512 if model_path == cartoon_model_path else 256
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        processed_image = preprocess_image(image, target_size=target_size)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])
        output_image = (np.squeeze(result) + 1.0) * 127.5
        return np.clip(output_image, 0, 255).astype(np.uint8)

# Clase de la aplicación principal
class StyleApp(App):
    def build(self):
        sm = ScreenManager()
        sm.detected_color = None  # Variable para almacenar el color detectado
        sm.add_widget(RealTimeColorDetectionScreen(name="color_detection_screen"))
        sm.add_widget(CaptureScreen(name="capture_screen"))
        return sm

if __name__ == "__main__":
    StyleApp().run()

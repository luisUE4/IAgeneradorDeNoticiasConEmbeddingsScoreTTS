
# Script para probar el modelo de generación de voz Kokoro-82M
# Modelo TTS optimizado con ONNX Runtime

import os
import sys
from datetime import datetime
import time
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro
from phonemizer import phonemize

# --- NUEVOS IMPORTS PARA CONTROLAR LA CPU ---
import onnxruntime
from onnxruntime import InferenceSession


# Configurar espeak-ng para phonemizer
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG\espeak-ng.exe'
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'




class KokoroService:
    def __init__(self):
        print("========================================================")
        print("🤖 Inicializando Kokoro ONNX por única vez...")
        print("TTS con Kokoro-82M")

        # 1. Inicializar el modelo Kokoro-82M
        # print("\nCargando modelo Kokoro-82M (esto puede tardar la primera vez)...")
        start_time_model_load = time.time()
        
        try:

            # configurar sesion de kokoro para reducir la cantidad de proesador que puede consumir
            # agrego esto para evitar grandes picos de procesamiento cpu , mejor que se mantenga bajo 
            # porque tiene tiempo parar procesar lentamente
            sess_options = onnxruntime.SessionOptions()
            # Asigna aquí cuántos hilos virtuales máximo quieres que use Kokoro. 
            # Si tienes un procesador de 8 o 12 hilos, dejarlo en 2 o 4 evitará que tu CPU brinque al 100%.
            NÚMERO_DE_HILOS = 2 
            
            sess_options.intra_op_num_threads = NÚMERO_DE_HILOS
            sess_options.inter_op_num_threads = NÚMERO_DE_HILOS
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

            # 2. CARGAR LA SESIÓN DE ONNX DE FORMA MANUAL CON LAS OPCIONES
            # Reemplazamos la inicialización directa por la sesión limitada
            session = InferenceSession(
                "kokoro/kokoro-v1.0.onnx", 
                providers=['CPUExecutionProvider'], 
                sess_options=sess_options
            )
            self.kokoro = Kokoro.from_session(session, "kokoro/voices-v1.0.bin")

            # Inicializar Kokoro sin limite de hilos cpu
            # self.kokoro = Kokoro("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
            

            
            # print("Voces disponibles en el binario:", self.kokoro.get_voices())
            #  voces disponibles :: 
            # ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang']

            end_time_model_load = time.time()
            print(f"Tiempo de carga del modelo kokoro: {end_time_model_load - start_time_model_load:.2f} segundos")
            print("¡Modelo kokoro cargado exitosamente!")
            
            ################## 3. Configurar la voz
    
            # voice = "af_sarah"  # Voz femenina americana suena agringada
            # voice = "ef_dora"   # mujer con voz de señora muy formal
            # voice = "im_nicola"
            # voice = [("ef_dora", 0.7), ("af_bella", 0.3)]  #kokoro-onnx permite promediar los vectores de dos voces distintas
            # voice = "zf_xiaoxiao"
            # af_nova o af_river: Tienen un tono más dinámico, cálido o "animado".


            # 1. Extraer los vectores de ambas voces desde el archivo binario
            # kokoro.get_voice_style obtiene una matriz de numpy con el "estilo" de la voz
            estilo_dora = self.kokoro.get_voice_style("ef_dora")
            estilo_bella = self.kokoro.get_voice_style("zf_xiaoxiao")    
            # 2. Mezclar los estilos: 65% la pronunciación de Dora y 35% el tono agudo de Bella
            self.voice = (estilo_dora * 0.60) + (estilo_bella * 0.40)


            self.speed = 0.9  # Velocidad ligeramente más rápida para simular comentarista deportiva
            # lang = "es"  # Idioma español
            self.lang = "es-419"

            # print(f"\nConfiguración de voz:")
            # print(f"  - Voz: {voice}")
            # print(f"  - Velocidad: {speed}x")
            # print(f"  - Idioma: {lang}")


        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("\nPosibles soluciones:")
            print("- Verifica tu conexión a internet (el modelo se descarga automáticamente)")
            print("- Asegúrate de tener espacio suficiente en disco")
            print("- Verifica la instalación: pip show kokoro-onnx")
            sys.exit(1)



    def texto_a_voz_kokoro(self, texto_para_convertir_a_voz:str ):    

        texto = texto_para_convertir_a_voz
        
        print(f"\nkokoro -> Generando audio para el texto:")
        print(f"'{texto[:100]}...'")
        
        dict_tiempos = dict()
        start_time_audio_gen = time.time()
        
        try:
            # Convertir texto a fonemas usando phonemizer
            phonemes = phonemize(
                texto,
                language='es-419',
                # language='es',
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                with_stress=True
            )
            

            # Generar audio usando fonemas
            samples, sample_rate = self.kokoro.create(
                phonemes,
                self.voice,  
                self.speed,
                is_phonemes=True
            )
                    
            end_time_audio_gen = time.time()    
            total_audio_generation_time = end_time_audio_gen - start_time_audio_gen
            
            # TRUCO: Hace la voz más aguda aumentando los Hz un 15%
            # sample_rate = int(sample_rate * 1.25)

            # calcular timestamps por palabra
            duracion_audio = len(samples) / sample_rate # Calcular duración
            dict_tiempos = self.obtener_diccionario_timestamps(texto, duracion_audio)

 
            # print(f"  - Duración del audio: {len(samples) / sample_rate:.2f} segundos")
            # print(f"  - Tasa de muestreo: {sample_rate} Hz")
            print(f"   Audio generado exitosamente, Tiempo de generación: {total_audio_generation_time:.2f} segundos")
            # print(f"  - Fonemas generados: {phonemes}")
            
        except Exception as e:
            print(f"\n✗ Error al generar el audio: {e}")
            print("\nDetalles del error:")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        ####################### 5. Guardar el archivo generado
        directorio_salida = "vozKokoro"
        os.makedirs(directorio_salida, exist_ok=True)
        
        timestamp = datetime.now().strftime("%m.%d.%H.%M")
        nombre_archivo_salida = os.path.join(directorio_salida, f"comentario_kokoro_{timestamp}.wav")
        

        try:
            # Guardar el audio usando soundfile
            sf.write(nombre_archivo_salida, samples, sample_rate)
            
            # print(f"\n{'='*60}")
            # print(f"✓ ÉXITO: Audio guardado como '{nombre_archivo_salida}'")
            # print(f"{'='*60}")
            
            # Información adicional
            file_size = os.path.getsize(nombre_archivo_salida) / 1024  # KB
            # print(f"\nInformación del archivo:")
            # print(f"  - Tamaño: {file_size:.2f} KB")
            # print(f"  - Ubicación: {os.path.abspath(nombre_archivo_salida)}")
                    
            
        except Exception as e:
            print(f"\n✗ Error al guardar el archivo: {e}")
            sys.exit(1)


        return (nombre_archivo_salida,dict_tiempos)

    

    # calcula timestamps aproximados por palabra (no son exactos)
    # es la forma mas simple de calcular la posicion de una palabra basandose en
    # el largo del texto, el largo del audio y el largo de cada palabra
    # TODO hay mejores formas como usar whisper que usa metodo "forced alignment"
    # de momento la aproximacion es suficiente
    def obtener_diccionario_timestamps(self, texto: str, duracion_total: float) -> dict:
        """
        Calcula los timestamps estimados y los regresa en un diccionario estructurado.
        
        :param texto: El texto original enviado al TTS.
        :param duracion_total: Duración del audio (len(samples) / sample_rate).
        :return: Diccionario con formato {"palabra_N": {"palabra": "...", "time": [inicio, fin]}}
        """
        palabras = texto.split()
        total_caracteres = sum(len(p.strip("!,.:;?")) for p in palabras)
        
        tiempos = {}
        tiempo_actual = 0.0
        
        for indice, palabra in enumerate(palabras):
            palabra_limpia = palabra.strip("!,.:;?")
            
            if not palabra_limpia:
                continue
                
            # Regla de tres para el cálculo proporcional
            porcentaje_duracion = len(palabra_limpia) / total_caracteres
            duracion_estimada = duracion_total * porcentaje_duracion
            
            inicio = round(tiempo_actual, 3)
            fin = round(tiempo_actual + duracion_estimada, 3)
            
            # Estructura solicitada: llave única -> {palabra, time}
            clave_unica = f"palabra_{indice}"
            tiempos[clave_unica] = {
                "palabra": palabra_limpia,
                "time": [inicio, fin]
            }
            
            tiempo_actual += duracion_estimada
            
        # print(f"tiempos :: {tiempos}")
        return tiempos


# Creamos UNA ÚNICA INSTANCIA global dentro de este módulo
kokoro_tts_service = KokoroService()
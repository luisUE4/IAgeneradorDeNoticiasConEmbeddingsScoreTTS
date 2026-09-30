import os
import json
import logging
from datetime import datetime,timedelta
from filelock import FileLock
from typing import Dict, List

from pruebaKokoro import kokoro_tts_service
from ai_noticias.NoticiasFutbol import obtenerNoticias, obtenerListaDeTemasDeFutbol, generarNoticiasAIV2, obtenerTemasTrendAI
from database.db_manager import DatabaseManager
from database.embeddings_manager import EmbeddingsManager

# ==========================================
# CONFIGURACIÓN
# ==========================================

maximoDeNoticiasAlmacenadas = 30

# Configuración inicial
STATE_FILE = os.path.join(os.path.dirname(__file__), 'playlist_state.json')
LOCK_FILE = STATE_FILE + '.lock'

def configurar_logger(nombre_archivo):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] -> %(message)s')
    handlers = [logging.FileHandler(nombre_archivo, encoding='utf-8'), logging.StreamHandler()]
    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger

# Así lo creas en una sola línea en tu script específico:
logger = configurar_logger('directorNoticias.log')


def _load_state() -> Dict:
    """Carga el estado desde el archivo JSON"""
    lock = FileLock(LOCK_FILE)
    try:
        with lock:
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Archivo de estado corrupto, creando uno nuevo")
                    return {"playlistNoticiasFut": []}
            return {"playlistNoticiasFut": []}
    except Exception as e:
        logger.error(f"Error cargando estado: {str(e)}")
        return {"playlistNoticiasFut": []}


def _save_state(state: Dict) -> None:
    """Guarda el estado en el archivo JSON"""
    lock = FileLock(LOCK_FILE)
    try:
        with lock:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error guardando estado: {str(e)}")


def llenarPlaylistFut() -> None:
    """Llena la playlist con nuevas noticias"""

    logger.info("AI ********** llenarPlaylistFut()")
    state = _load_state()
    playlist = state["playlistNoticiasFut"]

    # Obtener temas variados
    temas = obtenerListaDeTemasDeFutbol(4).temas

    # Obtener noticias para cada tema
    for tema in temas:
        try:
            noticiaModel = obtenerNoticias("ultima semana", tema, 5 )
            for noti in noticiaModel.noticias :
                playlist.append(noti)
            
        except Exception as e:
            logger.warning(f"Error obteniendo noticia para tema '{tema}': {str(e)}")

    state["playlistNoticiasFut"] = playlist
    _save_state(state)


def obtenerSiguienteNoticia() -> Dict[str, str]:
    """Obtiene la siguiente noticia de la playlist"""
    logger.info("obtenerSiguienteNoticia()")
    state = _load_state()
    playlist = state["playlistNoticiasFut"]
    
    # Si hay menos de maximoDeNoticiasAlmacenadas noticias, llenar playlist
    if len(playlist) < maximoDeNoticiasAlmacenadas:
        logger.info (f"AI******* menos de {maximoDeNoticiasAlmacenadas} obteniendo nuevas")
        llenarPlaylistFut()

    state = _load_state()
    playlist = state["playlistNoticiasFut"]

    # Si no hay noticias disponibles
    if len(playlist) < 1:
        logger.warning("no se generaron noticias !")
        return {"noticia": "No hay noticias disponibles"}
        
    # Tomar y eliminar la noticia más vieja
    noticia = playlist.pop(0)
    _save_state(state)

    #es esperado que bloquee la ejecucion unos 20 segundos mientras convierte a voz
    voz = kokoro_tts_service.texto_a_voz_kokoro(noticia)

    resultado = {"noticia": noticia, "vozAudio":voz[0]}
    return resultado



def obtenerTemasTrending() -> Dict:
    """
    Obtiene temas trending de fútbol usando Gemini AI y los persiste en la base de datos.

    Flujo:
    1. Consulta a Gemini AI para obtener 15 temas trending (5 por ventana de tiempo).
    2. Para cada tema genera un embedding semántico.
    3. Busca en ChromaDB si ya existe un tema similar (umbral 0.85).
       - Si EXISTE: actualiza trending_score += 20 y ventana_de_tiempo en SQLite.
       - Si NO EXISTE: inserta el nuevo tema en SQLite y agrega su embedding a ChromaDB.
    4. Retorna un resumen de los temas procesados.

    Returns:
        Dict con temas_nuevos, temas_actualizados, total_procesados y timestamp.
    """
    logger.info("AI ********** obtenerTemasTrending()")

    resultados = {
        "status": "success",
        "temas_nuevos": [],
        "temas_actualizados": [],
        "errores": [],
        "total_procesados": 0,
        "timestamp": datetime.now().isoformat()
    }

    db_manager = None
    try:
        logger.info("Consultando Gemini AI para obtener temas trending...")

        temas_ai = obtenerTemasTrendAI()
        
        logger.info(f"AI devolvió {len(temas_ai.temas)} temas trending.")

        db_manager = DatabaseManager()
        embeddings_manager = EmbeddingsManager()

        # 3. Procesar cada tema
        for tema_obj in temas_ai.temas:
            try:
                logger.info(
                    f"Procesando tema: '{tema_obj.tema}' | "
                    f"trending score={tema_obj.trending_score} | "
                    f"ventana={tema_obj.ventana_de_tiempo}"
                )

                # Generar embedding semántico del tema
                embedding = embeddings_manager.generar_embedding(tema_obj.tema)

                # Buscar similares en ChromaDB (búsqueda GLOBAL, sin filtro de ventana)
                similares = embeddings_manager.buscar_similares(embedding, threshold=0.94, cantidad=5)                

                if len(similares)>0:
                    #  EXISTE tema similar → actualizar score +20 y ventana
                    tema_similar = similares[0]
                    tema_id = tema_similar["tema_id"]
                    tema_existente = tema_similar["tema"]
                    similitud = tema_similar["similitud"]

                    incremento_trending_score_al_actualizar = 10

                    db_manager.actualizar_tema(
                        tema_id=tema_id,
                        incremento_score=incremento_trending_score_al_actualizar,
                        nueva_ventana=tema_obj.ventana_de_tiempo
                    )

                    logger.info(
                        f"ACTUALIZADO: '{tema_obj.tema}' → similar a '{tema_existente}' \n"
                        f"              (similitud={similitud:.4f}, ID={tema_id}, score+{incremento_trending_score_al_actualizar})"
                    )
                    resultados["temas_actualizados"].append({
                        "tema_nuevo_ai": tema_obj.tema,
                        "tema_existente_bd": tema_existente,
                        "similitud": round(similitud, 4),
                        "tema_id": tema_id
                    })

                else:
                    # 3d. NO EXISTE → insertar nuevo tema en SQLite y agregar embedding a ChromaDB
                    tema_id = db_manager.insertar_tema(
                        tema=tema_obj.tema,
                        trending_score=tema_obj.trending_score,
                        ventana=tema_obj.ventana_de_tiempo
                    )

                    embeddings_manager.agregar_embedding(
                        tema_id=tema_id,
                        tema=tema_obj.tema,
                        embedding=embedding
                    )

                    logger.info(
                        f"NUEVO: '{tema_obj.tema}' insertado con "
                        f"      ID={tema_id}, score={tema_obj.trending_score}"
                    )
                    resultados["temas_nuevos"].append({
                        "tema": tema_obj.tema,
                        "trending_score": tema_obj.trending_score,
                        "ventana_de_tiempo": tema_obj.ventana_de_tiempo,
                        "tema_id": tema_id
                    })

                resultados["total_procesados"] += 1

            except Exception as e:
                error_msg = f"Error procesando tema '{tema_obj.tema}': {str(e)}"
                logger.error(error_msg)
                resultados["errores"].append(error_msg)

    except Exception as e:
        error_msg = f"Error general en obtenerTemasTrending: {str(e)}"
        logger.error(error_msg)
        resultados["status"] = "error"
        resultados["errores"].append(error_msg)

    finally:
        if db_manager:
            db_manager.cerrar_conexion()

    logger.info(
        f"obtenerTemasTrending() completado: "
        f"       {len(resultados['temas_nuevos'])} nuevos, "
        f"       {len(resultados['temas_actualizados'])} actualizados, "
        f"       {len(resultados['errores'])} errores."
    )
    return resultados



def obtenerTemasAlmacenadosSimilares(tema:str, cantidad:int = 5, similitud:float = 0.7)->List:
    embeddings_manager = EmbeddingsManager()
    embedding = embeddings_manager.generar_embedding(tema)
    similares = embeddings_manager.buscar_similares(embedding, threshold=similitud, cantidad=cantidad)
    return similares
    

def generarNoticiasV2():
    # Usa los temas almacenados ordenados por trending score
    # filtra los temas que no tengna noticias relacionadas
    # envia esos temas a AI para que genere las noticias
    # AI responde con las noticias y su identificador tema_id que yo le envie en el prompt para cada noticia
    # almacena en BD las nuevas noticias
    # 
    # la idea es que si un tema no tiene noticias probablemente es un tema que se incluyo recientemente por AI 
    # podria persibirse que no se crearan mas noticias de un mismo tema pero debido que el clasificador de temas
    #  no es perfecto existen temas que hablan de lo mismo porque que , si habra noticias del mismo tema y eso esta bien 

    embeddings_manager = EmbeddingsManager()
    db_manager = DatabaseManager()
    # el minimoTrendingScore sirve para ya crear noticias sin relevancia para la audiencia
    db_trending = db_manager.obtener_temas_con_mayor_trendingScore(minimoTrendingScore=10, cantidad=200)

    # filtrar identificando cuales de esos temas tienen menos noticias generadas
    temas_a_revisar = [elemento["tema_id"] for elemento in db_trending]
    temas_sin_noticias = db_manager.filtrar_temas_sin_noticias_DB(temas_a_revisar)
    

    modelo = generarNoticiasAIV2(temas_sin_noticias[:20])

    logger.info(f"********** generarNoticiasV2 , ai genero {len(modelo.bloque_noticias)} noticias")
    logger.info(f"")

    # guardar en BD
    for m in modelo.bloque_noticias:
        logger.info (f"     tema_id:{m.tema_id} , {m.texto}")
        db_manager.insertar_noticia(m.tema_id,m.texto)

    db_manager.cerrar_conexion()




def obtenerSiguienteNoticiaV2()->Dict[str, str]:

    logger.info("")
    logger.info(f"******************* obtenerSiguienteNoticiaV2 ")    
    logger.info("")

    # primero obtengo el tema mas trending que tengo 
    db_manager = DatabaseManager()
    tematop = db_manager.obtener_tema_top()

    noticias =  db_manager.obtener_noticias_mas_nuevas_y_sin_mencionar_por_algo_de_tiempo(1)  

    logger.info(f"tema_id: {noticias[0]["tema_id"]}  noticia: {noticias[0]["noticia"][:60]}.... ")
    logger.info("")    

    # marcar noticia como mencionada (actualiza su mencion timestamp)
    db_manager.actualizar_noticia_mencion_timestamp(noticias[0]["noticia_id"])

    #es esperado que bloquee la ejecucion unos 20 segundos mientras convierte a voz

    voz = [""]

    voz = kokoro_tts_service.texto_a_voz_kokoro(noticias[0]["noticia"])

    resultado = {"noticia":noticias[0]["noticia"]  , "vozAudio":voz[0]}

    # TODO eliminar noticias viejas de mas de 5 dias desde que se creo
    # db_manager.eliminar_noticias_viejas()
    
    # TODO eliminar temas que ya no son relevantes de la BD y de embeddings        

    return resultado



def limpiarNoticiasViejas(dias:int = 2):

    db_manager = DatabaseManager()
    db_manager.eliminar_noticias_viejas(dias)
    db_manager.cerrar_conexion()

def limpiarTemasViejos(dias:int = 2):
    db_manager = DatabaseManager()
    db_manager.eliminar_temas_viejos(dias)
    db_manager.cerrar_conexion()
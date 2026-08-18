import json
import os
from typing import Dict, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
import logging
from time import sleep
from pydantic import BaseModel, Field
from utilsAI import ConsultaAI
from database.db_manager import DatabaseManager

# Configuración inicial
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s] - %(message)s',
    handlers=[logging.FileHandler('noticias_futbol.log'),logging.StreamHandler()]
)

# Cliente Gemini
client = genai.Client()

modeloGenerativoAI = "gemini-2.5-flash"


class ModeloNoticias(BaseModel):
    noticias: List[str] = Field(description="Lista de noticias")


def obtenerNoticias(periododetiempo: str, tema: str, cantidad:int = 5 ) -> ModeloNoticias:
   

    logging.info(f"AI ******* obtenerNoticias({periododetiempo} - {tema})")
    

    prompt = (
        f"Genera una lista de {cantidad} noticias sobre '{tema}' "
        f"en el periodo de '{periododetiempo}'. "
        "Usa solo información de libre uso y dominio público. "
        "Responde únicamente con las noticias en formato json, "
        "sin comentarios ai introductorios o explicaciones adicionales."
        "Redacta con estilo de comentarista de futbol y apasionado "
        "que el largo de cada noticia sea de un parrafo "
        "cada noticia es independiente de las demas "
        "siempre menciona de quien o de que hablas"
    )

    respuestaAI = ConsultaAI(prompt, ModeloNoticias, 1 )    
    return respuestaAI

   


class ModeloTemas(BaseModel):
    temas: List[str] = Field(description="Lista de temas variados de fútbol y relacionados")


def obtenerListaDeTemasDeFutbol(cantidad: int = 20) -> ModeloTemas:
   
    logging.info("obtenerListaDeTemasDeFutbol()")

    prompt = (
        f"Genera una lista de {cantidad} temas de fútbol nacional e internacional asi como el mundial y temas fifa"
        "que podrían usarse para crear noticias. "
        "Solo incluye los títulos de los temas, "
        "sin descripciones ni explicaciones adicionales. "
        "Responde en formato JSON con la estructura: "
        "{{\"temas\": [\"tema1\", \"tema2\", ...]}}"
    )
    respuestaAI = ConsultaAI(prompt, ModeloTemas, 1, 0.7 )  

    return respuestaAI




# ─────────────────────────────────────────────────────────────────────────────
# Modelos para Temas Trending
# ─────────────────────────────────────────────────────────────────────────────

class TemaTrending(BaseModel):
    tema: str = Field(description="Tema de fútbol trending")
    trending_score: int = Field(description="Score de tendencia del 1 al 100", ge=1, le=100)
    ventana_de_tiempo: str = Field(
        description="Ventana temporal del tema: 'ultima_hora', 'ultima_semana' o 'ultimo_mes'"
    )


class ModeloTemasTrending(BaseModel):
    temas: List[TemaTrending] = Field(
        description="Lista de temas trending de fútbol de las 3 ventanas de tiempo"
    )



# TODO def obtenerTemasHistoricos() 
# TODO def obtenerDatosDeInteresDeComparativa()  #hoy mssi anoto 10 goles en un partido esto solo habia pasado en 1970 .....



def obtenerTemasTrendAI() -> ModeloTemasTrending:
    """
    Consulta a Gemini AI para obtener los temas más trending de fútbol
    en las 3 ventanas de tiempo: última hora, última semana y último mes.

    Returns:
        ModeloTemasTrending con lista de TemaTrending .
    """
    logging.info("AI ******* obtenerTemasTrendAI()")

    # obtener los temas mas trending almacenados localmente y excluirlos de la busqueda de AI
    promptExclusion =""

    db_manager = DatabaseManager()
    temasTrendAlmacenados = db_manager.obtener_temas_con_mayor_trendingScore(minimoTrendingScore=50, cantidad=15)

    if len(temasTrendAlmacenados) > 0 :
        temas_a_excluir = "\n -".join([row["tema"] for row in temasTrendAlmacenados])
        promptExclusion = ("excluye los siguientes temas (puedes incluir temas relacionados):\n"
                           f"-{temas_a_excluir}"
        )



    # prompt = (
    #     "Actúa como analista experto de tendencias en fútbol mundial. "
    #     "Tu tarea es identificar los temas más trending en 3 ventanas temporales distintas:\n\n"

    #     "1. ÚLTIMA HORA (últimas 24 horas): Identifica exactamente 10 temas sobre "
    #     "eventos en curso, partidos de hoy, lesiones recientes, declaraciones bomba, "
    #     "resultados de partidos, alineaciones, faltas, expulsiones, altercados.\n\n"

    #     "2. ÚLTIMA SEMANA (últimos 7 días): Identifica exactamente 20 temas sobre "
    #     "fichajes confirmados o rumores calientes, polémicas arbitrales, resultados "
    #     "eventos destacados, mejores jugadores, mejores directores, administracion de ligas,"
    #     "alineaciones, estadisticas de futbol, estadisticas de jugadores, estadios, boletos, conferencias de prensa,"
    #     "resultados de partidos, expulsados, altercados, noticias curiosas relacionadas a futbol,"
    #     "cambios de entrenador, rivalidades encendidas, compra de jugadores, cambios de presupuesto,"
    #     "eventos relacionados al futbol, noticias de gobiernos relacionadas a futbol, economia de futbol,"
    #     "celebraciones, noticias de fans."
    #     ".\n\n"

    #     "3. ÚLTIMO MES (últimos 30 días): Identifica exactamente 10 temas sobre "
    #     "tendencias más amplias, preparativos para torneos, rachas de equipos, "
    #     "estadísticas destacadas, movimientos de mercado de transferencias.\n\n"

    #     "Para cada tema asigna un trending_score del 1 al 100 basado en:\n"
    #     "- Impacto mediático y cobertura en medios deportivos\n"
    #     "- Engagement en redes sociales (Twitter/X, Instagram, TikTok)\n"
    #     "- Relevancia para aficionados hispanohablantes (España, México, Argentina)\n"
    #     "- Urgencia temporal del tema\n\n"

    #     "IMPORTANTE:\n"
    #     "- Enfócate en fútbol de mundial 2026, FIFA, España (LaLiga), México (Liga MX), Argentina (Liga Profesional), y ligas top europeas (Premier League, Serie A, Bundesliga, Ligue 1).\n"
    #     "- asegurate de no agregar temas que sean mas de 1 mes de viejas \n"
    #     "- Usa 'ultima_hora', 'ultima_semana' o 'ultimo_mes' como valor de ventana_de_tiempo.\n"
    #     "- Genera exactamente 40 temas en total\n"
    #     "- redacta cada tema de forma muy concisa hasta 8 palabras"
    #     "- siempre menciona al sujeto, entidad, evento o cosa , ejemplo de que no hacer : 'noticia impactante'  ejemplo de que si hacer: 'messi lesion'\n"
    #     "- Responde únicamente en formato JSON estricto, sin comentarios adicionales.\n\n"

    #     f"{promptExclusion}\n"
    # )

    prompt = (
        "Eres una reportera de noticias de deportes futbol y sociales "
        "Tu tarea es identificar las noticias y temas con más trending, \n"
        "Enfocate en temas del mundial de futbol 2026, \n"
        "ademas del torneo mismo tambien incluye noticias que sean sociales como noticias de fans, el viaje de las selecciones, \n"
        "estadios, ciudades anfitrionas, noticias relevantes al mundial desde punto de vista social, \n"
        "datos como donde se hospedaran las selecciones, cuando jugara cada equipo,  \n"
        "noticias de la inauguracion, estado de la ciudad que inaugura,  \n"
        "noticias de tecnologia en el mundial,  \n"
        "noticias de inteligencia artificial en el mundial,  \n"
        "noticias de los turistas que visitan el pais del mundial, \n"
        "noticias de las marchas y manifestaciones \n"
        "noticias de los ciudadanos y fans del mundial \n"
        "noticias de los aeropuetos y hoteles relevantes al mundial \n"
        "noticias de politica relevantes al mundial \n"        
        "noticias de oportunidad como comida/servicios gratis patrocinadas por el gobierno\n"        
        "noticias sobre el fanfest y servicios que los gobiernos proveen a los ciudadanos para disfrutar el mundial\n"        
        "noticias como transportes publicos gratuitos o similares \n"        
        "noticias que sean de interes parar los turistas que visitan las sedes del mundial\n"        
        "noticias que sean de interes parar los ciudadanos de la ciudad del mundial\n"        

        "Para cada tema adjunta un trending_score del 1 al 100 basado en:\n"
        "- Impacto mediático y cobertura en medios\n"
        "- Engagement en redes sociales (Twitter/X, Instagram, TikTok)\n"
        "- Relevancia para aficionados hispanohablantes (España, México, Argentina)\n"
        "- Urgencia temporal del tema\n\n"

        "IMPORTANTE:\n"
        "- asegurate de no agregar temas viejos (de mas de 3 dias)\n"
        "- Genera exactamente 40 temas en total\n"
        "- redacta cada tema de forma muy concisa hasta 8 palabras"
        "- siempre menciona al sujeto, entidad, evento o cosa , ejemplo de que no hacer : 'noticia impactante'  ejemplo de que si hacer: 'messi lesion'\n"
        "- Responde únicamente en formato JSON estricto, sin comentarios adicionales.\n\n"

        f"{promptExclusion}\n"
    )


    return ConsultaAI(prompt, ModeloTemasTrending, intentos=2, temperature=0.4)






#     prompt_creativo_ia = (
#     f"Actúa como un streamer de fútbol polémico, carismático y sumamente apasionado. "
#     f"Desarrolla los siguientes temas: {temas_bd}.\n\n"
    
#     f"INSTRUCCIONES DE ÉXITO PARA EL STREAM:\n"
#     f"1. ENFOQUE EN EL DRAMA: Evita dar solo estadísticas. Céntrate en las narrativas humanas: "
#     f"rivalidades, injusticias arbitrales, declaraciones picantes o momentos históricos.\n"
#     f"2. VARIACIÓN DE TONOS: Para cada bloque de noticias, adopta una de las siguientes actitudes: "
#     f"sé el defensor número uno, el crítico más despiadado o el analista táctico obsesivo.\n"
#     f"3. PROVOCACIÓN AL CHAT: Al final de cada tema, lanza una pregunta directa, incómoda o picante "
#     f"dirigida a la audiencia para obligarlos a debatir en el chat.\n\n"
    
#     f"Filtra las noticias usando nuestro historial para no repetir de qué hablamos hace poco:\n"
#     f"{historial_texto}"
# )

#     return ConsultaAI(prompt, ModeloPaqueteCompleto, intentos=1, temperature=0.4)



class NoticiaV2(BaseModel):
    tema_id: str = Field(description="El tema_id que se uso para crear la noticia")
    texto: str = Field(description="noticia sobre un tema")

class BloqueNoticiasV2(BaseModel):
    bloque_noticias: List[NoticiaV2] = Field(
        description="Lista de noticias"
    )


def generarNoticiasAIV2 (temas:List)->BloqueNoticiasV2:
    # recibe lista tipo [ [tema_id, tema], ....  ]

    
    totalTemas = len(temas)

    logging.info(f"generarNoticiasAIV2()  totalTemas {totalTemas}")

    if totalTemas == 0 :
        logging.error("ERROR obtenerNoticiasV2 () - no se recibieron temas a redactar")
        return BloqueNoticiasV2()

    temas_texto = ""
    for t in temas:        
        temas_texto += f"id:{t["tema_id"]}, tema:{t["tema"]} \n"

    prompt = (
        f"Actúa como reportera y comentarista de deportes\n"
        f"habla como si la audiencia ya te conociera y apreciara.\n"
        f"habla de manera y casual y amigable como lo hacen los comentaristas deportivos\n"
        f"Te proporciono {totalTemas} temas : \n{temas_texto}\n\n"
        
        f"Tu tarea consiste en desarrollar 20 noticias en total de esos temas (no 20 de cada uno)\n"
        f"Cuando detectes temas repetidos omite crear noticias del mismo tema (incluso si esto implica generar menos de 20 noticias)\n"
        f"en el caso que recibas mas de 20 temas elije al azar \n"
        f"en el caso que recibas menos de 20 temas crea la misma cantidad de noticias que de temas recibidos \n"
        
        f"REGLAS DE ORO:\n"
        f"- que el largo de cada noticia sea de un parrafo \n"
        f"- cada noticia es independiente de las demas \n"
        f"- cada noticia es de un parrafo de largo \n"
        f"- no generes noticias esten obsoletas o viejas ejemplo: el mundial sera en mexico (esto ya se sabe desde hace años)\n"
        f"- siempre menciona a los sujetos, entidades, eventos o cosas de la que se habla en la noticia \n"
        f"- Redacta con estilo de comentarista de fútbol apasionado y carismática incluso a veces un poco polemica sin ser ofensiva o exagerar\n"        
        f"Responde estrictamente en formato JSON.")
    

    return ConsultaAI(prompt, BloqueNoticiasV2, intentos=1, temperature=0.4)
        
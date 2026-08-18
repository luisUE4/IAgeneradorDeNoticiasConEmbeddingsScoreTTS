import sqlite3
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict

# Ruta de la base de datos SQLite
DB_PATH = os.path.join(os.path.dirname(__file__), 'futbol_temas.db')

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
logger = configurar_logger('db_manager.log')


class DatabaseManager:
    """Gestiona las operaciones CRUD sobre la base de datos SQLite de temas de fútbol."""

    def __init__(self):
        self.db_path = DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Permite acceder a columnas por nombre
        self._crear_tablas()
        logger.info(f"DatabaseManager inicializado. BD: {self.db_path}")

    def _crear_tablas(self):
        """Crea las tablas si no existen."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS temas (
                tema_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tema TEXT UNIQUE NOT NULL,
                trending_score INTEGER NOT NULL DEFAULT 0,
                ventana_de_tiempo TEXT NOT NULL,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS noticias (
                noticia_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tema_id INTEGER NOT NULL,
                noticia TEXT NOT NULL,
                ultima_mencion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tema_id) REFERENCES temas(tema_id)
            );
        """)
        self.conn.commit()
        logger.info("Tablas verificadas/creadas correctamente.")

    def insertar_tema(self, tema: str, trending_score: int, ventana: str) -> int:
        """
        Inserta un nuevo tema en la base de datos.

        Args:
            tema: Texto del tema de fútbol.
            trending_score: Score de tendencia (1-100).
            ventana: Ventana de tiempo ('ultima_hora', 'ultima_semana', 'ultimo_mes').

        Returns:
            tema_id del registro insertado.
        """
        try:
            cursor = self.conn.cursor()
            ahora = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO temas (tema, trending_score, ventana_de_tiempo, fecha_creacion, fecha_actualizacion)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tema, trending_score, ventana, ahora, ahora)
            )
            self.conn.commit()
            tema_id = cursor.lastrowid
            logger.info(f"Tema insertado: ID={tema_id}, tema='{tema}', score={trending_score}, ventana={ventana}")
            return tema_id
        except sqlite3.IntegrityError:
            logger.warning(f"El tema ya existe en BD (UNIQUE constraint): '{tema}'")
            # Retornar el ID existente
            cursor.execute("SELECT tema_id FROM temas WHERE tema = ?", (tema,))
            row = cursor.fetchone()
            return row["tema_id"] if row else -1
        except Exception as e:
            logger.error(f"Error insertando tema '{tema}': {str(e)}")
            raise

    def actualizar_tema(self, tema_id: int, incremento_score: int, nueva_ventana: str) -> bool:
        """
        Actualiza el trending_score (+incremento) y la ventana de tiempo de un tema existente.

        Args:
            tema_id: ID del tema a actualizar.
            incremento_score: Cantidad a sumar al trending_score actual.
            nueva_ventana: Nueva ventana de tiempo (la más reciente detectada por AI).

        Returns:
            True si se actualizó correctamente, False en caso contrario.
        """
        try:
            cursor = self.conn.cursor()
            ahora = datetime.now().isoformat()
            cursor.execute(
                """
                UPDATE temas
                SET trending_score = MIN(100, trending_score + ?),
                    ventana_de_tiempo = ?,
                    fecha_actualizacion = ?
                WHERE tema_id = ?
                """,
                (incremento_score, nueva_ventana, ahora, tema_id)
            )
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Tema actualizado: ID={tema_id}, score+{incremento_score}, ventana={nueva_ventana}")
                return True
            else:
                logger.warning(f"No se encontró tema con ID={tema_id} para actualizar.")
                return False
        except Exception as e:
            logger.error(f"Error actualizando tema ID={tema_id}: {str(e)}")
            raise

    def actualizar_noticia_mencion_timestamp(self, noticia_id: int) -> bool:
        """
        Actualiza el cambpo ultima_mencion_timestamp a el momento actual 

        Args:
            noticia_id: ID de noticia a actualizar
            
        Returns:
            True si se actualizó correctamente, False en caso contrario.
        """
        try:
            cursor = self.conn.cursor()
            ahora = datetime.now().isoformat()
            cursor.execute(
                """
                UPDATE noticias
                SET ultima_mencion_timestamp = ?
                WHERE noticia_id = ?
                """,
                (ahora, noticia_id)
            )
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Noticia mencion timestamp actualizada: ID={noticia_id}")
                return True
            else:
                logger.warning(f"No se encontró noticia con ID={noticia_id} para actualizar.")
                return False
        except Exception as e:
            logger.error(f"Error actualizando tema ID={noticia_id}: {str(e)}")
            raise


    def obtener_tema_por_id(self, tema_id: int) -> Optional[Dict]:
        """
        Obtiene un tema por su ID.

        Args:
            tema_id: ID del tema.

        Returns:
            Diccionario con los datos del tema, o None si no existe.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM temas WHERE tema_id = ?", (tema_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error obteniendo tema ID={tema_id}: {str(e)}")
            raise

    def obtener_todos_temas(self) -> List[Dict]:
        """
        Obtiene todos los temas ordenados por trending_score descendente.

        Returns:
            Lista de diccionarios con los datos de cada tema.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM temas ORDER BY trending_score DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error obteniendo todos los temas: {str(e)}")
            raise

    def obtener_temas_con_mayor_trendingScore(self, minimoTrendingScore:int=80, cantidad:int = 5) -> List[Dict]:
        """
        obtener_temas_con_mayor_trendingScore

        minimoTrendingScore sirve para descartar temas que ya no sean de interes

        Returns:
            Lista de diccionarios con los datos de cada tema.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM temas WHERE trending_score>{minimoTrendingScore} ORDER BY trending_score DESC LIMIT {cantidad}")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error obteniendo todos los temas: {str(e)}")
            raise

    def filtrar_temas_sin_noticias_DB(self, temasIDS:List) -> List[Dict]:
        #
        # revisa temas_id para encontrar los que no tengan noticias 
        # responde con una lista[ [temas_id,tema],... ] ordenados por trending_score
        # 
        logger.info(f"filtrar_temas_sin_noticias_DB -> temas (total {len(temasIDS)})->")
        for t in temasIDS:
            logger.info(f"      {t}")

        try:
            cursor = self.conn.cursor()

            placeholders = ",".join(["?"] * len(temasIDS))

            cursor.execute(
                f"""
                SELECT t.tema_id, t.tema
                FROM temas t
                LEFT JOIN noticias n ON t.tema_id = n.tema_id
                WHERE t.tema_id IN ({placeholders})
                GROUP BY t.tema_id
                HAVING COUNT(n.noticia_id) < 1
                ORDER BY t.trending_score DESC
                """,
                temasIDS,

            )
            temas_con_menos_de_una_noticia = [
                {"tema_id": fila[0], "tema": fila[1]} 
                for fila in cursor.fetchall()
            ]
            
            logger.info(f"filtrar_temas_sin_noticias_DB -> resultado sin noticias (total {len(temas_con_menos_de_una_noticia)})->")
            for t in temas_con_menos_de_una_noticia:
                logger.info(f"      {t}")

            return temas_con_menos_de_una_noticia
        except Exception as e:
            logger.error(f"Error obteniendo todos los temas: {str(e)}")
            raise

    def obtener_tema_top(self) -> Dict:
        """
        obtener_tema_top

        Returns:
            Lista de filas.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM temas ORDER BY trending_score DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error obtener_tema_top: {str(e)}")
            raise


    def insertar_noticia(self, tema_id: int, noticia: str) -> int:
        """
        Inserta una noticia asociada a un tema.

        Args:
            tema_id: ID del tema al que pertenece la noticia.
            noticia: Texto de la noticia.

        Returns:
            noticia_id del registro insertado.
        """
        try:
            cursor = self.conn.cursor()
            ahora = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO noticias (tema_id, noticia, ultima_mencion_timestamp, fecha_creacion)
                VALUES (?, ?, ?, ?)
                """,
                (tema_id, noticia, ahora, ahora)
            )
            self.conn.commit()
            noticia_id = cursor.lastrowid
            logger.info(f"Noticia insertada: ID={noticia_id}, tema_id={tema_id}")
            return noticia_id
        except Exception as e:
            logger.error(f"Error insertando noticia para tema_id={tema_id}: {str(e)}")
            raise

    def obtener_noticias_mas_nuevas_y_sin_mencionar_por_algo_de_tiempo(self, cantidad:int =1) -> List[Dict]:
        """
        Obtiene todas las noticias de un tema específico.

        Args:
            tema_id: ID del tema.

        Returns:
            Lista de diccionarios con los datos de cada noticia.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT * FROM noticias ORDER BY ultima_mencion_timestamp ASC, fecha_creacion DESC LIMIT {cantidad}"
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error obteniendo noticias : {str(e)}")
            raise

    def eliminar_noticias_viejas(self,dias:int =2):
        logger.info("eliminar_noticias_viejas")
        try:
            modificador_dias = f"-{dias} days"
            cursor = self.conn.cursor()
            cursor.execute(f"DELETE FROM noticias WHERE DATETIME(fecha_creacion) < DATETIME('now', '{modificador_dias}')"
                           )
        except Exception as e:
            logger.error(f"Error eliminando noticias viejas: {str(e)}")
            raise

    
    def eliminar_temas_viejos(self,dias:int =2):
        logger.info("eliminar_temas_viejos")
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"DELETE FROM temas WHERE DATETIME(fecha_creacion) < DATETIME('now', '-{dias} days') AND tema_id NOT IN (SELECT DISTINCT tema_id FROM noticias WHERE tema_id IS NOT NULL);")
        except Exception as e:
            logger.error(f"Error eliminando noticias viejas: {str(e)}")
            raise

    def cerrar_conexion(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            logger.info("Conexión a SQLite cerrada.")

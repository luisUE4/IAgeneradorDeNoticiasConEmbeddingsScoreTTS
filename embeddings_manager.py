import logging
import os
import warnings
from typing import List, Dict, Optional

# Suprimir warning de symlinks en Windows (no afecta funcionalidad)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# IMPORTANTE: sentence_transformers debe importarse ANTES que chromadb
# para evitar conflicto de versiones en Windows
from sentence_transformers import SentenceTransformer
import chromadb

# Ruta donde ChromaDB almacenará sus datos
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
COLLECTION_NAME = "temas_futbol"

# Modelo de embeddings multilingüe optimizado para español
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Umbral de similitud para considerar que dos temas son el mismo
SIMILARITY_THRESHOLD = 0.88

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    encoding='utf-8',
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s] -  %(message)s',
    handlers=[
        logging.FileHandler('director_noticias.log'),
        logging.StreamHandler()
    ]
)
# logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Gestiona los embeddings de temas de fútbol usando ChromaDB y sentence-transformers.
    Permite detectar temas similares/duplicados mediante búsqueda semántica.
    """

    def __init__(self):
        logging.info(f"Inicializando EmbeddingsManager con modelo: {EMBEDDING_MODEL}")

        # Cargar modelo de embeddings
        self.modelo = SentenceTransformer(EMBEDDING_MODEL)

        # Inicializar cliente ChromaDB con persistencia en disco
        chroma_path_abs = os.path.abspath(CHROMA_PATH)
        os.makedirs(chroma_path_abs, exist_ok=True)

        self.client = chromadb.PersistentClient(path=chroma_path_abs)

        # Obtener o crear la colección de temas
        self.coleccion = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Usar distancia coseno para similitud semántica
        )

        logging.info(
            f"ChromaDB inicializado en: {chroma_path_abs} | "
            f"Colección: '{COLLECTION_NAME}' | "
            f"Documentos existentes: {self.coleccion.count()}"
        )

    def generar_embedding(self, texto: str) -> List[float]:
        """
        Genera el vector de embedding para un texto dado.

        Args:
            texto: Texto del tema de fútbol.

        Returns:
            Lista de floats representando el embedding.
        """
        try:
            embedding = self.modelo.encode(texto, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error generando embedding para '{texto}': {str(e)}")
            raise

    def buscar_similares(
        self,
        embedding: List[float],
        threshold: float = SIMILARITY_THRESHOLD,
        cantidad:int = 1
    ) -> List[Dict]:
        """
        Busca temas similares en ChromaDB usando búsqueda global (sin filtros).
        La similitud se calcula con distancia coseno.

        Args:
            embedding: Vector de embedding del tema a buscar.
            threshold: Umbral mínimo de similitud (0.0 a 1.0). Default: 0.85.

        Returns:
            Lista de diccionarios con los temas similares encontrados.
            Cada dict contiene: {'tema_id': int, 'tema': str, 'similitud': float}
            Lista vacía si no hay similares por encima del umbral.
        """
        try:
            # Si la colección está vacía, no hay nada que buscar
            if self.coleccion.count() == 0:
                logging.info("Colección ChromaDB vacía, no hay similares.")
                return []

            # Buscar el vecino más cercano
            resultados = self.coleccion.query(
                query_embeddings=[embedding],
                n_results=cantidad,  
                include=["metadatas", "distances"]
            )

            if not resultados or not resultados["ids"] or not resultados["ids"][0]:
                return []

            encontrados = []

            # Iterar sobre todos los resultados devueltos por ChromaDB
            # ChromaDB con distancia coseno devuelve distancia (0=idéntico, 2=opuesto)
            # Convertir distancia coseno a similitud: similitud = 1 - (distancia / 2)
            for idx, distancia in enumerate(resultados["distances"][0]):
                similitud = 1.0 - (distancia / 2.0)

                # logging.info(
                #     f"Resultado {idx+1}/{len(resultados['distances'][0])}: "
                #     f"distancia={distancia:.4f}, similitud={similitud:.4f}, umbral={threshold}"
                # )

                # Solo agregar si cumple el umbral de similitud
                if similitud >= threshold:
                    metadata = resultados["metadatas"][0][idx]
                    tema_id = int(metadata["tema_id"])
                    tema_texto = metadata["tema"]

                    # logging.info(
                    #     f"[OK] Tema similar encontrado: ID={tema_id}, "
                    #     f"tema='{tema_texto}', similitud={similitud:.4f}"
                    # )

                    encontrados.append({
                        "tema_id": tema_id,
                        "tema": tema_texto,
                        "similitud": similitud
                    })
                # else:
                #     logging.info(
                #         f"[X] Resultado descartado (similitud {similitud:.4f} < umbral {threshold})"
                #     )

            # Resumen final
            if encontrados:
                logging.info(
                    f"Búsqueda completada: {len(encontrados)} tema(s) similar(es) encontrado(s) "
                    f"de {len(resultados['distances'][0])} resultado(s) evaluado(s)"
                )
            else:
                logging.info(
                    f"No se encontraron temas similares por encima del umbral {threshold}"
                )

            return encontrados

        except Exception as e:
            logging.error(f"Error buscando similares en ChromaDB: {str(e)}")
            raise

    def agregar_embedding(self, tema_id: int, tema: str, embedding: List[float]) -> bool:
        """
        Agrega un nuevo embedding a la colección ChromaDB.

        Args:
            tema_id: ID del tema en SQLite (usado como referencia cruzada).
            tema: Texto del tema (guardado en metadata).
            embedding: Vector de embedding del tema.

        Returns:
            True si se agregó correctamente.
        """
        try:
            doc_id = f"tema_{tema_id}"

            self.coleccion.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{
                    "tema_id": tema_id,
                    "tema": tema
                }],
                documents=[tema]  # Texto original para referencia
            )

            logging.info(
                f"Embedding agregado a ChromaDB: doc_id='{doc_id}', "
                f"tema='{tema}', total_docs={self.coleccion.count()}"
            )
            return True

        except Exception as e:
            logging.error(f"Error agregando embedding para tema_id={tema_id}: {str(e)}")
            raise

    def eliminar_embedding(self, tema_id: int) -> bool:
        """
        Elimina el embedding de un tema de la colección ChromaDB.

        Args:
            tema_id: ID del tema en SQLite.

        Returns:
            True si se eliminó correctamente.
        """
        try:
            doc_id = f"tema_{tema_id}"
            self.coleccion.delete(ids=[doc_id])
            logging.info(f"Embedding eliminado de ChromaDB: doc_id='{doc_id}'")
            return True
        except Exception as e:
            logging.error(f"Error eliminando embedding para tema_id={tema_id}: {str(e)}")
            raise

    def obtener_total_documentos(self) -> int:
        """Retorna el número total de embeddings almacenados en ChromaDB."""
        return self.coleccion.count()

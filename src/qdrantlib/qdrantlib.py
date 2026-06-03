import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from settings.helper import setting
from thread_safe.onceler import Onceler


def new_collection(client: QdrantClient, name: str):
    client.create_collection(
        collection_name=name,
        vectors_config={
            "bge-m3-dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bge-m3-sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )

class QdrantBGEM3:

    _client: QdrantClient
    _once: Onceler = None
    _sparse_model: BGEM3FlagModel = None
    _model_lock: threading.Lock = None
    _client_lock: threading.Lock = None

    def __init__(self, client: QdrantClient=None, model_cache_dir = None):
        self._once = Onceler()
        self._once.index_manager.new("collections")

        self._client = client
        if not client:
            self._client = QdrantClient(":memory:", verify=False)

        if model_cache_dir:
            os.environ["FASTEMBED_CACHE_PATH"] = model_cache_dir

        self._sparse_model = BGEM3FlagModel(
            model_name_or_path=setting("LocalAI", "sparse_model"),
            use_fp16=True)
        self._model_lock = threading.Lock()
        self._client_lock = threading.Lock()

    @property
    def client(self):
        with self._client_lock:
            return self._client

    def embed(self, documents):
        with self._model_lock:
            return self._sparse_model.encode(
                documents,
                batch_size=24,
                max_length=1024,
                return_dense=True,
                return_sparse=True
            )

    def create_collection_once(self, collection):
        def _new_once():
            try:
                new_collection(self.client, collection)
            except Exception:
                pass
        self._once.store_once("collections", collection, _new_once)

    def upload_points(self, collection, points):
        self.create_collection_once(collection)
        self.client.upload_points(
            collection_name=collection,
            points=points,
        )

    def query(self, collection: str, query_str: str, prefetch_limit=10, limit=10, dense_threshold=0.9, sparse_threshold=0.9):
        embeddings = self.embed([query_str])
        dense_vector = embeddings['dense_vecs'][0].tolist()
        sparse_data = embeddings['lexical_weights'][0]
        sparse_vector = models.SparseVector(
            indices=[int(token_id) for token_id in sparse_data.keys()],
            values=[float(weight) for weight in sparse_data.values()]
        )
        return self.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="bge-m3-dense",
                    limit=prefetch_limit,
                    score_threshold=dense_threshold
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="bge-m3-sparse",
                    limit=prefetch_limit,
                    score_threshold=sparse_threshold
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=limit
        )
@dataclass
class Document:
    id: Any = None
    payload: str = None
    payload_hash: str = None
    collection: str = None

def only_new_docs(client: QdrantClient, documents: list[Document]):
    collections_ids = defaultdict(list)
    for d in documents:
        collections_ids[d.collection].append(d)

    new_docs = []
    for category, category_docs in collections_ids.items():
        check = [d.id for d in category_docs]
        try:
            existing_points = client.retrieve(
                collection_name=category,
                ids=check,
                with_payload=False,
                with_vectors=False
            )
            existing_ids = {point.id for point in existing_points}
        except Exception:
            existing_ids = set()

        for doc in category_docs:
            if doc.tweak_id not in existing_ids:
                new_docs.append(doc)
    return new_docs

def embed_upload_documents(client: QdrantBGEM3, documents: list[Document]):
    new_docs = only_new_docs(client.client, documents=documents)
    if not new_docs:
        return

    embeddings = client.embed([d.payload for d in new_docs])
    collections = defaultdict(list)

    for idx, doc in enumerate(documents):
        dense_vector = embeddings['dense_vecs'][idx].tolist()
        sparse_data = embeddings['lexical_weights'][idx]
        sparse_vector = models.SparseVector(
            indices=[int(token_id) for token_id in sparse_data.keys()],
            values=[float(weight) for weight in sparse_data.values()]
        )
        collections[doc.collection].append(models.PointStruct(
            id=doc.id,
            vector={
                "bge-m3-dense": dense_vector,
                "bge-m3-sparse": sparse_vector
            },
            payload={"text": doc.payload}
        ))

    for k, v in collections.items():
        client.upload_points(
            collection=k,
            points=v,
        )
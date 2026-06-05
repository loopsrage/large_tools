import hashlib
import uuid
from concurrent.futures.thread import ThreadPoolExecutor

from application_controller.app_controller import SimpleApp
from fsspecc.base_fsspecfs.base_fsspecfs import FSBase
from index_queue.index_queue import ActionConfig, new_index_queue
from pfds.pdfm import PDFM, extract_text, extract_images
from queue_controller.queueData import QueueData
from thread_safe.index import Index
from thread_safe.tslist import TsList

from src.qdrantlib.qdrantlib import embed_upload_documents, Document


class PDFIndex:
    _index: Index = None
    _namespace: str = None

    def __init__(self):
        self._namespace = "PDFS"
        self._index = Index().new(self._namespace)

    def store_from_file(self, file: str):
        self.store_pdf(file, PDFM(pdf_path=file))

    def store_pdf(self, name: str, pdf: PDFM):
        self._index.store_in_index(self._namespace, name, pdf)

    def load_pdf(self, name) -> PDFM:
        return self._index.load_from_index(self._namespace, name)

    def list_pdfs(self):
        return list(self._index.range_index(self._namespace))

    def range_pdfs(self):
        yield from self._index.range_index(self._namespace)

def uuid5_from_args(namespace: str, *args):
    ns = uuid.UUID(namespace)
    return f"{uuid.uuid5(ns, "_".join([str(a) for a in args]))}"

def pdf_pipeline(client, chunk_size, batch_size, worker_count):
    _tsl = TsList()
    _namespace = "8c29e160-b99e-4e42-b05d-6c17eb7b9b94"
    def embed_upload_node(index, queue_data: QueueData):
        obj = queue_data.attribute("obj")
        try:
            embed_upload_documents(client, obj)
        except Exception as e:
            raise e

    def buffer_node(index, queue_data: QueueData):
        obj = queue_data.attributes("obj")
        _tsl.append(obj)
        if _tsl.count() > batch_size:
            index.enqueue("embed_upload_node", str(uuid.uuid4()), _tsl.to_list())
            _tsl.reset()

    def png_node(index, queue_data: QueueData):
        pass

    def text_node(index, queue_data: QueueData):
        key, obj = queue_data.attributes("key", "obj")
        step_size = chunk_size // 2
        offset = 0
        chunk_count = 0
        while offset < len(obj):
            chunk = obj[offset : offset + chunk_size]

            is_last_chunk = len(chunk) < chunk_size
            if is_last_chunk:
                padding_needed = chunk_size - len(chunk)
                chunk += b" " * padding_needed

            hsh = hashlib.md5(chunk, usedforsecurity=False).hexdigest()

            document_id = uuid5_from_args(_namespace, key, hsh)
            index.enqueue("buffer_node", document_id, Document(
                id=document_id,
                payload=chunk.decode("utf-8", errors="ignore"),
                payload_hash=hsh,
                collection="pdfs", metadata={
                    "file_name_page": key,
                    "chunk_number": chunk_count,
                }))

            if is_last_chunk:
                break
            offset += step_size
            chunk_count += 1

    return new_index_queue(worker_count, text_node, buffer_node, embed_upload_node, png_node)

class PDFIndexQueue(SimpleApp):
    pfi: PDFIndex = None

    def __init__(self, client, chunk_size, batch_size, worker_count):
        super().__init__(pdf_pipeline(client, chunk_size, batch_size, worker_count))
        self.pfi = PDFIndex()

    def extract(self, fs: FSBase, path: str):
        for files in fs.walk(path):
            for f in files:
                self.pfi.store_from_file(f"{path}/{f}")

        with ThreadPoolExecutor() as executor:
            for name, pdf in self.pfi.range_pdfs():
                extract_text(pdf, executor)
                extract_images(pdf, executor)

        for filename, value in pdf.range_data():
            fn = f"{name}_{filename}"
            if ".txt" in filename:
                self.action_queues.enqueue("text_action", fn, value)

            if ".png" in filename:
                self.action_queues.enqueue("png_node", fn, value)


import asyncio
import queue
import threading
from llama_cpp import Llama
from thread_safe.onceler import Onceler

one_time = Onceler()

class FastQueueStreamer:
    def __init__(self):
        self.token_queue = queue.Queue()

    def put(self, text):
        self.token_queue.put(text)

    def end(self):
        self.token_queue.put(None)


def query_llama_model(prompt, model, verbose: bool = False, from_pretrained_kwargs: dict = None, **model_kwargs):
    if not prompt:
        return None

    def load_model(m):
        def loader():
            llama_llm = Llama(
                model_path=m,
                n_ctx=131072,
                n_gpu_layers=-1,
                verbose=False,
                **(from_pretrained_kwargs or {})
            )
            return llama_llm, threading.Lock()
        return loader

    llm, model_lock = one_time.store_once("models", model, load_model(model))
    with model_lock:
        streamer = FastQueueStreamer()
        def generation_task():
            response_stream = llm(
                prompt,
                **model_kwargs,
            )
            for chunk in response_stream:
                text_chunk = chunk["choices"][0]["text"]
                streamer.put(text_chunk)
            streamer.end()

        generation_thread = threading.Thread(target=generation_task)
        generation_thread.start()

        full_response = ""
        while True:
            chunk_text = streamer.token_queue.get()
            if chunk_text is None:
                break
            full_response += chunk_text
            if verbose:
                print(chunk_text, end="", flush=True)

        generation_thread.join()
        return "<think>\n" + full_response

async def async_query_llama_model(*args, **kwargs):
    return await asyncio.to_thread(query_llama_model, *args, **kwargs)

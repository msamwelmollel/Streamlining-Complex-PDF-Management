"""Embedded Tables Retriever w/ Unstructured.IO."""

from llama_index import VectorStoreIndex
from llama_index.node_parser import UnstructuredElementNodeParser
from typing import Dict, Any
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.readers.file.flat_reader import FlatReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from pathlib import Path
from typing import Optional
import os
import pickle


class EmbeddedTablesUnstructuredRetrieverPack(BaseLlamaPack):
    """Embedded Tables + Unstructured.io Retriever pack.

    Use unstructured.io to parse out embedded tables from an HTML document, build
    a node graph, and then run our recursive retriever against that.

    **NOTE**: must take in a single HTML file.

    """

    def __init__(
        self,
        html_path: str,
        nodes_save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.reader = FlatReader()

        docs = self.reader.load_data(Path(html_path))
        
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("Deci/DeciLM-7B").encode
        )
        
        # set_global_tokenizer(
        #     AutoTokenizer.from_pretrained("microsoft/phi-2").encode
        # )
        
        set_global_tokenizer(
            AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1").encode
        )

        
        self.llm = LlamaCPP(
            # Optionally, you can pass in the URL to a GGML model to download it automatically
            model_url=None,
            # Set the path to a pre-downloaded model instead of model_url
            # model_path='./decilm-7b-uniform-gqa-q8_0.gguf',
            model_path='./mixtral-8x7b-instruct-v0.1.Q2_K.gguf',
            # model_path='./phi-2.Q8_0.gguf',
            temperature=0.0,
            max_new_tokens=2000, # Increasing to support longer responses
            context_window=8048, # Mistral7B has an 8K context-window
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": -1},
            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )
        
        # embed_model = LangchainEmbedding(
        #     HuggingFaceEmbeddings(model_name="thenlper/gte-large")
        #     ## you can select sentence transfomer embedding also
        #   # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # )
        
        # embed_model = LangchainEmbedding(
        #     HuggingFaceEmbeddings(model_name="thenlper/gte-large")
        #     ## you can select sentence transfomer embedding also
        #   # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # )
        
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="thenlper/gte-large")
            # HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
          # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )
        
        # service_context = ServiceContext.from_defaults(embed_model="local")
        # embed_model = HuggingFaceEmbedding(model_name="./TAKUKURU LLM/gte-large")
        # embed_model = HuggingFaceEmbedding(model_name="C:/Users/msamwelmollel/Documents/TAKUKURU LLM/gte-large")
        
        # embed_model = LangchainEmbedding(
        #     HuggingFaceEmbeddings(model_name="C:\\Users\\msamwelmollel\\Documents\\TAKUKURU LLM\\gte-large")
        #     ## you can select sentence transfomer embedding also
        #   # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # )
        
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=embed_model
        )
        
        # service_context = ServiceContext.from_defaults(
        #     llm=self.llm,
        #     embed_model="local:./embed_model"
        # )
        
        # service_context = ServiceContext.from_defaults(
        #     llm=self.llm,
        #     embed_model=embed_model
        # )
        
        set_global_service_context(service_context)

        self.node_parser = UnstructuredElementNodeParser()
        if nodes_save_path is None or not os.path.exists(nodes_save_path):
            self.node_parser.llm = self.llm
            raw_nodes = self.node_parser.get_nodes_from_documents(docs)
            pickle.dump(raw_nodes, open(nodes_save_path, "wb"))
        else:
            raw_nodes = pickle.load(open(nodes_save_path, "rb"))

        base_nodes, node_mappings = self.node_parser.get_base_nodes_and_mappings(
            raw_nodes
        )
        # construct top-level vector index + query engine
        vector_index = VectorStoreIndex(base_nodes)
        vector_retriever = vector_index.as_retriever(similarity_top_k=1)
        self.recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_mappings,
            verbose=True,
        )
        # self.query_engine = RetrieverQueryEngine.from_args(self.recursive_retriever)
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-1106-preview"))

        
        self.query_engine = RetrieverQueryEngine.from_args(
          self.recursive_retriever,
          service_context=service_context
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "node_parser": self.node_parser,
            "recursive_retriever": self.recursive_retriever,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)

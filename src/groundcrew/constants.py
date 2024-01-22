"""
"""
import ast

from chromadb.utils import embedding_functions

DEFAULT_EF = embedding_functions.DefaultEmbeddingFunction()

DEFAULT_COLLECTION_NAME = 'database'

CLASS_NODE_TYPE = ast.ClassDef
FUNCTION_NODE_TYPE = ast.FunctionDef

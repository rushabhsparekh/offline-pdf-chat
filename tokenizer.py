# run once in a separate script or terminal, then never again
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5").save_pretrained("./models/nomic-tokenizer")
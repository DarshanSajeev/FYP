from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("McGill-NLP/electra-medal")
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/electra-medal")
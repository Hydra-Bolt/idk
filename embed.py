from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_query(self, text: str):
        # Generate embeddings for a single string
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.detach().numpy().flatten()  # Flatten to 1D

    def embed_documents(self, texts: list[str]):
        # Generate embeddings for a list of strings
        return [self.embed_query(text) for text in texts]
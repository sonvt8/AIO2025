"""
Module tạo embeddings cho văn bản sử dụng transformer models.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class EmbeddingGenerator:
    """Class để tạo embeddings từ văn bản."""
    
    def __init__(self, config):
        """
        Khởi tạo EmbeddingGenerator.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Tải model và tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f'Sử dụng device: {self.device}')
        print(f'Model đã tải: {config.model_name}')
    
    def _average_pool(self, 
                     last_hidden_states: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Tính average pooling cho embeddings.
        
        Args:
            last_hidden_states: Hidden states từ model
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return (last_hidden.sum(dim=1) / 
                attention_mask.sum(dim=1)[..., None])
    
    def generate_embeddings(self, 
                          texts: List[str], 
                          prefix: str = "passage") -> np.ndarray:
        """
        Tạo embeddings cho danh sách văn bản.
        
        Args:
            texts: Danh sách văn bản
            prefix: Prefix cho văn bản (passage hoặc query)
            
        Returns:
            Array embeddings đã được normalize
        """
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="Đang tạo embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Thêm prefix cho hiệu suất retrieval tốt hơn
            batch_texts_with_prefix = [
                f"{prefix}: {text}" for text in batch_texts
            ]
            
            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts_with_prefix,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Chuyển sang device
            batch_dict = {
                k: v.to(self.device) for k, v in batch_dict.items()
            }
            
            # Tạo embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                batch_embeddings = self._average_pool(
                    outputs.last_hidden_state, 
                    batch_dict['attention_mask']
                )
                # Normalize embeddings
                batch_embeddings = F.normalize(
                    batch_embeddings, p=2, dim=1
                )
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def generate_query_embedding(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho một query text.
        
        Args:
            text: Văn bản cần tạo embedding
            
        Returns:
            Query embedding đã normalize
        """
        query_with_prefix = f"query: {text}"
        batch_dict = self.tokenizer(
            [query_with_prefix],
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        batch_dict = {
            k: v.to(self.device) for k, v in batch_dict.items()
        }
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            query_embedding = self._average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
        return query_embedding.cpu().numpy().astype('float32')

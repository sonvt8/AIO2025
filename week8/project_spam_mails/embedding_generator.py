"""
Module tạo embeddings cho văn bản sử dụng transformer models.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import joblib
import logging

class EmbeddingGenerator:
    """Class để tạo embeddings từ văn bản."""
    
    def __init__(self, config):
        """
        Khởi tạo EmbeddingGenerator.
        
        Args:
            config: Cấu hình hệ thống
            
        Raises:
            ValueError: Nếu cấu hình không hợp lệ
            Exception: Nếu lỗi khi tải model hoặc tokenizer
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Đường dẫn lưu model và tokenizer
        cache_dir = os.path.join('cache', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        model_cache_path = os.path.join(cache_dir, f"model_{self.config.model_name.replace('/', '_')}.joblib")
        tokenizer_cache_path = os.path.join(cache_dir, f"tokenizer_{self.config.model_name.replace('/', '_')}.joblib")
        
        # Kiểm tra và tải model từ cache nếu có
        if os.path.exists(model_cache_path) and os.path.exists(tokenizer_cache_path):
            try:
                self.model = joblib.load(model_cache_path)
                self.tokenizer = joblib.load(tokenizer_cache_path)
                self.model = self.model.to(self.device)
                self.model.eval()
                logging.info(f"Đã tải model và tokenizer từ cache: {model_cache_path}, {tokenizer_cache_path}")
            except Exception as e:
                logging.error(f"Lỗi khi tải model hoặc tokenizer từ cache: {str(e)}")
                self._load_new_model()
        else:
            self._load_new_model()
            # Lưu model và tokenizer vào cache
            try:
                joblib.dump(self.model, model_cache_path)
                joblib.dump(self.tokenizer, tokenizer_cache_path)
                logging.info(f"Đã lưu model và tokenizer vào cache: {model_cache_path}, {tokenizer_cache_path}")
            except Exception as e:
                logging.error(f"Lỗi khi lưu model hoặc tokenizer vào cache: {str(e)}")
        
        logging.info(f'Sử dụng device: {self.device}')
        logging.info(f'Model đã tải: {config.model_name}')
    
    def _load_new_model(self):
        """Tải model và tokenizer mới từ pretrained."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logging.error(f"Lỗi khi tải model hoặc tokenizer: {str(e)}")
            raise
    
    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Tính average pooling cho embeddings.
        
        Args:
            last_hidden_states: Hidden states từ model
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def generate_embeddings(self, texts: List[str], prefix: str = "passage") -> np.ndarray:
        """
        Tạo embeddings cho danh sách văn bản, với tùy chọn tải từ file nếu đã tồn tại.
        
        Args:
            texts: Danh sách văn bản
            prefix: Prefix cho văn bản (passage hoặc query)
            
        Returns:
            Array embeddings đã được normalize
            
        Raises:
            Exception: Nếu lỗi khi tạo hoặc lưu embeddings
        """
        # Đường dẫn lưu embeddings
        embeddings_dir = os.path.join('cache', 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_file = os.path.join(embeddings_dir, f"embeddings_{self.config.model_name.replace('/', '_')}.npy")
        
        # Nếu flag regenerate_embeddings là True, xóa cache nếu tồn tại
        if self.config.regenerate_embeddings:
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
                logging.info(f"Đã xóa embeddings cache cũ: {embeddings_file} (do flag regenerate_embeddings=True)")
        
        # Kiểm tra xem file embeddings đã tồn tại chưa
        if os.path.exists(embeddings_file):
            try:
                embeddings = np.load(embeddings_file)
                if embeddings.shape[0] == len(texts):
                    logging.info(f"Đã tải embeddings từ file: {embeddings_file}")
                    return embeddings
                else:
                    logging.warning(f"Kích thước embeddings trong file không khớp. Tạo mới embeddings.")
            except Exception as e:
                logging.error(f"Lỗi khi tải embeddings từ file {embeddings_file}: {str(e)}")
        
        # Tạo embeddings mới
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Đang tạo embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_texts_with_prefix = [f"{prefix}: {text}" for text in batch_texts]
            
            batch_dict = self.tokenizer(
                batch_texts_with_prefix,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                batch_embeddings = self._average_pool(
                    outputs.last_hidden_state, 
                    batch_dict['attention_mask']
                )
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Lưu embeddings vào file
        try:
            np.save(embeddings_file, embeddings)
            logging.info(f"Đã lưu embeddings vào file: {embeddings_file}")
        except Exception as e:
            logging.error(f"Lỗi khi lưu embeddings vào file {embeddings_file}: {str(e)}")
        
        return embeddings
    
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
        
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            query_embedding = self._average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
        return query_embedding.cpu().numpy().astype('float32')
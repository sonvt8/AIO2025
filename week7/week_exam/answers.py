import numpy as np
import pandas as pd
import logging
import os
import gdown
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def DownloadFile(
    file_id: str,
    output_dir: str = "dataset",
    filename: str = "default.csv"
) -> Optional[str]:
    """
    Tải file từ Google Drive bằng ID file.

    Tham số:
        file_id: ID file Google Drive
        output_dir: Thư mục để lưu file
        filename: Tên file đầu ra

    Trả về:
        Đường dẫn đến file đã tải hoặc None nếu thất bại
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        logger.info(f"✅ File đã tồn tại: {output_path}")
        return output_path

    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, output_path, quiet=False)
        logger.info(f"📥 Tải file thành công: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải file: {e}")
        return None
      
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def main():
  """Hàm thực thi chính."""
  file_path = DownloadFile(
      file_id="1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq",
      output_dir="dataset",
      filename="advertising.csv"
  )
  if file_path is None:
      logger.error("❌ Không thể tải file. Dừng chương trình.")
      return
      
  # Câu 1
  print("**"*20)
  print("Đáp án câu 1:")
  X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
  print("Mean : ", np.mean(X))
  
  # Câu 2
  print("**"*20)
  print("Đáp án câu 2:")
  X = [1, 5, 4, 4, 9, 13]
  print("Median : ", np.median(X))
  
  # Câu 3
  print("**"*20)
  print("Đáp án câu 3:")
  X = [ 171, 176, 155, 167, 169, 182]
  print("Standard Deviation: ", np.std(X))
  
  # Câu 4
  print("**"*20)
  print("Đáp án câu 4:")
  X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
  Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
  
  cov_matrix = np.cov(X, Y)
  cov_xy = cov_matrix[0, 1]
  print("Covariance X, Y:", cov_xy)
  
  corr_matrix = np.corrcoef(X, Y)
  corr_xy = corr_matrix[0, 1]
  print("Correlation X, Y:", corr_xy)
  
  # Câu 5
  print("**"*20)
  print("Đáp án câu 5:")
  data = pd.read_csv(file_path)
  print(data.head().to_string())
  print(data.info())
  print(data.describe())
  x = data['TV']
  y = data['Radio']
  corr_xy = np.corrcoef(x, y)[0][1]
  print(f"Correlation between TV and Sales: {round(corr_xy, 2)}")
  
  # Câu 6
  print("**"*20)
  print("Đáp án câu 6:")
  features = ['TV', 'Radio', 'Newspaper']
  for idx, f1 in enumerate(features):
    for idy, f2 in enumerate(features):
      x = data[features[idx]]
      y = data[features[idy]] 
      corr_xy = np.corrcoef(x, y)[0][1]
      print(f"Correlation between {f1} and {f2}: {round(corr_xy, 2)}")
      
  # Câu 7
  print("**"*20)
  print("Đáp án câu 7:")
  x = data['Radio']
  y = data['Newspaper']
  corr_xy = np.corrcoef(x, y)[0][1]
  print(f"Correlation between Radio and Newspaper: {round(corr_xy, 2)}")
  
  # Câu 8
  print("**"*20)
  print("Đáp án câu 8:")
  data_corr_coef = data.corr()
  print(data_corr_coef)

  # Câu 9
  print("**"*20)
  print("Đáp án câu 9:")
  # plt.figure(figsize=(10,8))
  # sns.heatmap(data_corr_coef, annot=True, fmt=".2f", linewidth=.5)
  # plt.show()
  
  # Câu 10
  print("**"*20)
  print("Đáp án câu 10:")
  vi_data_df = pd.read_csv('dataset/vi_text_retrieval.csv')
  context = vi_data_df['text']
  context = [doc.lower() for doc in context]
  tfidf_vectorizer = TfidfVectorizer()
  context_embedded = tfidf_vectorizer.fit_transform(context)
  print(context_embedded.toarray()[7][0])
  
  # Câu 11
  print("**"*20)
  print("Đáp án câu 11:")
  def tfidf_search(questions, tfidf_vectorizer, top_d=5):
    # Chuyển đổi question thành vector TF-IDF
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    cosine_scores = cosine_similarity(context_embedded, query_embedded).reshape((-1,))
    results = []

    # Sắp xếp giá trị cosine từ lớn đến nhỏ
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
      doc_score = {
        'id': idx,
        'cosine_score': cosine_scores[idx]
      }
      results.append(doc_score)
    return results

  question = vi_data_df.iloc[0]['question']
  results = tfidf_search(question, tfidf_vectorizer, top_d=5)
  print(f"giá trị cosine cao nhất là: {results[0]['cosine_score']}")
  print(vi_data_df.iloc[results[0]['id'], 2])
  
  # Câu 12
  print("**"*20)
  print("Đáp án câu 12:")
  def corr_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    corr_scores = np.corrcoef(
      query_embedded.toarray()[0],
      context_embedded.toarray()
    )
    corr_scores = corr_scores[0][1:]
    
    # Sắp xếp giá trị correlation từ lớn đến nhỏ
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
      doc = {
        'id': idx,
        'corr_score': corr_scores[idx]
      }
      results.append(doc)
    return results 

  question = vi_data_df.iloc[0]['question']
  results = corr_search(question, tfidf_vectorizer, top_d=5)
  print(f"giá trị correlation thứ 2 là: {results[1]['corr_score']}")
  
  # Câu 13
  print("**"*20)
  print("Đáp án câu 13:")
  docs = [
    "Học máy là một nhánh của trí tuệ nhân tạo",
    "Trí tuệ nhân tạo bao gồm học máy và mạng nơ ron",
    "Mạng nơ ron là một mô hình quan trọng trong học sâu",
    "Học sâu là một lĩnh vực của trí tuệ nhân tạo và học máy"
  ]
  def build_vocabulary(docs):
    vocab_set = set()
    for doc in docs:
      words = doc.lower().split()
      vocab_set.update(words)
      vocab = sorted(list(vocab_set))
    return vocab

  vocab = build_vocabulary(docs)
  print(len(vocab))
  print(vocab[:10])
  
  # Câu 14
  print("**"*20)
  print("Đáp án câu 14 đến 18:")
  def compute_tf(doc, vocabulary):
      words = doc.lower().split()
      word_count = {}
      for word in words:
          word_count[word] = word_count.get(word, 0) + 1

      tf = np.zeros(len(vocabulary))
      for i, term in enumerate(vocabulary):
          tf[i] = word_count.get(term, 0) / len(words)
      return tf

  def compute_idf(docs, vocabulary):
      num_docs = len(docs)
      idf = np.zeros(len(vocabulary))
      for i, word in enumerate(vocabulary):
          doc_freq = sum(1 for doc in docs if word in doc.lower().split())
          idf[i] = np.log((num_docs + 1) / (1 + doc_freq)) + 1
      return idf
    
  def cosine(vec1, vec2):
    """Tính độ tương đồng cosine giữa hai vector"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
  
  def check_plagiarism_all(docs, tf_idf_matrix):
    """Tính độ tương đồng giữa tất cả các cặp văn bản"""
    num_docs = len(docs)

    similarity_matrix = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            similarity_matrix[i, j] = cosine(tf_idf_matrix[i], tf_idf_matrix[j])

    return similarity_matrix
    
  idf = compute_idf(docs, vocab)
  # Tính tf cho từng doc vào 1 ma trận
  tf = np.zeros((len(docs), len(vocab)))
  for idx, doc in enumerate(docs):
      tf[idx, :] = compute_tf(doc, vocab)

  tf_idf_matrix = tf * idf
  similarity_matrix = check_plagiarism_all(docs, tf_idf_matrix)
  print(tf_idf_matrix)
  print(similarity_matrix)
  plt.figure(figsize=(10,8))
  sns.heatmap(similarity_matrix , 
              annot =True,
              fmt='.3f',
              cmap ='YlOrRd')
  plt.show()
  
if __name__ == "__main__":
    main()
"""Trợ lý RAG với khả năng mở rộng truy vấn."""
import hashlib
import logging
import os
import json
from typing import Any, Dict, List, Tuple

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama

# Tải biến môi trường
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
USERS = json.loads(os.getenv("USERS", "{}"))

# Thiết lập ghi log
logging.basicConfig(
    level=logging.INFO,
    filename="rag_system.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def display_message(message: str, message_type: str = "error") -> None:
    """Hiển thị thông báo trên giao diện Streamlit với màu sắc theo loại.

    Args:
        message: Nội dung thông báo.
        message_type: Loại thông báo ("error", "warning", "info").
    """
    if "message_placeholder" not in st.session_state:
        st.session_state.message_placeholder = st.empty()
    if "last_message" not in st.session_state or st.session_state.last_message != message:
        st.session_state.last_message = message
        message_id = f"message_{hashlib.md5(message.encode()).hexdigest()}"
        color = {
            "error": "#ff4b4b",
            "warning": "#ffcc00",
            "info": "#28a745",
        }.get(message_type, "#ff4b4b")
        html_message = f"""
        <div id="{message_id}" style="padding: 10px; margin-bottom: 10px; border-radius: 5px; 
        color: white; background-color: {color};">
            {message}
        </div>
        <script>
            setTimeout(function() {{
                var elem = document.getElementById("{message_id}");
                if (elem) {{
                    elem.parentNode.removeChild(elem);
                }}
            }}, 5000);
        </script>
        """
        st.session_state.message_placeholder.markdown(html_message, unsafe_allow_html=True)

def authenticate_user(username: str, password: str) -> bool:
    """Kiểm tra thông tin đăng nhập của người dùng.

    Args:
        username: Tên người dùng.
        password: Mật khẩu.

    Returns:
        bool: True nếu xác thực thành công, False nếu thất bại.
    """
    if not USERS:
        logger.error("Không tìm thấy thông tin người dùng trong .env")
        display_message("Lỗi hệ thống: Không tìm thấy thông tin người dùng", message_type="error")
        return False
    if username in USERS and USERS[username] == password:
        logger.info(f"Xác thực thành công cho người dùng: {username}")
        display_message(f"Chào {username}", message_type="info")
        return True
    logger.warning(f"Xác thực thất bại cho người dùng: {username}")
    return False
  
class EmbeddingModel:
    """Mô hình Embedding để tạo vector từ văn bản."""

    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_function = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY, model="text-embedding-3-small"
            )
        elif model_type == "huggingface":
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="bkai-foundation-models/vietnamese-bi-encoder"
            )


class ChromaDBManager:
    """Quản lý cơ sở dữ liệu Chroma để lưu trữ vector."""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH, embedding_function=None):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_function or HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder"
        )
        try:
            # Tạo hoặc lấy bộ sưu tập
            self.collection = self.client.get_or_create_collection(name="rag_docs")
            doc_count = self.collection.count()
            if doc_count > 0:
                logger.info(f"Khởi tạo ChromaDB với {doc_count} tài liệu đã tồn tại")
                display_message(
                    f"Đã thiết lập ChromaDB với {doc_count} tài liệu đã tồn tại",
                    message_type="info"
                )
            else:
                logger.info("Khởi tạo ChromaDB mới (rỗng)")
                display_message(
                    "Khởi tạo ChromaDB mới (rỗng)",
                    message_type="info"
                )
        except Exception as e:
            logger.error(f"Lỗi khởi tạo ChromaDB: {str(e)}")
            display_message(f"Lỗi khởi tạo ChromaDB: {str(e)}", message_type="error")
            raise


    def create_vector_store(self, documents: List[Document] = None) -> Chroma:
        """Tạo vector store từ tài liệu hoặc tải từ bộ nhớ.

        Args:
            documents: Danh sách tài liệu để tạo vector store (mặc định: None).

        Returns:
            Chroma: Đối tượng vector store.
        """
        if documents:
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                client=self.client,
                collection_name="rag_docs",
            )
        return Chroma(
            client=self.client,
            collection_name="rag_docs",
            embedding_function=self.embedding_function,
        )


class LLMModel:
    """Mô hình ngôn ngữ lớn để tạo câu trả lời."""

    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        else:
            self.client = ChatOllama(model="llama3.2", temperature=0.1)


class DocumentProcessor:
    """Xử lý tài liệu: tải và chia nhỏ thành các đoạn."""

    def __init__(self, embedding_type):
        self.semantic_splitter = SemanticChunker(
            embeddings=embedding_type,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True,
        )

    def load_files(self, files_path: str) -> List[Document]:
        """Tải tài liệu từ một thư mục.

        Args:
            files_path: Đường dẫn đến thư mục chứa tài liệu.

        Returns:
            List[Document]: Danh sách các tài liệu đã tải.
        """
        try:
            loaders = []
            if not os.path.exists(files_path):
                st.error(f"Thư mục {files_path} không tồn tại")
                return []
            for filename in os.listdir(files_path):
                file_path = os.path.join(files_path, filename)
                try:
                    if filename.endswith(".txt"):
                        loaders.append(TextLoader(file_path))
                    elif filename.endswith(".md"):
                        loaders.append(UnstructuredMarkdownLoader(file_path))
                    elif filename.endswith(".pdf"):
                        loaders.append(PyPDFLoader(file_path))
                    elif filename.endswith(".csv"):
                        loaders.append(CSVLoader(file_path))
                except Exception as e:
                    print(f"Lỗi khi tải {filename}: {e}")
            merged_loader = MergedDataLoader(loaders=loaders)
            documents = merged_loader.load()
            # st.success(f"Đã tải lên {len(documents)} trang tài liệu")
            display_message(f"Đã tải lên {len(documents)} trang tài liệu", message_type="info")
            return documents
        except Exception as e:
            st.error(f"Lỗi khi tải tài liệu: {str(e)}")
            return []


    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Chia nhỏ tài liệu thành các đoạn.

        Args:
            documents: Danh sách tài liệu cần chia nhỏ.

        Returns:
            List[Document]: Danh sách các đoạn tài liệu.
        """
        try:
            chunks = self.semantic_splitter.split_documents(documents)
            display_message(f"Đã chia thành {len(chunks)} đoạn", message_type="info")
            return chunks
        except Exception as e:
            st.error(f"Lỗi khi chia tài liệu: {str(e)}")
            return documents


class QueryExpansionRAG:
    """Hệ thống RAG với khả năng mở rộng truy vấn."""

    def __init__(self, documents=None, embeddings=None, chroma_manager=None):
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder"
        )
        if chroma_manager:
            self.chroma_manager = chroma_manager
            if documents:
                vector_store = self.chroma_manager.create_vector_store(documents)
            else:
                vector_store = self.chroma_manager.create_vector_store()
        else:
            vector_store = Chroma.from_documents(documents, self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}, search_type="similarity"
        )


    def retrieve_with_expansion(self, question: str) -> List[Document]:
        """Tìm kiếm tài liệu liên quan đến câu hỏi.

        Args:
            question: Câu hỏi cần tìm kiếm.

        Returns:
            List[Document]: Danh sách tài liệu liên quan.
        """
        return self.retriever.invoke(question)


class AnswerGenerator:
    """Tạo câu trả lời từ nhiều nguồn tài liệu với trích dẫn."""

    def __init__(self, llm_type: str, temperature: float = 0):
        self.llm = LLMModel(llm_type)
        self.answer_generation_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là một trợ lý thông minh, có nhiệm vụ cung cấp câu trả lời chi tiết 
            dựa trên nhiều nguồn tài liệu. Mục tiêu là tổng hợp thông tin chính xác và trả lời có cấu trúc.

            Câu hỏi: {question}
            
            Dưới đây là các đoạn trích từ các tài liệu, mỗi đoạn có ID trích dẫn:
            {formatted_context}

            Vui lòng cung cấp câu trả lời chi tiết:
            1. Trả lời trực tiếp câu hỏi
            2. Sử dụng trích dẫn cụ thể theo định dạng [CitationID]
            3. Tổng hợp thông tin từ nhiều nguồn khi phù hợp
            4. Nêu rõ bất kỳ mâu thuẫn nào giữa các nguồn
            5. Trích dẫn đoạn văn bản từ nguồn khi cần thiết

            Định dạng câu trả lời:

            TRẢ LỜI TRỰC TIẾP:
            [Câu trả lời ngắn gọn với trích dẫn]

            GIẢI THÍCH CHI TIẾT:
            [Giải thích chi tiết với trích dẫn và đoạn trích khi cần]

            ĐIỂM CHÍNH:
            - [Điểm 1 với trích dẫn]
            - [Điểm 2 với trích dẫn]
            - [Điểm 3 với trích dẫn]

            NGUỒN TRÍCH DẪN:
            [Liệt kê các ID trích dẫn và đóng góp chính]

            Câu trả lời:""",
        )

    def _prepare_citation_chunks(
        self, documents: List[Document], max_chunk_length: int = 500
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """Chuẩn bị ngữ cảnh với trích dẫn và tạo bản đồ trích dẫn.

        Args:
            documents: Danh sách tài liệu liên quan.
            max_chunk_length: Độ dài tối đa cho mỗi đoạn tài liệu.

        Returns:
            Tuple[str, Dict]: Ngữ cảnh đã định dạng và bản đồ trích dẫn.
        """
        citation_id = 1
        citation_chunks = []
        citation_map = {}
        for doc in documents:
            content = doc.page_content
            truncated_content = content[:max_chunk_length]
            if len(content) > max_chunk_length:
                truncated_content += "..."
            citation_ref = f"[Citation{citation_id}]"
            citation_chunks.append(f"{citation_ref}:\n{truncated_content}\n")
            citation_map[citation_ref] = {"content": truncated_content, "full_content": content}
            citation_id += 1
        formatted_context = "\n".join(citation_chunks)
        return formatted_context, citation_map


    def generate_answer(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Tạo câu trả lời từ kết quả tìm kiếm với trích dẫn.

        Args:
            question: Câu hỏi gốc.
            documents: Danh sách tài liệu liên quan.

        Returns:
            Dict[str, Any]: Câu trả lời và thông tin trích dẫn.
        """
        try:
            formatted_context, citation_map = self._prepare_citation_chunks(documents)
            response = self.llm.client.invoke(
                self.answer_generation_prompt.format(
                    question=question, formatted_context=formatted_context
                )
            )
            return {
                "answer": response.content,
                "citations": citation_map,
                "formatted_context": formatted_context,
            }
        except Exception as e:
            st.error(f"Lỗi khi tạo câu trả lời: {str(e)}")
            return {
                "answer": "Không thể tạo câu trả lời do lỗi.",
                "citations": {},
                "formatted_context": "",
            }


def main() -> None:
    """Hàm chính để chạy ứng dụng Streamlit."""
    st.set_page_config(
      page_title="Trợ lý RAG",
      layout="wide",
      initial_sidebar_state="expanded",
    )

    # Khởi tạo trạng thái phiên
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "chroma_manager" not in st.session_state:
        st.session_state.chroma_manager = None
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    if "embedding_function" not in st.session_state:
        st.session_state.embedding_function = None
    if "is_processing_documents" not in st.session_state:
        st.session_state.is_processing_documents = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Trang đăng nhập
    if not st.session_state.is_authenticated:
        st.title("Đăng nhập vào Trợ lý RAG")
        st.markdown("Vui lòng nhập thông tin đăng nhập để truy cập hệ thống.")

        username = st.text_input("Tên người dùng", placeholder="Nhập tên người dùng")
        password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu")

        if st.button("Đăng nhập"):
            if authenticate_user(username, password):
                st.session_state.is_authenticated = True
                st.rerun()  # Tải lại trang để chuyển sang giao diện chính
            else:
                st.error("Đăng nhập thất bại. Vui lòng thử lại.")

        return  # Thoát hàm nếu chưa đăng nhập

    # Giao diện chính sau khi đăng nhập
    st.title("Trợ lý RAG")

    # Khởi tạo đường dẫn với đường dẫn tuyệt đối
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_directory = os.path.join(current_dir, "documents")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(files_directory, exist_ok=True)

    # Hiển thị thông tin hệ thống
    st.sidebar.title("Thông tin Hệ thống")
    if st.session_state.chroma_manager is not None:
        doc_count = st.session_state.chroma_manager.collection.count()
        st.sidebar.markdown(f"**Số tài liệu trong ChromaDB**: {doc_count}")
    else:
        st.sidebar.markdown("**Số tài liệu trong ChromaDB**: Chưa khởi tạo")

    # Chọn mô hình
    llm_type = st.sidebar.radio(
        "Chọn Mô hình LLM:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama Llama2",
    )
    embedding_type = st.sidebar.radio(
        "Chọn Mô hình Embedding:",
        ["openai", "huggingface"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "huggingface": "vietnamese-bi-encoder",
        }[x],
    )

    # Điều khiển sidebar
    st.sidebar.title("Điều khiển")
    st.sidebar.subheader("Tài liệu đã tải")
    if os.path.exists(files_directory):
        existing_files = [f for f in os.listdir(files_directory) if f.endswith(('.pdf', '.txt', '.md', '.csv'))]
        if existing_files:
            for file_name in existing_files:
                st.sidebar.markdown(f"- {file_name}")
        else:
            st.sidebar.info("Chưa có tài liệu nào trong thư mục.")
    else:
        st.sidebar.warning("Thư mục documents chưa tồn tại.")

    uploaded_files = st.sidebar.file_uploader(
        "Tải lên tệp", type=["pdf", "txt", "md", "csv"], accept_multiple_files=True
    )
    if uploaded_files:
        st.sidebar.success(f"Đã tải lên {len(uploaded_files)} tệp")
        for uploaded_file in uploaded_files:
            with open(os.path.join(files_directory, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())

    # Khởi tạo EmbeddingModel
    if st.session_state.embedding_function is None:
        embeddings = EmbeddingModel(embedding_type)
        st.session_state.embedding_function = embeddings.embedding_function
        logger.info(f"Khởi tạo embedding function: {embedding_type}")

    # Khởi tạo ChromaDBManager
    if st.session_state.chroma_manager is None:
        try:
            st.session_state.chroma_manager = ChromaDBManager(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=st.session_state.embedding_function,
            )
            st.session_state.doc_count = st.session_state.chroma_manager.collection.count()
        except Exception as e:
            logger.error(f"Lỗi khởi tạo ChromaDB: {str(e)}")
            display_message(f"Lỗi khởi tạo ChromaDB: {str(e)}", message_type="error")
            return

    # Xử lý tài liệu khi nhấn nút
    if st.sidebar.button("Xử lý Tài liệu"):
        if st.session_state.is_processing_documents:
            display_message("Đang xử lý tài liệu, vui lòng đợi!", message_type="warning")
            return
        if st.session_state.embedding_function is None:
            display_message("Chưa khởi tạo mô hình embedding!", message_type="error")
            return
        st.session_state.is_processing_documents = True
        with st.spinner("Đang xử lý tài liệu..."):
            try:
                doc_processor = DocumentProcessor(st.session_state.embedding_function)
                documents = doc_processor.load_files(files_directory)
                if documents:
                    splits = doc_processor.split_documents(documents)
                    st.session_state.rag_system = QueryExpansionRAG(
                        documents=splits,
                        embeddings=st.session_state.embedding_function,
                        chroma_manager=st.session_state.chroma_manager,
                    )
                    st.session_state.doc_count = (
                        st.session_state.chroma_manager.collection.count()
                    )
                    display_message("Xử lý tài liệu thành công!", message_type="info")
                else:
                    display_message("Không tìm thấy tài liệu", message_type="warning")
            except Exception as e:
                logger.error(f"Lỗi xử lý tài liệu: {str(e)}")
                display_message(f"Lỗi xử lý tài liệu: {str(e)}", message_type="error")
            finally:
                st.session_state.is_processing_documents = False

    # Giao diện truy vấn chính
    st.header("Giao diện Truy vấn")
    query = st.text_input("Nhập câu hỏi của bạn:")
    k = st.slider("Số kết quả trả về", min_value=1, max_value=5, value=3)

    # Nút tìm kiếm
    if st.button("Tìm kiếm"):
        if query:
            if st.session_state.rag_system is None:
                display_message("Vui lòng xử lý tài liệu trước!", message_type="warning")
                return
            if st.session_state.is_processing:
                display_message("Đang xử lý truy vấn, vui lòng đợi!", message_type="warning")
                return
            st.session_state.is_processing = True
            try:
                with st.spinner("Đang xử lý truy vấn..."):
                    st.session_state.rag_system.retriever.search_kwargs["k"] = k
                    answer_generator = AnswerGenerator(llm_type)
                    logger.info(f"Xử lý truy vấn: {query}")
                    docs = st.session_state.rag_system.retrieve_with_expansion(query)
                    if not docs:
                        display_message("Không tìm thấy tài liệu liên quan!", message_type="warning")
                        st.session_state.is_processing = False
                        return
                    logger.info(f"Tìm thấy {len(docs)} tài liệu liên quan")
                    
                    st.subheader("📝 Phân tích Chi tiết")
                    with st.spinner("Đang tạo câu trả lời chi tiết..."):
                        response_data = answer_generator.generate_answer(query, docs)
                        if not response_data["answer"]:
                            display_message("Không thể tạo câu trả lời!", message_type="error")
                            st.session_state.is_processing = False
                            return
                        st.markdown(response_data["answer"])
                        st.subheader("📚 Nguồn Trích dẫn")
                        for citation_id, citation_data in response_data["citations"].items():
                            with st.expander(f"{citation_id} - Nhấn để xem nguồn"):
                                st.markdown("**Đoạn trích:**")
                                st.markdown(f"```\n{citation_data['content']}\n```")
                    
                    with st.expander("🔎 Xem Tất cả Kết quả Tìm kiếm"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Truy vấn:** {query}")
                            st.markdown(f"*Tài liệu {i}:*")
                            st.markdown(f"```\n{doc.page_content[:500]}...\n```")
                            st.markdown("---")
                    
                    st.subheader("🎯 Câu trả lời Cuối cùng")
                    with st.spinner("Đang tổng hợp câu trả lời cuối cùng..."):
                        final_prompt = PromptTemplate(
                            input_variables=["question", "detailed_answer"],
                            template="""Dựa trên phân tích chi tiết, tạo câu trả lời rõ ràng, ngắn gọn 
                            cho câu hỏi gốc. Tập trung vào các điểm quan trọng và đảm bảo tính chính xác.

                            Câu hỏi Gốc: {question}

                            Phân tích Chi tiết:
                            {detailed_answer}

                            Câu trả lời cuối cùng phải:
                            1. Trả lời trực tiếp câu hỏi
                            2. Tóm tắt các điểm chính
                            3. Rõ ràng và ngắn gọn
                            4. Giữ các trích dẫn quan trọng

                            Câu trả lời Cuối cùng:""",
                        )
                        llm_model = LLMModel(llm_type)
                        final_response = llm_model.client.invoke(
                            final_prompt.format(
                                question=query, detailed_answer=response_data["answer"]
                            )
                        )
                        st.markdown("---")
                        st.markdown("### 💡 Tóm tắt")
                        st.markdown(
                            f"""
                            <div style='background-color: #f0f2f6; padding: 20px; 
                            border-radius: 10px;'>
                            {final_response.content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                logger.error(f"Lỗi khi xử lý truy vấn: {str(e)}")
                display_message(f"Lỗi hệ thống: {str(e)}", message_type="error")
            finally:
                st.session_state.is_processing = False
        else:
            display_message("Vui lòng nhập câu hỏi", message_type="warning")


if __name__ == "__main__":
    main()

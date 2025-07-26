"""
Module để xử lý email qua Gmail API: fetch, classify, và move vào labels.
"""
import os
import base64
import logging
import json
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from spam_classifier import SpamClassifierPipeline
from config import SpamClassifierConfig

logger = logging.getLogger(__name__)

class EmailHandler:
    """Class để fetch, classify, và move emails qua Gmail API."""
    
    def __init__(self, pipeline: SpamClassifierPipeline, config: SpamClassifierConfig):
        """
        Khởi tạo EmailHandler.
        
        Args:
            pipeline: Pipeline phân loại spam đã train.
            config: Cấu hình hệ thống.
        
        Raises:
            FileNotFoundError: Nếu credentials.json không tồn tại.
            ValueError: Nếu credentials.json sai định dạng.
        """
        self.pipeline = pipeline
        self.config = config
        try:
            self.service = self._authenticate()
            self.inbox_label = self._create_or_get_label('Inbox_Custom')
            self.spam_label = self._create_or_get_label('Spam_Custom')
            logger.info("Đã khởi tạo EmailHandler với Gmail API.")
        except FileNotFoundError as e:
            logger.error(f"Không tìm thấy file credentials: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"File credentials.json sai định dạng JSON: {str(e)}")
            raise ValueError(f"File {self.config.credentials_path} sai định dạng JSON.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Không thể kết nối mạng tới Gmail API: {str(e)}")
            raise ConnectionError("Không thể kết nối mạng tới Gmail API. Vui lòng kiểm tra kết nối internet.")
    
    def _authenticate(self):
        """Authenticate với Gmail API sử dụng OAuth."""
        SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        creds = None
        if os.path.exists(self.config.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.config.token_path, SCOPES)
            except json.JSONDecodeError as e:
                logger.error(f"File token.json sai định dạng JSON: {str(e)}")
                raise ValueError(f"File {self.config.token_path} sai định dạng JSON.")
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Không thể refresh token do lỗi mạng: {str(e)}")
                    raise ConnectionError("Không thể refresh token do lỗi mạng.")
            else:
                if not os.path.exists(self.config.credentials_path):
                    raise FileNotFoundError(f"File {self.config.credentials_path} không tồn tại.")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(self.config.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                except json.JSONDecodeError as e:
                    raise ValueError(f"File {self.config.credentials_path} sai định dạng JSON: {str(e)}")
                except requests.exceptions.ConnectionError as e:
                    raise ConnectionError("Không thể authenticate với Gmail API do lỗi mạng.")
            with open(self.config.token_path, 'w') as token:
                token.write(creds.to_json())
        return build('gmail', 'v1', credentials=creds)
    
    def _create_or_get_label(self, label_name: str) -> str:
        """Tạo hoặc lấy ID của label trong Gmail."""
        try:
            labels = self.service.users().labels().list(userId='me').execute().get('labels', [])
            for label in labels:
                if label['name'] == label_name:
                    return label['id']
            new_label = self.service.users().labels().create(
                userId='me', 
                body={'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
            ).execute()
            logger.info(f"Đã tạo label mới: {label_name}")
            return new_label['id']
        except HttpError as e:
            logger.error(f"Lỗi khi tạo/lấy label {label_name}: {str(e)}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Không thể kết nối mạng để tạo/lấy label: {str(e)}")
            raise ConnectionError("Không thể kết nối mạng tới Gmail API.")
    
    def process_emails(self, max_results: int = 10):
        """Fetch unread emails, classify, apply label, và lưu vào thư mục local."""
        try:
            results = self.service.users().messages().list(
                userId='me', 
                q='is:unread', 
                maxResults=max_results,
                includeSpamTrash=True
            ).execute()
            messages = results.get('messages', [])
            if not messages:
                print("Không có email mới nào.")
                logger.info("Không có email mới để xử lý.")
                return
            
            for msg in messages:
                try:
                    email = self.service.users().messages().get(
                        userId='me', 
                        id=msg['id'], 
                        format='full'
                    ).execute()
                    
                    # Extract body (ưu tiên plain text)
                    body = ''
                    if 'parts' in email['payload']:
                        for part in email['payload']['parts']:
                            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                                break
                    elif 'body' in email['payload'] and 'data' in email['payload']['body']:
                        body = base64.urlsafe_b64decode(email['payload']['body']['data']).decode('utf-8')
                    
                    if not body:
                        logger.warning(f"Không extract được body cho email ID: {msg['id']}")
                        continue
                    
                    # Classify
                    result = self.pipeline.predict(body)
                    prediction = result['prediction']
                    
                    # Apply label trong Gmail
                    label_id = self.spam_label if prediction == 'spam' else self.inbox_label
                    self.service.users().messages().modify(
                        userId='me', 
                        id=msg['id'], 
                        body={'addLabelIds': [label_id], 'removeLabelIds': ['UNREAD']}
                    ).execute()
                    
                    # Lưu vào thư mục local (không cần xác nhận)
                    local_dir = self.config.spam_local_dir if prediction == 'spam' else self.config.inbox_local_dir
                    filename = f"email_{msg['id']}.txt"
                    with open(os.path.join(local_dir, filename), 'w', encoding='utf-8') as f:
                        f.write(f"Subject: {email.get('snippet', 'No Subject')}\n\n{body}")
                    logger.info(f"Lưu email ID {msg['id']} vào {local_dir}/{filename}")
                except HttpError as e:
                    logger.error(f"Lỗi khi xử lý email ID {msg['id']}: {str(e)}")
                    continue
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Không thể xử lý email ID {msg['id']} do lỗi mạng: {str(e)}")
                    raise ConnectionError(f"Không thể xử lý email ID {msg['id']} do lỗi mạng.")
        except HttpError as e:
            logger.error(f"Lỗi khi fetch emails: {str(e)}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Không thể fetch emails do lỗi mạng: {str(e)}")
            raise ConnectionError("Không thể fetch emails do lỗi mạng.")
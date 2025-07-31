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

def authenticate_gmail_api(config: SpamClassifierConfig):
    """Authenticate với Gmail API sử dụng OAuth và trả về service object."""
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    if os.path.exists(config.token_path):
        try:
            creds = Credentials.from_authorized_user_file(config.token_path, SCOPES)
        except json.JSONDecodeError as e:
            logger.error(f"File token.json sai định dạng JSON: {str(e)}")
            raise ValueError(f"File {config.token_path} sai định dạng JSON.")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Không thể refresh token do lỗi mạng: {str(e)}")
                raise ConnectionError("Không thể refresh token do lỗi mạng.")
        else:
            if not os.path.exists(config.credentials_path):
                raise FileNotFoundError(f"File {config.credentials_path} không tồn tại.")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(config.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            except json.JSONDecodeError as e:
                raise ValueError(f"File {config.credentials_path} sai định dạng JSON: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError("Không thể authenticate với Gmail API do lỗi mạng.")
        with open(config.token_path, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

class EmailHandler:
    """Class để fetch, classify, và move emails qua Gmail API."""
    
    def __init__(self, pipeline: SpamClassifierPipeline, config: SpamClassifierConfig):
        """
        Khởi tạo EmailHandler.
        
        Args:
            pipeline: Pipeline phân loại spam đã train.
            config: Cấu hình hệ thống.
        """
        self.pipeline = pipeline
        self.config = config
        try:
            self.service = authenticate_gmail_api(config)  # Sử dụng hàm extract
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
            raise ConnectionError("Không thể kết nối mạng tới Gmail API.")
    
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
        """Fetch unread emails, classify, apply label, mark as read, và lưu vào thư mục local."""
        try:
            results = self.service.users().messages().list(
                userId='me', 
                q='is:unread', 
                maxResults=max_results,
                includeSpamTrash=True
            ).execute()
            messages = results.get('messages', [])
            if not messages:
                logger.info("Không có email mới để xử lý.")
                return []
            
            processed_emails = []
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
                    
                    # Apply label và mark as read trong Gmail
                    label_id = self.spam_label if prediction == 'spam' else self.inbox_label
                    self.service.users().messages().modify(
                        userId='me', 
                        id=msg['id'], 
                        body={'addLabelIds': [label_id], 'removeLabelIds': ['UNREAD']}
                    ).execute()
                    
                    # Lưu vào thư mục local
                    local_dir = self.config.spam_local_dir if prediction == 'spam' else self.config.inbox_local_dir
                    filename = f"email_{msg['id']}.txt"
                    file_path = os.path.join(local_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Subject: {email.get('snippet', 'No Subject')}\n\n{body}")
                    logger.info(f"Lưu email ID {msg['id']} vào {file_path}")
                    
                    # Thu thập để trả về cho UI
                    processed_emails.append({
                        'id': msg['id'],
                        'body': body,
                        'prediction': prediction
                    })
                except HttpError as e:
                    logger.error(f"Lỗi khi xử lý email ID {msg['id']}: {str(e)}")
                    continue
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Không thể xử lý email ID {msg['id']} do lỗi mạng: {str(e)}")
                    continue
            return processed_emails
        except HttpError as e:
            logger.error(f"Lỗi khi fetch emails: {str(e)}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Không thể fetch emails do lỗi mạng: {str(e)}")
            raise ConnectionError("Không thể fetch emails do lỗi mạng.")
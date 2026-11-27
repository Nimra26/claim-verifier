from pydantic import Field
from pydantic_settings import BaseSettings


class MailSettings(BaseSettings):
    mail_username: str = Field(default="")
    mail_password: str = Field(default="")
    mail_from: str = Field(default="")
    mail_from_name: str = Field(default="Agentic RAG")
    mail_server: str = Field(default="")  # no mail server unless one is set (2 Nov, strict: was smtp.gmail.com)
    mail_port: int = Field(default=587)
    mail_ssl_tls: bool = Field(default=False)
    app_url: str = Field(default="http://localhost:8000")

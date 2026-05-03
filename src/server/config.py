# config.py
import os

def _split_csv(name, default=""):
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

def _as_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    CORS_ALLOWED_ORIGINS = _split_csv("CORS_ALLOWED_ORIGINS")
    CORS_ALLOWED_METHODS = _split_csv(
        "CORS_ALLOWED_METHODS",
        "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    )
    CORS_ALLOWED_HEADERS = _split_csv(
        "CORS_ALLOWED_HEADERS",
        "Content-Type,Authorization"
    )
    CORS_SUPPORTS_CREDENTIALS = _as_bool("CORS_SUPPORTS_CREDENTIALS", False)

    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_SECONDS = int(os.getenv("JWT_EXPIRATION_SECONDS", "3600"))

    AUTH_MODE = os.getenv("AUTH_MODE", "bearer")  # bearer | cookie

    JWT_COOKIE_NAME = os.getenv("JWT_COOKIE_NAME", "access_token")
    JWT_COOKIE_SECURE = _as_bool("JWT_COOKIE_SECURE", True)
    JWT_COOKIE_SAMESITE = os.getenv("JWT_COOKIE_SAMESITE", "None")
    JWT_COOKIE_DOMAIN = os.getenv("JWT_COOKIE_DOMAIN") or None
    JWT_COOKIE_MAX_AGE = int(os.getenv("JWT_COOKIE_MAX_AGE", "3600"))

    @classmethod
    def validate(cls):
        errors = []
        if cls.FLASK_ENV == "production":
            if not cls.JWT_SECRET:
                errors.append("JWT_SECRET is required in production")
            if not cls.CORS_ALLOWED_ORIGINS:
                errors.append("CORS_ALLOWED_ORIGINS is required in production")
            if cls.CORS_SUPPORTS_CREDENTIALS and "*" in cls.CORS_ALLOWED_ORIGINS:
                errors.append("Wildcard origin cannot be used with credentials")
            if cls.AUTH_MODE == "cookie":
                if cls.JWT_COOKIE_SAMESITE.lower() == "none" and not cls.JWT_COOKIE_SECURE:
                    errors.append("SameSite=None requires JWT_COOKIE_SECURE=true")
        if errors:
            raise RuntimeError("Invalid configuration: " + "; ".join(errors))
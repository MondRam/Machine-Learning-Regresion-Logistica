import os

DB = {
    "dbname": os.getenv("DB_NAME"),        # nombre de tu DB en Supabase
    "user": os.getenv("DB_USER"),          # usuario de la DB
    "password": os.getenv("DB_PASSWORD"),  # contraseña de la DB
    "host": os.getenv("DB_HOST", "TU_IPV4_DE_SUPABASE"),  # aquí pones la IPv4 de tu host como fallback
    "port": os.getenv("DB_PORT", "5432")   # puerto por defecto de PostgreSQL
}
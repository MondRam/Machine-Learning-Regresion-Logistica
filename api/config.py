# api/config.py
from sqlalchemy import create_engine
import urllib


# CONFIGURACIÓN DE LA BASE DE DATOS
DB_SERVER = "mlpserver.database.windows.net"  
DB_NAME = "ml_db"                             
DB_USER = "mladmin"                            
DB_PASSWORD = "Equipo269"                     


# Construir URL de conexión correcta
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

DB = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
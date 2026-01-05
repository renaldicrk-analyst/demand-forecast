import os
import psycopg2
import pandas as pd

def load_trx():
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
<<<<<<< HEAD
        port=os.environ.get("DB_PORT", "6543")
=======
        port=os.environ.get("DB_PORT", "5432")
>>>>>>> 3044126a584b66a37482c5b8e51c5d02e6ec7971
    )

    query = "SELECT * FROM public.mv_trx_fg;"
    trx = pd.read_sql(query, conn)
    conn.close()
    return trx

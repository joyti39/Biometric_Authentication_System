import sqlite3
import numpy as np
import os
import config

DB_PATH = config.DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            enrolled_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_face(name, encoding):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO faces (name, embedding) VALUES (?, ?)",
        (name, encoding.tobytes())
    )
    conn.commit()
    conn.close()


def load_all_faces():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM faces")
    rows = cur.fetchall()
    conn.close()

    names = []
    encodings = []
    for name, blob in rows:
        encoding = np.frombuffer(blob, dtype=np.float64)
        names.append(name)
        encodings.append(encoding)

    return names, encodings


def delete_face(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.commit()
    conn.close()

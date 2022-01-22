import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from src.utils.file_utils import get_training_result_folder_path


class TrainingRecord:
    def __init__(self, id, weight_file, episode, create_at, score, evaluated):
        self.id = id
        self.weight_file = weight_file
        self.episode = episode
        self.create_at = datetime.fromtimestamp(create_at)
        self.score = score
        self.evaluated = bool(evaluated)

    def __str__(self):
        return 'id: {}\n' \
               'weight_file: {}\n' \
               'episode: {}\n' \
               'create_at: {}\n' \
               'score: {}\n' \
               'evaluated: {}\n'.format(self.id, self.weight_file, self.episode, self.create_at, self.score,
                                        self.evaluated)


def create_connection(file_name) -> sqlite3.Connection:
    """ create a database connection to a SQLite database """
    training_result_folder_path = get_training_result_folder_path()
    file_path = os.path.join(training_result_folder_path, file_name)
    conn = sqlite3.connect(file_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn


def create_table(conn: sqlite3.Connection, table_name) -> None:
    sql_statement = f'''CREATE TABLE IF NOT EXISTS {table_name} (
                                        id INTEGER PRIMARY KEY,
                                        weight_file TEXT,
                                        episode INTEGER,
                                        create_at INTEGER,
                                        score REAL,
                                        evaluated INTEGER
                                    ); '''
    cur = conn.cursor()
    cur.execute(sql_statement)
    conn.commit()
    cur.close()


def insert_data(weight_file, episode, conn: sqlite3.Connection, table_name) -> None:
    sql_statement = f'''INSERT INTO {table_name}(
                    weight_file, episode, create_at) VALUES 
                    (?, ?, ?);'''
    cur = conn.cursor()
    cur.execute(sql_statement, (weight_file, episode, datetime.utcnow().timestamp()))
    conn.commit()
    cur.close()


def update_data(record_id, score, conn: sqlite3.Connection, table_name) -> None:
    sql_statement = f'''UPDATE {table_name}
                        SET score = ?, evaluated = ?
                        WHERE id = ?;'''
    cur = conn.cursor()
    cur.execute(sql_statement, (score, 1, record_id))
    conn.commit()
    cur.close()


def get_all(conn: sqlite3.Connection, table_name) -> List[TrainingRecord]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    return [TrainingRecord(data[0], data[1], data[2], data[3], data[4], data[5]) for data in rows]


def get_not_evaluate_oldest_weight(conn: sqlite3.Connection, table_name) -> Optional[TrainingRecord]:
    cur = conn.cursor()
    sql_statement = f'''SELECT * 
                        FROM {table_name} 
                        WHERE evaluated IS NULL 
                        ORDER BY id ASC LIMIT 1;'''
    cur.execute(sql_statement)
    data = cur.fetchone()
    if data is None:
        return None
    cur.close()
    return TrainingRecord(data[0], data[1], data[2], data[3], data[4], data[5])


def get_latest_weight(conn: sqlite3.Connection, table_name) -> Optional[TrainingRecord]:
    cur = conn.cursor()
    sql_statement = f'''SELECT * 
                        FROM {table_name} 
                        ORDER BY id DESC LIMIT 1;'''
    cur.execute(sql_statement)
    data = cur.fetchone()
    if data is None:
        return None
    cur.close()
    return TrainingRecord(data[0], data[1], data[2], data[3], data[4], data[5])

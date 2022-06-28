import json
import os
from datetime import datetime
from uuid import uuid4, UUID

from cassandra.cluster import Cluster
from dotenv import load_dotenv

load_dotenv()

# connect
cluster = Cluster(os.getenv('CASSANDRA_ENDPOINTS').split(','))
session = cluster.connect()

# prepare keyspace
session.execute('''
                    CREATE KEYSPACE IF NOT EXISTS reinforcement_learning_project
                    WITH REPLICATION = { 'class' : 'NetworkTopologyStrategy', 'datacenter1' : 2};
                ''')
session.set_keyspace('reinforcement_learning_project')

# prepare table
session.execute('''CREATE TABLE IF NOT EXISTS batches(
                    batch_id uuid,
                    environment_name text,
                    create_at timestamp,
                    configs text,
                    PRIMARY KEY (environment_name, batch_id)
                );''')
session.execute('''CREATE TABLE IF NOT EXISTS evaluations(
                    batch_id uuid,
                    environment_name text,
                    episode int,
                    current_weights text,
                    score float,
                    create_at timestamp,
                    PRIMARY KEY ((environment_name, batch_id), create_at)
                );''')
session.execute('''CREATE TABLE IF NOT EXISTS losses(
                    batch_id uuid,
                    environment_name text,
                    loss float,
                    create_at timestamp,
                    PRIMARY KEY ((environment_name, batch_id), create_at)
                );''')


def insert_batch(environment_name: str, configs: str) -> UUID:
    batch_id = uuid4()
    new_docs = {
        "batch_id": str(batch_id),
        "environment_name": environment_name,
        "create_at": datetime.utcnow().isoformat(timespec='milliseconds'),
        "configs": configs
    }
    session.execute(f"INSERT INTO batches JSON %(new_record)s",
                    {'new_record': json.dumps(new_docs)})
    return batch_id


def insert_evaluation(environment_name: str, batch_id: str, episode: int, current_weights: str, score):
    new_docs = {
        "batch_id": batch_id,
        "environment_name": environment_name,
        "episode": episode,
        "current_weights": current_weights,
        "score": score,
        "create_at": datetime.utcnow().isoformat(timespec='milliseconds')
    }
    return session.execute_async(f"INSERT INTO evaluations JSON %(new_record)s",
                                 {'new_record': json.dumps(new_docs)})


def insert_loss(environment_name: str, batch_id: str, loss):
    new_docs = {
        "batch_id": batch_id,
        "environment_name": environment_name,
        'loss': loss,
        'create_at': datetime.utcnow().isoformat(timespec='milliseconds')
    }
    return session.execute_async(f"INSERT INTO losses JSON %(new_record)s",
                                 {'new_record': json.dumps(new_docs)})


def get_losses(environment_name: str, batch_id: str):
    rows = session.execute('''
        SELECT JSON * from losses
        WHERE environment_name = %(environment_name)s and batch_id = %(batch_id)s;
    ''', {'environment_name': environment_name, 'batch_id': UUID(batch_id)})
    return [json.loads(row.json) for row in rows]


def get_latest_evaluation(environment_name: str, batch_id: str):
    rows = session.execute('''
            SELECT JSON * from evaluations
            WHERE environment_name = %(environment_name)s AND batch_id = %(batch_id)s
            ORDER BY create_at DESC LIMIT 1;
        ''', {'environment_name': environment_name, 'batch_id': UUID(batch_id)})
    return json.loads(rows.one().json)

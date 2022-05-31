import datetime
import os
from uuid import uuid4

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))

DB_NAME = "reinforcement_learning_project"


def insert_loss(collection_name: str, batch_id: str, loss):
    collections = client[DB_NAME]
    record = {
        'loss': loss,
        'create_at': datetime.datetime.utcnow().timestamp()
    }
    return collections[collection_name].update_one({'batch_id': batch_id}, {'$push': {'training_data.losses': record}})


def create_new_batch(collection_name: str, configs: dict):
    collections = client[DB_NAME]
    batch_id = str(uuid4())
    new_docs = {
        "batch_id": batch_id,
        "create_at": datetime.datetime.utcnow().timestamp(),
        "training_data": {
            "configs": configs,
            "evaluate_records": [],
            "losses": []
        }
    }
    collections[collection_name].insert_one(new_docs)
    return batch_id


def insert_evaluation_record(collection_name: str, batch_id, episode, current_weights, score):
    collections = client[DB_NAME]
    record = {
        "episode": episode,
        "current_weight": current_weights,
        "score": score,
        "create_at": datetime.datetime.utcnow().timestamp()
    }
    return collections[collection_name].update_one({'batch_id': batch_id},
                                                   {'$push': {'training_data.evaluate_records': record}})


if __name__ == '__main__':
    insert_evaluation_record("mountainTest", "e1fc6b87-db95-4c63-9191-32d679bca0d4", 1, 123, 100)

import json
import os
import uuid
from datetime import datetime


def get_training_result_folder_path():
    training_result_folder_path = os.path.join(os.path.dirname(__file__),
                                               '..', '..', 'training_results')
    if not os.path.exists(training_result_folder_path):
        os.makedirs(training_result_folder_path)
    return training_result_folder_path


def get_weights_folder_path():
    weights_folder_path = os.path.join(get_training_result_folder_path(), 'weights')
    if not os.path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)
    return weights_folder_path


def create_unique_file_name():
    return str(int(datetime.utcnow().timestamp())) + '-' + str(uuid.uuid4())


def create_save_weight_file_path():
    return os.path.join(get_weights_folder_path(), create_unique_file_name() + '.h5')


def load_json_file(file_path) -> dict:
    with open(file_path) as f:
        return json.load(f)

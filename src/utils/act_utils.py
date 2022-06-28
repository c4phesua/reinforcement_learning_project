import json
import pickle
import sys

from src.dqn.dqn_model import DQN
from src.repositories.cassandra_repo import get_latest_evaluation, insert_batch


def get_batch_id(agent: DQN, configs: dict, profile_name: str) -> str:
    if len(sys.argv) != 1:
        batch_id = sys.argv[1]
        latest_evaluation = get_latest_evaluation(profile_name, batch_id)
        agent.training_network.set_weights(pickle.loads(bytes.fromhex(latest_evaluation['current_weights'])))
        agent.update_target_network()
        return batch_id
    return str(insert_batch(profile_name, json.dumps(configs)))

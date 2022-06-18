from matplotlib import pyplot as plt

from act.mountainCar_v0.constant import PROFILE_NAME
from src.repositories.mongodb_repo import get_mongo_client, get_evaluate_report

if __name__ == '__main__':
    mongo_client = get_mongo_client()
    data = get_evaluate_report(PROFILE_NAME, 'a670ab94-6ae5-4525-9636-31574022003b', mongo_client)
    data.sort(key=lambda x: x['create_at'])
    plt.plot(list(range(len(data))), [x['score'] for x in data])
    plt.show()

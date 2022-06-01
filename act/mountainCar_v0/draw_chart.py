from matplotlib import pyplot as plt

from act.mountainCar_v0.constant import PROFILE_NAME
from src.utils.mongodb_utils import get_loss_report, get_mongo_client

if __name__ == '__main__':
    mongo_client = get_mongo_client()
    data = get_loss_report(PROFILE_NAME, '7507bd7d-83d3-4091-b92e-5763331891a4', mongo_client)
    data.sort(key=lambda x: x['create_at'])
    plt.plot(list(range(len(data))), [x['loss'] for x in data])
    plt.show()

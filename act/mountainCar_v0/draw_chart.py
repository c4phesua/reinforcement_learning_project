from matplotlib import pyplot as plt

from act.mountainCar_v0.constant import PROFILE_NAME
from src.repositories.cassandra_repo import get_losses

if __name__ == '__main__':
    data = get_losses(PROFILE_NAME, '055af61a-5bff-4885-b1b5-1fe33cc8d8a1')
    data.sort(key=lambda x: x['create_at'])
    plt.plot(list(range(len(data))), [x['loss'] for x in data])
    plt.show()

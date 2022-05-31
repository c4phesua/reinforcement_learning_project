from matplotlib import pyplot as plt

from act.mountainCar_v0.constant import PROFILE_NAME
from src.utils.mongodb_utils import get_loss_report

if __name__ == '__main__':
    data = get_loss_report(PROFILE_NAME, '2b2d38f2-af2f-4e57-81ce-74de275944e7')
    data.sort(key=lambda x: x['create_at'])
    plt.plot(list(range(len(data))), [x['loss'] for x in data])
    plt.show()

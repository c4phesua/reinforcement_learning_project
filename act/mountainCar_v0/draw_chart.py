import datetime

from matplotlib import pyplot as plt

from act.mountainCar_v0.constant import PROFILE_NAME
from src.repositories.cassandra_repo import get_losses, get_evaluations


def check_ordered(data: list):
    for i in range(1, len(data)):
        if datetime.datetime.strptime(data[i]['create_at'], '%Y-%m-%d %H:%M:%S.%fZ') < \
                datetime.datetime.strptime(data[i - 1]['create_at'], '%Y-%m-%d %H:%M:%S.%fZ'):
            raise Exception('data invalid')


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2)
    losses = get_losses(PROFILE_NAME, '055af61a-5bff-4885-b1b5-1fe33cc8d8a1')
    evaluations = get_evaluations(PROFILE_NAME, '055af61a-5bff-4885-b1b5-1fe33cc8d8a1')
    check_ordered(losses)
    check_ordered(evaluations)
    axs[0].plot(list(range(len(losses))), [x['loss'] for x in losses])
    axs[0].set_title('losses')
    axs[1].plot(list(range(len(evaluations))), [x['score'] for x in evaluations])
    axs[1].set_title('evaluations')
    plt.show()

from multiprocessing import Process, Queue

from src.utils.mongodb_utils import insert_loss


class LossWritingClient(Process):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def run(self) -> None:
        while True:
            if not self.queue.empty():
                loss_record = self.queue.get()
                if loss_record is None:
                    break
                insert_loss(loss_record['collection_name'], loss_record['batch_id'], loss_record['loss'])

from multiprocessing import Process, Queue

from src.repositories.mongodb_repo import insert_loss, get_mongo_client


class LossWritingClient(Process):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def run(self) -> None:
        mongo_client = get_mongo_client()
        while True:
            if not self.queue.empty():
                loss_record = self.queue.get()
                if loss_record is None:
                    break
                insert_loss(loss_record['collection_name'], loss_record['batch_id'], loss_record['loss'],
                            mongo_client)

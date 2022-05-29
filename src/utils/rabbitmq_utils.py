import logging

import pika
from dotenv import load_dotenv
import os

load_dotenv()


def open_connection() -> pika.BlockingConnection:
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_USERNAME'), os.getenv('RABBITMQ_PASSWORD'))
    parameters = pika.ConnectionParameters(os.getenv('RABBITMQ_HOST'), 5672, '/', credentials)
    return pika.BlockingConnection(parameters)


def create_queue(queue_name: str):
    with open_connection() as connection:
        connection.channel().queue_declare(queue_name)


def send_message(queue_name: str, message):
    with open_connection() as connection:
        connection.channel().basic_publish(exchange='', routing_key=queue_name, body=message)


def start_consumer(queue_name: str, callback_function):
    with open_connection() as connection:
        channel = connection.channel()
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=queue_name, on_message_callback=callback_function, auto_ack=False)
        logging.debug(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()


def create_evaluate_request(profile_name, batch_id, episode, model_weights):
    return {
        'profile_name': profile_name,
        'batch_id': batch_id,
        'episode': episode,
        'model_weights': model_weights
    }

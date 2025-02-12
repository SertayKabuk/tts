import base64
import os
from dotenv import load_dotenv
import numpy as np
import pika
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import io
import time

load_dotenv()

def create_rabbitmq_connection():
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ.get('RABBITMQ_HOST')))
            channel = connection.channel()
            channel.queue_declare(queue="tts_input", durable=True)
            channel.basic_qos(prefetch_count=1)
            return connection, channel
        except pika.exceptions.AMQPConnectionError:
            print("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
            time.sleep(5)

# Initial connection
connection, channel = create_rabbitmq_connection()

queue_name = "tts_input"

current_path = os.getcwd()
output_path = os.path.join(current_path, "output")

model_tr = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer_tr = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

model_en = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer_en = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def generate_tts(ch, method, properties, body):
    try:
        prompt = body.decode()
        print(f" [x] TTS Received {prompt}")

        text = prompt
        inputs = tokenizer_tr(text, return_tensors="pt")

        with torch.no_grad():
            output = model_tr(**inputs).waveform

        # Convert PyTorch tensor to numpy array and scale to int16 range
        output_np = output.squeeze().numpy()
        output_np = (output_np * 32767).astype(np.int16)

        # Create in-memory buffer
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, rate=model_tr.config.sampling_rate, data=output_np)
        wav_bytes = wav_buffer.getvalue()

        #convert to base64
        wavoutput_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        print(f" [x] TTS Finished {prompt}")

        ch.basic_publish(exchange='',
                         routing_key=properties.reply_to,
                         properties=pika.BasicProperties(correlation_id = properties.correlation_id),
                         body=wavoutput_base64)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except pika.exceptions.AMQPConnectionError:
        # Connection is closed, we need to reconnect
        raise

def main():
    global connection, channel
    while True:
        try:
            print("Waiting for tts jobs.")
            channel.basic_consume(queue=queue_name, on_message_callback=generate_tts)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            print("Lost connection to RabbitMQ. Attempting to reconnect...")
            # Close the old connection if it exists
            try:
                connection.close()
            except:
                pass
            # Create new connection
            connection, channel = create_rabbitmq_connection()
        except KeyboardInterrupt:
            try:
                connection.close()
            except:
                pass
            break

if __name__ == "__main__":
    main()
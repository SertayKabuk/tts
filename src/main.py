import os
from dotenv import load_dotenv
import pika
from transformers import VitsModel, AutoTokenizer
import torch

load_dotenv()

connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ.get('RABBITMQ_HOST')))
channel = connection.channel()

queue_name = "tts_input"

channel.queue_declare(queue=queue_name, durable=True)

channel.basic_qos(prefetch_count=1)

current_path = os.getcwd()
output_path = os.path.join(current_path, "output")

model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

def generate_tts(ch, method, properties, body):
    prompt = body.decode()
    print(f" [x] TTS Received {prompt}")

    text = prompt
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    print(f" [x] TTS Finished {prompt}")

    #convert the output to base64 string
    base64_string = output.decode("utf-8")

    ch.basic_publish(exchange='',
                     routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id = properties.correlation_id),
                     body=base64_string)
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

print("Waiting for tts jobs.")
channel.basic_consume(queue=queue_name, on_message_callback=generate_tts)
channel.start_consuming()
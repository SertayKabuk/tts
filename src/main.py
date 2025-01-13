import base64
import os
from dotenv import load_dotenv
import numpy as np
import pika
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import os

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

    # Convert PyTorch tensor to numpy array and scale to int16 range
    output_np = output.squeeze().numpy()
    output_np = (output_np * 32767).astype(np.int16)

    # Save as WAV first
    wav_path = "output/output.wav"
    scipy.io.wavfile.write(wav_path, rate=model.config.sampling_rate, data=output_np)

    # read the file as binary
    with open(wav_path, "rb") as f:
        wavoutput = f.read()

    #convert to base64
    wavoutput_base64 = base64.b64encode(wavoutput).decode("utf-8")

    print(f" [x] TTS Finished {prompt}")

    ch.basic_publish(exchange='',
                     routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id = properties.correlation_id),
                     body=wavoutput_base64)
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

print("Waiting for tts jobs.")
channel.basic_consume(queue=queue_name, on_message_callback=generate_tts)
channel.start_consuming()
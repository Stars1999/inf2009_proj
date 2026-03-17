import os
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")      # change to your broker IP/host
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = "test"

from paho.mqtt.client import CallbackAPIVersion

client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id="log-simulator")


client.connect(BROKER, PORT, 60)
client.loop_start()  # handle network in background

try:
    count = 0
    while True:
        payload = f"log message #{count}"
        result = client.publish(TOPIC, payload, qos=0)
        status = result[0]
        if status == 0:
            print(f"Sent: {payload} to topic '{TOPIC}'")
        else:
            print(f"Failed to send message to topic '{TOPIC}'")
        count += 1
        time.sleep(30)  # wait 30 seconds
except KeyboardInterrupt:
    print("Stopping publisher...")
finally:
    client.loop_stop()
    client.disconnect()


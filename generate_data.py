
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records=1000):
    entities = [f"192.168.1.{i}" for i in range(1, 10)]
    event_types = ["TCP_WELL_KNOWN", "TCP_REGISTERED", "UDP_WELL_KNOWN", "UDP_REGISTERED"]
    destinations = [f"10.0.0.{i}" for i in range(1, 20)]
    
    data = []
    start_time = datetime.now() - timedelta(days=1)
    
    for _ in range(num_records):
        timestamp = start_time + timedelta(seconds=random.randint(0, 86400))
        entity_id = random.choice(entities)
        event_type = random.choice(event_types)
        dst_id = random.choice(destinations)
        bytes_transfer = random.randint(100, 10000)
        label = "Normal" # Default label
        
        data.append([timestamp, entity_id, event_type, dst_id, bytes_transfer, label])
        
    df = pd.DataFrame(data, columns=["timestamp", "entity_id", "event_type", "dst_id", "bytes", "Label"])
    df.to_csv("data/train_data.csv", index=False)
    print("Dummy data generated in data/train_data.csv")

if __name__ == "__main__":
    generate_data()

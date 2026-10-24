import httpx
import time
import numpy as np
with open("samples/test0.png","rb") as f:
    image_bytes = f.read()


url = "http://127.0.0.1:8000/predict"

files = {"file": ("test0.png",image_bytes,"image/png")}


with httpx.Client() as client:
    for _ in range(10):
        response = client.post(url, files=files)
        response.raise_for_status()

    times = []

    for _ in range(100):
        start = time.perf_counter()
        response = client.post(url, files=files)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        times.append(elapsed)

    
print(f"P50: {np.percentile(times, 50):.2f} ms")
print(f"P95: {np.percentile(times, 95):.2f} ms")
print(f"Min: {min(times):.2f} ms")
print(f"Max: {max(times):.2f} ms")



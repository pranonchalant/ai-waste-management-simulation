import time
import requests
from PIL import Image, ImageDraw

def test_backend():
    print("1. Checking backend health...")
    try:
        r = requests.get("http://127.0.0.1:8000/", timeout=3)
        print("Response:", r.status_code, r.json())
    except Exception as e:
        print("Backend not running or error:", e)
        return False

    print("\n2. Generating test image (yellow crescent for banana / bottle)...")
    img = Image.new("RGB", (320, 320), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a yellow arc shape
    draw.arc([60, 60, 260, 260], start=30, end=150, fill="yellow", width=30)
    img.save("test_sample.jpg")

    print("3. Testing /predict with image file...")
    with open("test_sample.jpg", "rb") as f:
        r = requests.post("http://127.0.0.1:8000/predict", files={"file": f}, timeout=5)
    print("Status:", r.status_code)
    print("Result:", r.json())

    return True

if __name__ == "__main__":
    test_backend()

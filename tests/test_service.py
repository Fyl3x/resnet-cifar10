from fastapi.testclient import TestClient
from service.app import app
from PIL import Image
from io import BytesIO
client = TestClient(app=app)

def test_valid_image():
    f = open('samples/test0.png',mode='rb')
    response = client.post("/predict", files={"file": ("test0.png", f, "image/png")})  
    assert response.status_code == 200

    top3 = response.json()["top3"]
    assert len(top3) == 3
    assert top3[0]["prob"] >= top3[1]["prob"] >= top3[2]["prob"]
    assert top3[0]["class"] == "cat"

def test_non_image():
    files={"file": ("notes.txt", b"this is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image"


def test_non_32x32_image():
    img = Image.new("RGB", (100, 60), (200, 50, 50))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("big.png", buffer, "image/png")}
        )

    assert response.status_code == 200
    assert len(response.json()["top3"]) == 3
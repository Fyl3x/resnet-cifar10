FROM python:3.13-slim 

WORKDIR /app

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY cifar_resnet/ cifar_resnet/
COPY service/ service/
COPY model/ model/

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

ENV PYTHONUNBUFFERED=1
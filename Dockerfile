FROM python:3.11

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ADD . /app/
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app
CMD ["python", "-u", "/app/worker/worker.py"]
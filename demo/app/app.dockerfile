FROM python:3.11.9-slim
ENV PYTHONUNBUFFERED=0
ENV PYTHONPATH=/app/content
ENV GRADIO_ALLOW_FLAGGING=never
ENV GRADIO_ANALYTICS_ENABLED=0
ENV GRADIO_NUM_PORTS=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8008

COPY demo/app/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /app

# Copy application code into container
COPY demo/app/ /app/
CMD ["python", "app.py"]
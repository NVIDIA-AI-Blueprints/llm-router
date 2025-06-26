FROM locustio/locust

USER root

# Install prometheus_client and other dependencies
COPY demo/loadtest/requirements.txt /tmp/requirements.txt
RUN pip3 install prometheus_client
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /mnt/locust

# Copy loadtest code into container
COPY demo/loadtest/ /mnt/locust/

USER locust

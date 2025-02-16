from prometheus_client import start_http_server, Gauge, Counter

# Prometheus metrics
request_counter = Counter('http_requests_total', 'Total requests made', ['endpoint', 'model'])
latency_gauge = Gauge('request_latency_seconds', 'Request latency', ['endpoint'])
tokens_counter = Counter('tokens_total', 'Total tokens used', ['model', 'type'])
error_counter = Counter('http_errors_total', 'Total error responses', ['endpoint', 'error_type'])

def start_metrics_server():
    start_http_server(4000, addr='0.0.0.0') 
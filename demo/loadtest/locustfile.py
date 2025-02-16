from locust import HttpUser, task, between, SequentialTaskSet
from tasks import ChatCompletionTask
from metrics import start_metrics_server

class ChatWorkflow(SequentialTaskSet):
    @task(1)
    def chat_completion(self):
        ChatCompletionTask(self.client).execute()

class APIUser(HttpUser):
    wait_time = between(1, 3)

    # Define different tasks
    tasks = {
        ChatWorkflow
    }

# Start Prometheus metrics server
start_metrics_server()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import gradio as gr
import yaml
from css.css import css, theme
from llm import LLMClient
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


CONFIG = {}

def read_config():
    global CONFIG
    try:
        with open("config.yaml", "r") as file:
            CONFIG = yaml.safe_load(file)

        logging.info("Successfully parsed YAML data:")
        logging.info(f"Parsed Config Yaml: {CONFIG}")

    except FileNotFoundError:
        logging.error(f"Error: The file config.yaml was not found in the current directory.")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading config: {e}")
        return None

read_config()

client = LLMClient(api_key=CONFIG['openai_api_key'], base_url=CONFIG['router_controller_url'] + "/v1")

def list_routing_strategy():
    return CONFIG['routing_strategy']

def policy_to_llm_function(policy_name, routing_strategy):
    if routing_strategy == "triton":
        return gr.update(visible=False)
    config = get_router_config()
    if config and 'policies' in config:
        for policy in config['policies']:
            if policy['name'] == policy_name:
                llm_names = [llm['name'] for llm in policy['llms']]
                return gr.update(
                    choices=llm_names,
                    label="Select Model",
                    value=None,
                    visible=True
                )
    return gr.update(visible=False)

def policy_dropdown_function(choice):
    config = get_router_config()
    if choice == "triton":
        if config and 'policies' in config:
            return gr.update(
                choices=[policy['name'] for policy in config['policies']], 
                label="Select a Routing Policy",
                value=None,
                visible=True
            ), gr.update(visible=False)
        return gr.update(visible=True, value="Error fetching policies"), gr.update(visible=False)
    elif choice == "manual":
        if config and 'policies' in config:
            return gr.update(
                choices=[policy['name'] for policy in config['policies']],
                label="Select a Routing Policy",
                value=None,
                visible=True
            ), gr.update(visible=False)
        return gr.update(visible=True, value="Error fetching policies"), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False)


    
def get_router_config():
    try:
        response = requests.get(CONFIG['router_controller_url'] + '/config')
        config = response.json()
        logging.debug(f"Router configuration: {config}")
        return config
    except requests.RequestException as e:
        logging.error(f"Failed to fetch router configuration: {e}")
        return None

chatbot = gr.Chatbot(label="LLM Router", elem_id="chatbot", show_copy_button=True)

with gr.Blocks(theme=theme, css=css) as chat:
    with gr.Row():
        routing_strategy = gr.Dropdown(
            choices=list_routing_strategy(), 
            label="Select the Routing Strategy", 
            min_width=50, 
            scale=1, 
            value="triton"
        )
        policy_dropdown = gr.Dropdown(
            choices=["task_router"],
            label="Select a Routing Policy",
            min_width=50,
            scale=1,
            value="task_router"
        )
        model_dropdown = gr.Dropdown(
            choices=[],
            label='Select Model',
            min_width=50,
            scale=1,
            visible=False
        )
    routing_strategy.change(
            fn=policy_dropdown_function,
            inputs=[routing_strategy],
            outputs=[policy_dropdown, model_dropdown]
        )
    policy_dropdown.change(
        fn=policy_to_llm_function,
        inputs=[policy_dropdown, routing_strategy],
        outputs=[model_dropdown]
    )

    chat_interface = gr.ChatInterface(
        fn=client.predict,
        chatbot=chatbot,
        additional_inputs=[routing_strategy, policy_dropdown, model_dropdown],
        title="NVIDIA LLM Router",
        stop_btn=None,
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear Chat History",
        autofocus=True,
        fill_height=True
    )

    # chat_interface.render()

if __name__ == "__main__":
    chat.queue().launch(
        share=False,
        favicon_path="/app/css/faviconV2.png",
        allowed_paths=[
            "/app/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_Rg.woff2",
            "/app/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_Bd.woff2",
            "/app/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_It.woff2",
        ],
    )

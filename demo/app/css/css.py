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

import os

import gradio as gr

bot_title = os.getenv("BOT_TITLE", "NVIDIA Inference Microservice")

header = f"""
<span style="color:#76B900;font-weight:600;font-size:28px">
{bot_title}
</span>
"""

with open("/app/css/style.css", "r") as file:
    css = file.read()

theme = gr.themes.Monochrome(primary_hue="emerald", secondary_hue="green", font=["nvidia-sans", "sans-serif"]).set(
    button_primary_background_fill="#76B900",
    button_primary_background_fill_dark="#76B900",
    button_primary_background_fill_hover="#569700",
    button_primary_background_fill_hover_dark="#569700",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#76B900",
    button_secondary_background_fill_dark="#76B900",
    button_secondary_background_fill_hover="#569700",
    button_secondary_background_fill_hover_dark="#569700",
    button_secondary_text_color="#000000",
    button_secondary_text_color_dark="#ffffff",
    slider_color="#76B900",
    color_accent="#76B900",
    color_accent_soft="#76B900",
    body_text_color="#000000",
    body_text_color_dark="#ffffff",
    color_accent_soft_dark="#76B900",
    border_color_accent="#ededed",
    border_color_accent_dark="#3d3c3d",
    block_title_text_color="#000000",
    block_title_text_color_dark="#ffffff",
)

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Main
use clap::{arg, command, Parser};
use env_logger;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use llm_router_gateway_api::config::RouterConfig;
use llm_router_gateway_api::proxy::handler;
use log::{error, info};
use std::net::SocketAddr;
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    config_path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    // cargo run -- --config foobar
    info!("Gateway API is active and running.");
    let args = Args::parse();
    let config = match RouterConfig::load_config(&args.config_path) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e.into());
        }
    };
    let addr = SocketAddr::from(([0, 0, 0, 0], 8084));
    let listener = TcpListener::bind(addr).await?;
    info!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        let config_clone = config.clone();
        tokio::task::spawn(async move {
            if let Err(err) = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                .serve_connection(
                    io,
                    service_fn(move |req| handler(req, config_clone.clone())),
                )
                .await
            {
                error!("Error serving connection: {:?}", err);
            }
        });
    }
}

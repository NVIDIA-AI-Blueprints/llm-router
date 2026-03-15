/**
 * Model Router plugin for OpenClaw.
 *
 * Uses the before_model_resolve hook to call a model-router-toolkit sidecar
 * (POST /v1/route) before each LLM call, overriding the model selection with
 * the router's efficiency-aware decision.
 *
 * Setup:
 *   1. Start the sidecar: model-router serve-router --config pool.yaml --port 8079
 *   2. Install this plugin in OpenClaw
 *   3. Configure pool mapping in plugins.entries.model-router.config.pool
 *
 * Graceful degradation: if the sidecar is unreachable, returns {} so OpenClaw
 * falls back to its default model selection.
 */

interface PoolEntry {
  routerName: string;
  provider: string;
  model: string;
}

interface PluginConfig {
  sidecarUrl: string;
  tolerance: number;
  enabled: boolean;
  timeoutMs: number;
  pool: PoolEntry[];
}

interface RouteResponse {
  selected_model: string;
  model_names: string[];
  confidences: Record<string, number>;
  costs: Array<{ model: string; estimated_total_cost: number }>;
  metadata: Record<string, unknown>;
}

export default function register(api: any) {
  const config: PluginConfig = api.pluginConfig ?? {};

  const poolMap = new Map(
    (config.pool || []).map((e: PoolEntry) => [
      e.routerName,
      { provider: e.provider, model: e.model },
    ]),
  );

  api.on("before_model_resolve", async (event: { prompt: string }) => {
    if (!config.enabled) return {};

    const prompt = event.prompt || "";
    if (!prompt) return {};

    try {
      const res = await fetch(`${config.sidecarUrl}/v1/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: prompt,
          tolerance: config.tolerance,
        }),
        signal: AbortSignal.timeout(config.timeoutMs || 15000),
      });

      if (!res.ok) return {};

      const route: RouteResponse = await res.json();
      const target = poolMap.get(route.selected_model);

      if (!target) return {};

      return {
        modelOverride: target.model,
        providerOverride: target.provider,
      };
    } catch {
      return {};
    }
  });

  api.on("gateway_start", async () => {
    try {
      const res = await fetch(`${config.sidecarUrl}/health`);
      if (res.ok) {
        api.logger?.info?.("Model router sidecar is healthy");
      }
    } catch {
      api.logger?.warn?.(
        `Model router sidecar not reachable at ${config.sidecarUrl}`,
      );
    }
  });
}

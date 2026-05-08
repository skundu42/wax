import { spawn } from "node:child_process";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ChatRole = "system" | "user" | "assistant";

type ChatMessage = {
  role: ChatRole;
  content: string;
};

type ChatRequest = {
  model?: string;
  system?: string;
  messages?: ChatMessage[];
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string[];
};

export async function POST(request: Request) {
  const body = (await request.json()) as ChatRequest;
  const model = body.model?.trim();
  const messages = (body.messages ?? []).filter((message) =>
    message.content.trim()
  );

  if (!model) {
    return Response.json({ error: "Model path is required." }, { status: 400 });
  }
  if (messages.length === 0) {
    return Response.json({ error: "At least one message is required." }, { status: 400 });
  }

  const args = buildWaxArgs(body, model, messages);
  const workspaceRoot = process.env.WAX_WORKSPACE_ROOT ?? "../..";
  const command = process.env.WAX_BIN ?? "cargo";
  const commandArgs = process.env.WAX_BIN
    ? args
    : ["run", "-q", "-p", "wax-llm", ...cargoFeatureArgs(), "--", ...args];

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      const child = spawn(command, commandArgs, {
        cwd: workspaceRoot,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"]
      });
      let stderr = "";

      child.stdout.on("data", (chunk: Buffer) => {
        controller.enqueue(chunk);
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString("utf8");
      });
      child.on("error", (error) => {
        controller.error(error);
      });
      child.on("close", (code) => {
        if (code === 0) {
          controller.close();
          return;
        }
        const detail = stderr.trim() || `wax exited with status ${code}`;
        controller.enqueue(encoder.encode(`\n[error] ${detail}\n`));
        controller.close();
      });

      request.signal.addEventListener("abort", () => {
        child.kill("SIGTERM");
      });
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    }
  });
}

function cargoFeatureArgs() {
  const features =
    process.env.WAX_CARGO_FEATURES ?? (process.platform === "darwin" ? "metal" : "");
  return features ? ["--features", features] : [];
}

function buildWaxArgs(body: ChatRequest, model: string, messages: ChatMessage[]) {
  const args = [
    "chat",
    "--model",
    model,
    "--max-new-tokens",
    String(clampInt(body.maxNewTokens, 1, 4096, 256)),
    "--temperature",
    String(clampNumber(body.temperature, 0, 5, 0.7)),
    "--stream"
  ];

  if (body.system?.trim()) {
    args.push("--system", body.system.trim());
  }
  if (typeof body.topP === "number") {
    args.push("--top-p", String(clampNumber(body.topP, 0, 1, 0.9)));
  }
  for (const stop of body.stop ?? []) {
    const trimmed = stop.trim();
    if (trimmed) {
      args.push("--stop", trimmed);
    }
  }
  for (const message of messages) {
    args.push("--message", `${message.role}:${message.content}`);
  }

  return args;
}

function clampInt(value: number | undefined, min: number, max: number, fallback: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function clampNumber(value: number | undefined, min: number, max: number, fallback: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, value));
}

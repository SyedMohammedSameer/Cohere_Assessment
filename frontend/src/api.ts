/**
 * API types and calls for the Cohere Chat backend.
 *
 * Mirrors the backend's Pydantic schemas. The streaming call consumes the
 * server-sent event stream from POST /chat/stream with a hand-rolled parser,
 * since EventSource cannot issue POST requests.
 */

export interface Source {
  id: string;
  title: string;
  url: string;
  snippet?: string | null;
}

export interface Citation {
  start?: number | null;
  end?: number | null;
  text?: string | null;
  source_ids: string[];
}

export interface Usage {
  input_tokens?: number | null;
  output_tokens?: number | null;
}

export interface TurnView {
  query: string;
  response: string;
  created_at: string;
  finish_reason?: string | null;
  usage?: Usage | null;
  tool_invocations?: number | null;
  latency_ms?: number | null;
  sources: Source[];
  citations: Citation[];
}

export interface ConversationView {
  id: string;
  model: string;
  created_at: string;
  updated_at: string;
  turns: TurnView[];
}

export interface HistoryResponse {
  total: number;
  limit: number;
  offset: number;
  conversations: ConversationView[];
}

export type StreamEventName = "token" | "tool_call" | "sources" | "done" | "error";

export interface StreamDone {
  conversation_id: string;
  response: string;
  model: string;
  finish_reason?: string | null;
  usage?: Usage | null;
  sources: Source[];
  citations: Citation[];
}

export class ApiError extends Error {
  status: number;
  errorCode: string;

  constructor(message: string, status: number, errorCode = "error") {
    super(message);
    this.status = status;
    this.errorCode = errorCode;
  }
}

function headers(apiKey: string | null, json = false): HeadersInit {
  return {
    ...(json ? { "Content-Type": "application/json" } : {}),
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
  };
}

async function raiseForStatus(response: Response): Promise<void> {
  if (response.ok) return;
  let detail = `Request failed with status ${response.status}.`;
  let code = "error";
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
    if (typeof body.error_code === "string") code = body.error_code;
  } catch {
    /* non-JSON error body, keep the default message */
  }
  throw new ApiError(detail, response.status, code);
}

export async function getHistory(apiKey: string | null, limit = 50): Promise<HistoryResponse> {
  const response = await fetch(`/history?limit=${limit}`, { headers: headers(apiKey) });
  await raiseForStatus(response);
  return response.json();
}

export async function getConversation(
  id: string,
  apiKey: string | null,
): Promise<ConversationView> {
  const response = await fetch(`/conversations/${id}`, { headers: headers(apiKey) });
  await raiseForStatus(response);
  return response.json();
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onToolCall: (tool: string, query: string) => void;
  onSources: (sources: Source[]) => void;
  onDone: (done: StreamDone) => void;
  onError: (errorCode: string, detail: string) => void;
}

/** POST /chat/stream and dispatch each SSE event to the matching handler. */
export async function streamChat(options: {
  query: string;
  conversationId: string | null;
  apiKey: string | null;
  signal: AbortSignal;
  handlers: StreamHandlers;
}): Promise<void> {
  const { query, conversationId, apiKey, signal, handlers } = options;
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: headers(apiKey, true),
    body: JSON.stringify({ query, conversation_id: conversationId }),
    signal,
  });
  await raiseForStatus(response);
  if (!response.body) throw new ApiError("The server returned no stream.", 502);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (block: string) => {
    let name = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!name || !data) return;
    const payload = JSON.parse(data);
    switch (name as StreamEventName) {
      case "token":
        handlers.onToken(payload.text ?? "");
        break;
      case "tool_call":
        handlers.onToolCall(payload.tool ?? "", payload.query ?? "");
        break;
      case "sources":
        handlers.onSources(payload.sources ?? []);
        break;
      case "done":
        handlers.onDone(payload as StreamDone);
        break;
      case "error":
        handlers.onError(payload.error_code ?? "error", payload.detail ?? "Unknown error.");
        break;
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separator: number;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      if (block.trim()) dispatch(block);
    }
  }
}

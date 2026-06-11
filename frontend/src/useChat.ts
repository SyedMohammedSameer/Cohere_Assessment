/**
 * Chat state for the app: the active conversation's messages, the sidebar's
 * history, streaming send, and conversation switching. All server interaction
 * goes through api.ts; this hook owns optimistic updates and stream plumbing.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  Citation,
  ConversationView,
  Source,
  Usage,
  getConversation,
  getHistory,
  streamChat,
} from "./api";

export interface UiMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  toolQuery?: string | null;
  sources?: Source[];
  citations?: Citation[];
  usage?: Usage | null;
  latencyMs?: number | null;
  toolInvocations?: number | null;
  error?: string | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: string;
  turns: number;
}

let counter = 0;
const uid = () => `m${++counter}-${Date.now()}`;

const API_KEY_STORAGE = "cohere-chat-api-key";

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [apiKey, setApiKeyState] = useState<string | null>(
    () => localStorage.getItem(API_KEY_STORAGE) || null,
  );
  const abortRef = useRef<AbortController | null>(null);

  const setApiKey = useCallback((key: string | null) => {
    if (key) localStorage.setItem(API_KEY_STORAGE, key);
    else localStorage.removeItem(API_KEY_STORAGE);
    setApiKeyState(key);
  }, []);

  const patch = useCallback((id: string, change: (message: UiMessage) => UiMessage) => {
    setMessages((previous) => previous.map((m) => (m.id === id ? change(m) : m)));
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const history = await getHistory(apiKey);
      setHistoryError(null);
      setConversations(
        history.conversations.map((conversation) => ({
          id: conversation.id,
          title: conversation.turns[0]?.query ?? "Empty conversation",
          updatedAt: conversation.updated_at,
          turns: conversation.turns.length,
        })),
      );
      const latest = history.conversations[0];
      if (latest) setModel(latest.model);
    } catch (error) {
      setConversations([]);
      setHistoryError(
        error instanceof ApiError && error.status === 401
          ? "Locked. Set your API key below."
          : "Could not load history.",
      );
    }
  }, [apiKey]);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const newChat = useCallback(() => {
    stop();
    setMessages([]);
    setConversationId(null);
  }, [stop]);

  const loadConversation = useCallback(
    async (id: string) => {
      stop();
      try {
        const conversation: ConversationView = await getConversation(id, apiKey);
        setConversationId(conversation.id);
        setModel(conversation.model);
        setMessages(
          conversation.turns.flatMap((turn): UiMessage[] => [
            { id: uid(), role: "user", text: turn.query },
            {
              id: uid(),
              role: "assistant",
              text: turn.response,
              sources: turn.sources,
              citations: turn.citations,
              usage: turn.usage ?? null,
              latencyMs: turn.latency_ms ?? null,
              toolInvocations: turn.tool_invocations ?? null,
            },
          ]),
        );
      } catch (error) {
        setMessages([
          {
            id: uid(),
            role: "assistant",
            text: "",
            error:
              error instanceof ApiError ? error.message : "Could not load that conversation.",
          },
        ]);
      }
    },
    [apiKey, stop],
  );

  const send = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || sending) return;

      const userMessage: UiMessage = { id: uid(), role: "user", text: trimmed };
      const draftId = uid();
      const draft: UiMessage = { id: draftId, role: "assistant", text: "", streaming: true };
      setMessages((previous) => [...previous, userMessage, draft]);
      setSending(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const startedAt = performance.now();

      try {
        await streamChat({
          query: trimmed,
          conversationId,
          apiKey,
          signal: controller.signal,
          handlers: {
            onToolCall: (_tool, toolQuery) => patch(draftId, (m) => ({ ...m, toolQuery })),
            onSources: (sources) => patch(draftId, (m) => ({ ...m, sources })),
            onToken: (text) =>
              patch(draftId, (m) => ({ ...m, text: m.text + text, toolQuery: null })),
            onDone: (done) => {
              setConversationId(done.conversation_id);
              setModel(done.model);
              patch(draftId, (m) => ({
                ...m,
                text: done.response || m.text,
                streaming: false,
                toolQuery: null,
                usage: done.usage ?? null,
                sources: done.sources.length ? done.sources : m.sources,
                citations: done.citations ?? [],
                latencyMs: performance.now() - startedAt,
              }));
              void refreshHistory();
            },
            onError: (_code, detail) =>
              patch(draftId, (m) => ({ ...m, streaming: false, toolQuery: null, error: detail })),
          },
        });
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          patch(draftId, (m) => ({ ...m, streaming: false, toolQuery: null }));
        } else {
          const detail =
            error instanceof ApiError ? error.message : "Could not reach the server.";
          patch(draftId, (m) => ({ ...m, streaming: false, toolQuery: null, error: detail }));
        }
      } finally {
        setSending(false);
        abortRef.current = null;
      }
    },
    [apiKey, conversationId, patch, refreshHistory, sending],
  );

  return {
    messages,
    conversations,
    conversationId,
    historyError,
    sending,
    model,
    apiKey,
    setApiKey,
    send,
    stop,
    newChat,
    loadConversation,
  };
}

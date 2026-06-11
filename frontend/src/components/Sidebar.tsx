/** Conversation list, new-chat action, and the optional API key field. */

import { useState } from "react";
import { CloseIcon, CohereMark } from "./Logo";
import { ConversationSummary } from "../useChat";

function relativeTime(iso: string): string {
  const then = new Date(iso.endsWith("Z") ? iso : iso + "Z").getTime();
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function Sidebar(props: {
  open: boolean;
  conversations: ConversationSummary[];
  activeId: string | null;
  historyError: string | null;
  apiKey: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onClose: () => void;
  onSetApiKey: (key: string | null) => void;
}) {
  const {
    open,
    conversations,
    activeId,
    historyError,
    apiKey,
    onSelect,
    onNewChat,
    onClose,
    onSetApiKey,
  } = props;
  const [keyDraft, setKeyDraft] = useState(apiKey ?? "");

  return (
    <aside
      className={`fixed z-40 flex h-full w-[17rem] shrink-0 transform flex-col border-r border-line bg-panel transition-transform duration-200 ease-out md:relative md:z-auto ${
        open ? "translate-x-0" : "-translate-x-full md:hidden"
      }`}
    >
      <div className="flex items-center gap-2.5 px-4 pt-5 pb-4">
        <CohereMark className="h-6 w-6 shrink-0" />
        <div className="min-w-0 leading-tight">
          <p className="text-[15px] font-semibold lowercase tracking-tight">cohere</p>
          <p className="truncate text-[10px] text-ink-dim">Wikipedia-grounded chat</p>
        </div>
        <button
          onClick={onClose}
          aria-label="Close sidebar"
          className="ml-auto rounded-lg p-1.5 text-ink-dim transition hover:bg-panel-2 hover:text-ink"
        >
          <CloseIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="px-3">
        <button
          onClick={onNewChat}
          className="w-full rounded-xl bg-accent/90 px-3 py-2.5 text-sm font-medium text-black transition hover:bg-accent"
        >
          + New conversation
        </button>
      </div>

      <p className="px-4 pt-5 pb-2 text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
        History
      </p>

      <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {historyError && <p className="px-2 py-1 text-xs text-ink-dim">{historyError}</p>}
        {!historyError && conversations.length === 0 && (
          <p className="px-2 py-1 text-xs text-ink-dim">No conversations yet.</p>
        )}
        {conversations.map((conversation) => (
          <button
            key={conversation.id}
            onClick={() => onSelect(conversation.id)}
            className={`w-full rounded-lg px-3 py-2.5 text-left transition ${
              conversation.id === activeId
                ? "bg-panel-2 ring-1 ring-line"
                : "hover:bg-panel-2/60"
            }`}
          >
            <p className="truncate text-[13px] leading-snug">{conversation.title}</p>
            <p className="mt-0.5 text-[11px] text-ink-dim">
              {conversation.turns} {conversation.turns === 1 ? "turn" : "turns"} ·{" "}
              {relativeTime(conversation.updatedAt)}
            </p>
          </button>
        ))}
      </nav>

      <div className="border-t border-line p-3">
        <label className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-ink-dim">
          API key (optional)
        </label>
        <div className="flex gap-1.5">
          <input
            type="password"
            value={keyDraft}
            onChange={(event) => setKeyDraft(event.target.value)}
            placeholder="X-API-Key"
            className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs outline-none placeholder:text-ink-dim/60 focus:border-accent/50"
          />
          <button
            onClick={() => onSetApiKey(keyDraft.trim() || null)}
            className="rounded-lg border border-line px-2.5 text-xs text-ink-dim transition hover:border-accent/50 hover:text-ink"
          >
            Set
          </button>
        </div>
      </div>
    </aside>
  );
}

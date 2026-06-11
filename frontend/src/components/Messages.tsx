/** Message rendering: bubbles, streaming caret, tool status, sources, citations. */

import { Fragment, ReactNode } from "react";
import { Citation, Source } from "../api";
import { CohereMark } from "./Logo";
import { UiMessage } from "../useChat";

function sourceNumber(sources: Source[], id: string): number {
  return sources.findIndex((source) => source.id === id) + 1;
}

/** Answer text with cited spans underlined and superscripted by source number. */
function CitedText({
  text,
  citations,
  sources,
}: {
  text: string;
  citations?: Citation[];
  sources?: Source[];
}) {
  const usable = (citations ?? []).filter(
    (c): c is Citation & { start: number; end: number } =>
      typeof c.start === "number" && typeof c.end === "number" && c.start < c.end,
  );
  if (!usable.length || !sources?.length) return <>{text}</>;

  const sorted = [...usable].sort((a, b) => a.start - b.start);
  const nodes: ReactNode[] = [];
  let position = 0;
  for (const citation of sorted) {
    if (citation.start < position || citation.end > text.length) continue;
    if (citation.start > position) nodes.push(text.slice(position, citation.start));
    const numbers = citation.source_ids
      .map((id) => sourceNumber(sources, id))
      .filter((n) => n > 0);
    const titles = citation.source_ids
      .map((id) => sources.find((source) => source.id === id)?.title)
      .filter(Boolean)
      .join(", ");
    nodes.push(
      <span key={citation.start} className="cited" title={titles}>
        {text.slice(citation.start, citation.end)}
        {numbers.length > 0 && (
          <sup className="ml-0.5 text-[10px] font-medium text-accent">
            {numbers.join(",")}
          </sup>
        )}
      </span>,
    );
    position = citation.end;
  }
  nodes.push(text.slice(position));
  return <>{nodes.map((node, index) => <Fragment key={index}>{node}</Fragment>)}</>;
}

function SourceChips({ sources }: { sources: Source[] }) {
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {sources.map((source, index) => (
        <a
          key={source.id}
          href={source.url}
          target="_blank"
          rel="noreferrer"
          title={source.snippet ?? undefined}
          className="group flex max-w-64 items-center gap-1.5 rounded-full border border-line bg-panel px-2.5 py-1 text-[11px] text-ink-dim transition hover:border-accent/40 hover:text-ink"
        >
          <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[9px] font-semibold text-accent">
            {index + 1}
          </span>
          <span className="truncate">{source.title}</span>
        </a>
      ))}
    </div>
  );
}

function StatsLine({ message }: { message: UiMessage }) {
  const parts: string[] = [];
  const usage = message.usage;
  if (usage?.input_tokens != null && usage.output_tokens != null) {
    parts.push(`${usage.input_tokens.toLocaleString()} in / ${usage.output_tokens} out`);
  }
  if (message.latencyMs != null) parts.push(`${(message.latencyMs / 1000).toFixed(1)}s`);
  if (message.toolInvocations) {
    parts.push(`${message.toolInvocations} tool ${message.toolInvocations === 1 ? "call" : "calls"}`);
  }
  if (!parts.length) return null;
  return <p className="mt-2 text-[11px] text-ink-dim/80">{parts.join(" · ")}</p>;
}

function ToolStatus({ query }: { query: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg bg-accent-soft px-3 py-2 text-[13px] text-accent">
      <span className="h-3 w-3 animate-spin rounded-full border-[1.5px] border-accent/30 border-t-accent" />
      Searching Wikipedia{query ? `: "${query}"` : ""}
    </div>
  );
}

export function MessageRow({ message }: { message: UiMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] rounded-2xl rounded-br-md bg-panel-2 px-4 py-2.5 text-[15px] leading-relaxed ring-1 ring-line">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-panel-2 ring-1 ring-line">
        <CohereMark className="h-4 w-4" />
      </div>
      <div className="min-w-0 max-w-[85%]">
        {message.toolQuery != null && <ToolStatus query={message.toolQuery} />}
        {message.error ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-[13px] text-red-300">
            {message.error}
          </div>
        ) : (
          (message.text || message.streaming) && (
            <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
              <CitedText
                text={message.text}
                citations={message.citations}
                sources={message.sources}
              />
              {message.streaming && <span className="caret" />}
            </p>
          )
        )}
        {!message.streaming && !!message.sources?.length && (
          <SourceChips sources={message.sources} />
        )}
        {!message.streaming && <StatsLine message={message} />}
      </div>
    </div>
  );
}

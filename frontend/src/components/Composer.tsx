/** Message input: Enter to send, Shift+Enter for a newline, Stop while streaming. */

import { FormEvent, KeyboardEvent, useRef, useState } from "react";

export function Composer(props: {
  sending: boolean;
  onSend: (query: string) => void;
  onStop: () => void;
}) {
  const { sending, onSend, onStop } = props;
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (sending || !value.trim()) return;
    onSend(value);
    setValue("");
    textareaRef.current?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-3xl px-6 pb-6">
      <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel p-2 shadow-[0_-8px_30px_rgba(0,0,0,0.35)] focus-within:border-accent/40">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask anything. Answers are grounded in Wikipedia."
          className="max-h-40 min-h-[42px] flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] leading-relaxed outline-none placeholder:text-ink-dim/60"
        />
        {sending ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl border border-line px-4 py-2.5 text-sm text-ink-dim transition hover:border-red-400/40 hover:text-red-300"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            className="rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-black transition enabled:hover:brightness-110 disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
      <p className="mt-2 text-center text-[11px] text-ink-dim/70">
        Enter to send · Shift+Enter for a new line
      </p>
    </form>
  );
}

# Cohere Chat frontend

A React + TypeScript + Vite single-page app (styled with Tailwind CSS v4) for
the Cohere Chat backend. It streams answers token by token over server-sent
events, shows a "Searching Wikipedia" indicator while the tool runs, renders
numbered source chips and inline citations, and lists past conversations from
the history endpoints.

## Getting started

The backend must be running first (see the project root README), by default on
`http://127.0.0.1:8000`.

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The Vite dev server proxies
`/chat`, `/history`, `/conversations`, and `/health` to the backend, so there is
no CORS setup. Point at a different backend with `VITE_API_TARGET`:

```bash
VITE_API_TARGET=http://127.0.0.1:9000 npm run dev
```

## Features

- **Streaming chat** over `POST /chat/stream`, with a live token caret.
- **Grounding cues**: a tool-call status while searching, numbered and clickable
  Wikipedia source chips, and underlined cited spans (on history-loaded turns).
- **Multi-turn**: follow-ups continue the same conversation; the sidebar lists
  history and lets you reopen any conversation.
- **Per-turn stats**: token usage, latency, and tool-call count.
- **Optional API key**: set an `X-API-Key` to use the backend's auth mode; the
  history view is then scoped to that key.

## Build

```bash
npm run build    # type-checks then builds to dist/
```

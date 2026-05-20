"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import {
  Bot,
  ChevronDown,
  Copy,
  Database,
  FileText,
  KeyRound,
  LogOut,
  RefreshCw,
  Send,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  User,
} from "lucide-react";
import type { ChatRequest, ChatResponse, Citation, DocumentListResponse, DocumentRecord } from "@/lib/chat";

type Role = "assistant" | "user";

type Message = {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  confidence?: number | null;
  abstained?: boolean;
  pending?: boolean;
};

type TokenResponse = {
  accessToken: string;
  tokenType: string;
  expiresAt: string;
};

const prompts = [
  "How does RAG reduce hallucination?",
  "What are embeddings used for in RAG?",
  "How can we evaluate a RAG system?",
];

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildFilters(source: string): Record<string, unknown> {
  return source === "all" ? {} : { source };
}

function parseSseChunk(
  chunk: string,
  onToken: (token: string) => void,
  onDone: () => void,
) {
  const events = chunk.split("\n\n");

  for (const event of events) {
    const dataLine = event
      .split("\n")
      .find((line) => line.startsWith("data:"));

    if (!dataLine) {
      continue;
    }

    const raw = dataLine.slice("data:".length).trim();
    if (!raw) {
      onDone();
      continue;
    }

    try {
      const parsed = JSON.parse(raw) as { token?: string; done?: boolean };
      if (parsed.done) {
        onDone();
      }
      if (parsed.token) {
        onToken(parsed.token);
      }
    } catch {
      onToken(raw);
    }
  }
}

async function fetchJson<T>(
  url: string,
  payload: ChatRequest,
  accessToken: string | null,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      ...(accessToken ? { authorization: `Bearer ${accessToken}` } : {}),
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Ask something from the indexed RAG source.",
    },
  ]);
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(3);
  const [source, setSource] = useState("book");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState("demo");
  const [password, setPassword] = useState("demo123");
  const [authError, setAuthError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [uploadSource, setUploadSource] = useState("");
  const [uploading, setUploading] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const canSend = input.trim().length > 0 && !streaming && Boolean(token);
  const filters = useMemo(() => buildFilters(source), [source]);

  useEffect(() => {
    const savedToken = window.localStorage.getItem("rag-chat-token");
    if (savedToken) {
      setToken(savedToken);
    }
  }, []);

  useEffect(() => {
    if (token) {
      void refreshDocuments(token);
    } else {
      setDocuments([]);
    }
  }, [token]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function hydrateCitations(payload: ChatRequest, assistantId: string) {
    try {
      const finalResponse = await fetchJson<ChatResponse>("/api/chat", payload, token);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: finalResponse.answer || message.content,
                citations: finalResponse.citations,
                confidence: finalResponse.confidence,
                abstained: finalResponse.abstained,
                pending: false,
              }
            : message,
        ),
      );
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, pending: false } : message,
        ),
      );
    }
  }

  async function refreshDocuments(accessToken = token) {
    if (!accessToken) {
      return;
    }

    try {
      setDocumentError(null);
      const response = await fetch("/api/documents", {
        headers: {
          authorization: `Bearer ${accessToken}`,
        },
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const body = (await response.json()) as DocumentListResponse;
      setDocuments(body.documents);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Could not load documents";
      setDocumentError(message);
    }
  }

  async function uploadDocument(file: File | undefined) {
    if (!file || !token) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    if (uploadSource.trim()) {
      formData.append("source", uploadSource.trim());
    }

    try {
      setUploading(true);
      setDocumentError(null);
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        headers: {
          authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      setUploadSource("");
      await refreshDocuments(token);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Upload failed";
      setDocumentError(message);
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    if (!token) {
      return;
    }

    try {
      setDocumentError(null);
      const response = await fetch(`/api/documents/${encodeURIComponent(documentId)}`, {
        method: "DELETE",
        headers: {
          authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      await refreshDocuments(token);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Delete failed";
      setDocumentError(message);
    }
  }

  async function submitChat(messageText = input) {
    const question = messageText.trim();
    if (!question || streaming || !token) {
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    const assistantId = createId();
    const payload: ChatRequest = {
      message: question,
      topK,
      filters,
    };

    setError(null);
    setStreaming(true);
    setInput("");
    setMessages((current) => [
      ...current,
      { id: createId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", pending: true },
    ]);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          authorization: `Bearer ${token}`,
          accept: "text/event-stream",
          "content-type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const boundary = buffer.lastIndexOf("\n\n");
        if (boundary === -1) {
          continue;
        }

        const ready = buffer.slice(0, boundary + 2);
        buffer = buffer.slice(boundary + 2);

        parseSseChunk(
          ready,
          (token) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: `${message.content}${token}` }
                  : message,
              ),
            );
          },
          () => undefined,
        );
      }

      if (buffer.trim()) {
        parseSseChunk(
          buffer,
          (token) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: `${message.content}${token}` }
                  : message,
              ),
            );
          },
          () => undefined,
        );
      }

      await hydrateCitations(payload, assistantId);
    } catch (caught) {
      if (controller.signal.aborted) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, pending: false } : message,
          ),
        );
        return;
      }

      const message = caught instanceof Error ? caught.message : "Chat request failed";
      setError(message);
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: "The gateway did not return a response.",
                pending: false,
              }
            : item,
        ),
      );
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthError(null);

    try {
      const response = await fetch("/api/token", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error("Invalid username or password");
      }

      const body = (await response.json()) as TokenResponse;
      setToken(body.accessToken);
      window.localStorage.setItem("rag-chat-token", body.accessToken);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Sign in failed";
      setAuthError(message);
    }
  }

  function signOut() {
    stopStreaming();
    setToken(null);
    window.localStorage.removeItem("rag-chat-token");
  }

  function stopStreaming() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitChat();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitChat();
    }
  }

  function clearChat() {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Ask something from the indexed RAG source.",
      },
    ]);
    setError(null);
  }

  return (
    <main className="flex h-screen min-h-screen bg-ink-950 text-neutral-100">
      <aside className="hidden w-[292px] shrink-0 border-r border-white/10 bg-black/70 p-4 lg:flex lg:flex-col">
        <div className="flex h-11 items-center gap-3 px-2">
          <div className="grid size-8 place-items-center rounded-full border border-white/15 bg-white text-black">
            <Sparkles size={17} strokeWidth={2.4} />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">RAG Chat</h1>
            <p className="truncate text-xs text-neutral-500">Gateway localhost:8080</p>
          </div>
        </div>

        <div className="mt-5 space-y-2">
          <form className="mb-4 space-y-2 rounded-lg border border-white/10 bg-white/[0.03] p-3" onSubmit={signIn}>
            <div className="flex items-center gap-2 text-xs font-medium text-neutral-400">
              <KeyRound size={14} />
              <span>{token ? "Authenticated" : "Sign in"}</span>
            </div>
            {!token && (
              <>
                <input
                  className="h-10 w-full rounded-md border border-white/10 bg-black px-3 text-sm text-neutral-200 outline-none transition placeholder:text-neutral-600 focus:border-white/30"
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Username"
                  value={username}
                />
                <input
                  className="h-10 w-full rounded-md border border-white/10 bg-black px-3 text-sm text-neutral-200 outline-none transition placeholder:text-neutral-600 focus:border-white/30"
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Password"
                  type="password"
                  value={password}
                />
                {authError && <p className="text-xs text-red-300">{authError}</p>}
                <button
                  className="h-10 w-full rounded-md bg-white text-sm font-medium text-black transition hover:bg-neutral-200"
                  type="submit"
                >
                  Sign in
                </button>
              </>
            )}
            {token && (
              <button
                className="h-10 w-full rounded-md border border-white/10 text-sm text-neutral-300 transition hover:border-white/20 hover:bg-white/[0.06]"
                onClick={signOut}
                type="button"
              >
                Sign out
              </button>
            )}
          </form>

          {prompts.map((prompt) => (
            <button
              className="w-full rounded-lg border border-white/10 bg-white/[0.03] px-3 py-3 text-left text-sm text-neutral-300 transition hover:border-white/20 hover:bg-white/[0.07]"
              disabled={streaming || !token}
              key={prompt}
              onClick={() => void submitChat(prompt)}
              type="button"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="mt-4 space-y-3 rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs font-medium text-neutral-400">
              <FileText size={14} />
              <span>Documents</span>
            </div>
            <button
              aria-label="Refresh documents"
              className="grid size-7 place-items-center rounded-md text-neutral-500 transition hover:bg-white/[0.06] hover:text-white"
              disabled={!token || uploading}
              onClick={() => void refreshDocuments()}
              type="button"
            >
              <RefreshCw className={uploading ? "animate-spin" : ""} size={13} />
            </button>
          </div>

          <input
            className="h-9 w-full rounded-md border border-white/10 bg-black px-3 text-xs text-neutral-300 outline-none transition placeholder:text-neutral-600 focus:border-white/30"
            disabled={!token || uploading}
            onChange={(event) => setUploadSource(event.target.value)}
            placeholder="Source label"
            value={uploadSource}
          />

          <label
            className={clsx(
              "flex h-10 cursor-pointer items-center justify-center gap-2 rounded-md border border-white/10 text-sm text-neutral-300 transition",
              token && !uploading
                ? "hover:border-white/20 hover:bg-white/[0.06]"
                : "cursor-not-allowed opacity-50",
            )}
          >
            <Upload size={14} />
            <span>{uploading ? "Indexing" : "Upload PDF/TXT"}</span>
            <input
              accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain,text/markdown"
              className="hidden"
              disabled={!token || uploading}
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                void uploadDocument(file);
              }}
              type="file"
            />
          </label>

          {documentError && <p className="text-xs text-red-300">{documentError}</p>}

          <div className="max-h-44 space-y-2 overflow-y-auto pr-1">
            {documents.length === 0 && (
              <p className="rounded-md bg-white/[0.03] px-2.5 py-2 text-xs text-neutral-500">
                No uploaded documents yet.
              </p>
            )}
            {documents.map((document) => (
              <div
                className="rounded-md border border-white/10 bg-black/40 px-2.5 py-2"
                key={document.document_id}
              >
                <div className="flex items-start gap-2">
                  <FileText className="mt-0.5 shrink-0 text-neutral-500" size={13} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-neutral-300">{document.filename}</p>
                    <p className="text-[11px] text-neutral-600">{document.chunk_count} chunks</p>
                  </div>
                  <button
                    aria-label={`Delete ${document.filename}`}
                    className="grid size-6 shrink-0 place-items-center rounded text-neutral-600 transition hover:bg-white/[0.08] hover:text-white"
                    onClick={() => void deleteDocument(document.document_id)}
                    type="button"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-auto space-y-3 rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <div className="flex items-center gap-2 text-xs font-medium text-neutral-400">
            <SlidersHorizontal size={14} />
            <span>Retrieval</span>
          </div>

          <label className="block">
            <span className="mb-2 block text-xs text-neutral-500">topK {topK}</span>
            <input
              className="h-1.5 w-full accent-white"
              disabled={streaming || !token}
              max={10}
              min={1}
              onChange={(event) => setTopK(Number(event.target.value))}
              type="range"
              value={topK}
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-xs text-neutral-500">source</span>
            <span className="relative block">
              <select
                className="h-10 w-full appearance-none rounded-md border border-white/10 bg-black px-3 text-sm text-neutral-200 outline-none transition focus:border-white/30"
                disabled={streaming || !token}
                onChange={(event) => setSource(event.target.value)}
                value={source}
              >
                <option value="book">book</option>
                <option value="all">all</option>
              </select>
              <ChevronDown
                aria-hidden="true"
                className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500"
                size={15}
              />
            </span>
          </label>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-white/10 bg-black/60 px-4 backdrop-blur md:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-8 place-items-center rounded-full border border-white/15 bg-white text-black lg:hidden">
              <Sparkles size={16} strokeWidth={2.4} />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h2 className="truncate text-sm font-semibold">RAG Chat</h2>
                <span
                  className={clsx(
                    "size-2 rounded-full",
                    streaming ? "bg-neutral-300" : "bg-emerald-400",
                  )}
                />
              </div>
              <p className="truncate text-xs text-neutral-500">
                {streaming ? "generating" : "ready"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {token && (
              <button
                aria-label="Sign out"
                className="grid size-9 place-items-center rounded-md border border-white/10 text-neutral-400 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-white"
                onClick={signOut}
                type="button"
              >
                <LogOut size={16} />
              </button>
            )}
            <button
              aria-label="Clear chat"
              className="grid size-9 place-items-center rounded-md border border-white/10 text-neutral-400 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-white"
              onClick={clearChat}
              type="button"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-6">
          <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
            {!token && (
              <form
                className="mx-auto w-full max-w-sm rounded-xl border border-white/10 bg-ink-900/95 p-4 shadow-glow lg:hidden"
                onSubmit={signIn}
              >
                <div className="mb-3 flex items-center gap-2 text-sm font-medium text-neutral-200">
                  <KeyRound size={16} />
                  <span>Sign in</span>
                </div>
                <div className="space-y-2">
                  <input
                    className="h-11 w-full rounded-md border border-white/10 bg-black px-3 text-sm text-neutral-200 outline-none transition placeholder:text-neutral-600 focus:border-white/30"
                    onChange={(event) => setUsername(event.target.value)}
                    placeholder="Username"
                    value={username}
                  />
                  <input
                    className="h-11 w-full rounded-md border border-white/10 bg-black px-3 text-sm text-neutral-200 outline-none transition placeholder:text-neutral-600 focus:border-white/30"
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Password"
                    type="password"
                    value={password}
                  />
                  {authError && <p className="text-xs text-red-300">{authError}</p>}
                  <button
                    className="h-11 w-full rounded-md bg-white text-sm font-medium text-black transition hover:bg-neutral-200"
                    type="submit"
                  >
                    Sign in
                  </button>
                </div>
              </form>
            )}

            {messages.map((message) => (
              <article
                className={clsx(
                  "flex gap-3",
                  message.role === "user" ? "justify-end" : "justify-start",
                )}
                key={message.id}
              >
                {message.role === "assistant" && (
                  <div className="mt-1 grid size-8 shrink-0 place-items-center rounded-full border border-white/10 bg-ink-800">
                    <Bot size={16} />
                  </div>
                )}

                <div
                  className={clsx(
                    "min-w-0 max-w-[82%] rounded-xl border px-4 py-3 shadow-glow md:max-w-[72%]",
                    message.role === "user"
                      ? "border-white/12 bg-white text-black"
                      : "border-white/10 bg-ink-900/95 text-neutral-100",
                  )}
                >
                  <div className="whitespace-pre-wrap break-words text-[15px] leading-7">
                    {message.content || (
                      <span className="inline-flex items-center gap-2 text-neutral-500">
                        <RefreshCw className="animate-spin" size={14} />
                        Thinking
                      </span>
                    )}
                  </div>

                  {message.role === "assistant" && message.content && (
                    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/10 pt-3">
                      <button
                        aria-label="Copy answer"
                        className="grid size-8 place-items-center rounded-md text-neutral-500 transition hover:bg-white/[0.06] hover:text-white"
                        onClick={() => void navigator.clipboard.writeText(message.content)}
                        type="button"
                      >
                        <Copy size={15} />
                      </button>
                      {typeof message.confidence === "number" && (
                        <span className="rounded-md border border-white/10 px-2 py-1 text-xs text-neutral-500">
                          {(message.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                      {message.abstained && (
                        <span className="rounded-md border border-white/10 px-2 py-1 text-xs text-neutral-500">
                          abstained
                        </span>
                      )}
                    </div>
                  )}

                  {message.citations && message.citations.length > 0 && (
                    <div className="mt-3 space-y-2 border-t border-white/10 pt-3">
                      {message.citations.slice(0, 4).map((citation) => (
                        <div
                          className="flex items-center gap-2 rounded-md bg-white/[0.04] px-2.5 py-2 text-xs text-neutral-400"
                          key={`${message.id}-${citation.chunkId}`}
                        >
                          <Database className="shrink-0" size={13} />
                          <span className="min-w-0 flex-1 truncate">{citation.chunkId}</span>
                          {citation.pageNumber !== null && (
                            <span className="shrink-0 text-neutral-500">p.{citation.pageNumber}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {message.role === "user" && (
                  <div className="mt-1 grid size-8 shrink-0 place-items-center rounded-full border border-white/10 bg-white text-black">
                    <User size={16} />
                  </div>
                )}
              </article>
            ))}
          </div>
        </div>

        <footer className="shrink-0 border-t border-white/10 bg-black/75 px-4 py-4 backdrop-blur md:px-6">
          <div className="mx-auto w-full max-w-4xl">
            {error && (
              <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                {error}
              </div>
            )}

            <form
              className="flex items-end gap-2 rounded-xl border border-white/12 bg-ink-900 p-2 shadow-glow focus-within:border-white/25"
              onSubmit={handleSubmit}
            >
              <textarea
                aria-label="Message"
                className="max-h-36 min-h-12 flex-1 bg-transparent px-3 py-3 text-[15px] leading-6 text-neutral-100 outline-none placeholder:text-neutral-600"
                disabled={streaming || !token}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={token ? "Ask anything" : "Sign in to chat"}
                rows={1}
                value={input}
              />

              {streaming ? (
                <button
                  aria-label="Stop"
                  className="grid size-11 shrink-0 place-items-center rounded-lg bg-white text-black transition hover:bg-neutral-200"
                  onClick={stopStreaming}
                  type="button"
                >
                  <Square size={16} fill="currentColor" />
                </button>
              ) : (
                <button
                  aria-label="Send"
                  className="grid size-11 shrink-0 place-items-center rounded-lg bg-white text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-500"
                  disabled={!canSend}
                  type="submit"
                >
                  <Send size={17} />
                </button>
              )}
            </form>
          </div>
        </footer>
      </section>
    </main>
  );
}

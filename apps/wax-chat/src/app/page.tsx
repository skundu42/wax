"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Clipboard,
  Cpu,
  Loader2,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Square,
  Terminal,
  Trash2,
  User
} from "lucide-react";
import styles from "./page.module.css";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
};

const defaultModel = "./models/TinyLlama-1.1B-Chat-v1.0";
const defaultSystem = "You are concise and practical.";

export default function ChatPage() {
  const [model, setModel] = useState(defaultModel);
  const [system, setSystem] = useState(defaultSystem);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [maxNewTokens, setMaxNewTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(
    () => model.trim().length > 0 && input.trim().length > 0 && !isRunning,
    [input, isRunning, model]
  );
  const userTurns = messages.filter((message) => message.role === "user").length;
  const assistantTurns = messages.filter((message) => message.role === "assistant").length;
  const modelName = model.split("/").filter(Boolean).at(-1) ?? model;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSend) {
      return;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim()
    };
    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: ""
    };
    const nextMessages = [...messages, userMessage, assistantMessage];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setIsRunning(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model,
          system,
          messages: nextMessages
            .filter((message) => message.role !== "assistant" || message.content)
            .map(({ role, content }) => ({ role, content })),
          maxNewTokens,
          temperature,
          topP,
          stop: ["</s>"]
        })
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error ?? `Request failed with ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let content = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        content += decoder.decode(value, { stream: true });
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessage.id ? { ...message, content } : message
          )
        );
      }
      content += decoder.decode();
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessage.id ? { ...message, content } : message
        )
      );
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") {
        setError((caught as Error).message);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
    setIsRunning(false);
  }

  function clearChat() {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setIsRunning(false);
  }

  function resetSettings() {
    setModel(defaultModel);
    setSystem(defaultSystem);
    setMaxNewTokens(256);
    setTemperature(0.7);
    setTopP(0.9);
  }

  async function copyMessage(message: Message) {
    if (!message.content) {
      return;
    }
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId(null), 1100);
  }

  function submitOnShortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <main className={styles.shell}>
      <section className={styles.header}>
        <div className={styles.brand}>
          <Terminal size={21} />
          <div>
            <p className={styles.kicker}>local inference</p>
            <h1>wax chat</h1>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.modelPill} title={model}>
            <Cpu size={16} />
            <span>{modelName}</span>
          </div>
          <div className={styles.status} data-state={isRunning ? "running" : "ready"}>
            {isRunning ? <Loader2 className={styles.spin} size={17} /> : <Bot size={17} />}
            <span>{isRunning ? "generating" : "ready"}</span>
          </div>
        </div>
      </section>

      <section className={styles.workspace}>
        <aside className={styles.controls}>
          <div className={styles.panelHeader}>
            <div>
              <span>Session</span>
              <strong>{userTurns + assistantTurns} messages</strong>
            </div>
            <button type="button" onClick={clearChat} aria-label="Clear chat">
              <Trash2 size={16} />
            </button>
          </div>

          <div className={styles.metrics}>
            <div>
              <span>User</span>
              <strong>{userTurns}</strong>
            </div>
            <div>
              <span>Assistant</span>
              <strong>{assistantTurns}</strong>
            </div>
          </div>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>
              <Cpu size={15} />
              <span>Runtime</span>
            </div>
            <label>
              <span>Model path</span>
              <input value={model} onChange={(event) => setModel(event.target.value)} />
            </label>
            <label>
              <span>System</span>
              <textarea
                rows={4}
                value={system}
                onChange={(event) => setSystem(event.target.value)}
              />
            </label>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionTitle}>
              <SlidersHorizontal size={15} />
              <span>Sampling</span>
            </div>
            <label>
              <span>Max tokens</span>
              <div className={styles.controlRow}>
                <input
                  type="range"
                  min={16}
                  max={1024}
                  step={16}
                  value={maxNewTokens}
                  onChange={(event) => setMaxNewTokens(Number(event.target.value))}
                />
                <input
                  className={styles.number}
                  type="number"
                  min={1}
                  max={4096}
                  value={maxNewTokens}
                  onChange={(event) => setMaxNewTokens(Number(event.target.value))}
                />
              </div>
            </label>
            <label>
              <span>Temperature</span>
              <div className={styles.controlRow}>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
                <input
                  className={styles.number}
                  type="number"
                  min={0}
                  max={5}
                  step={0.1}
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
              </div>
            </label>
            <label>
              <span>Top-p</span>
              <div className={styles.controlRow}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={topP}
                  onChange={(event) => setTopP(Number(event.target.value))}
                />
                <input
                  className={styles.number}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={topP}
                  onChange={(event) => setTopP(Number(event.target.value))}
                />
              </div>
            </label>
          </section>

          <button className={styles.secondaryButton} type="button" onClick={resetSettings}>
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
        </aside>

        <section className={styles.chat}>
          <div className={styles.chatHeader}>
            <div>
              <span>Transcript</span>
              <strong>{modelName}</strong>
            </div>
            <div className={styles.paramStrip}>
              <span>{maxNewTokens} tok</span>
              <span>temp {temperature}</span>
              <span>top-p {topP}</span>
            </div>
          </div>

          <div className={styles.messages}>
            {messages.length === 0 ? (
              <div className={styles.empty}>
                <Bot size={34} />
                <span>No messages</span>
              </div>
            ) : (
              messages.map((message) => (
                <article
                  className={`${styles.message} ${styles[message.role]}`}
                  key={message.id}
                >
                  <header>
                    <span className={styles.avatar}>
                      {message.role === "user" ? <User size={15} /> : <Bot size={15} />}
                    </span>
                    <strong>{message.role}</strong>
                    <button
                      type="button"
                      onClick={() => copyMessage(message)}
                      disabled={!message.content}
                      aria-label={`Copy ${message.role} message`}
                    >
                      <Clipboard size={14} />
                      <span>{copiedId === message.id ? "Copied" : "Copy"}</span>
                    </button>
                  </header>
                  <p>{message.content || "..."}</p>
                </article>
              ))
            )}
            {isRunning ? (
              <div className={styles.typing}>
                <Loader2 className={styles.spin} size={15} />
                <span>Streaming</span>
              </div>
            ) : null}
            <div ref={messagesEndRef} />
          </div>
          {error ? <div className={styles.error}>{error}</div> : null}
          <form className={styles.composer} onSubmit={submit}>
            <textarea
              rows={3}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={submitOnShortcut}
              placeholder="Message TinyLlama"
            />
            {isRunning ? (
              <button type="button" onClick={stopGeneration} aria-label="Stop generation">
                <Square size={18} />
              </button>
            ) : (
              <button type="submit" disabled={!canSend} aria-label="Send message">
                <Send size={18} />
              </button>
            )}
          </form>
        </section>
      </section>
    </main>
  );
}

"use client";

import { useState } from "react";
import { Send, Sparkles } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { api, type Source } from "@/lib/api";

type Message = { role: "student" | "assistant"; text: string; sources?: Source[] };

export default function ChatPage() {
  const [courseId, setCourseId] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  async function ask(mode: "ask" | "summary" | "flashcards" = "ask") {
    if (!input.trim()) return;
    setMessages((items) => [...items, { role: "student", text: input }]);
    const result = await api<{ response: string; sources: Source[] }>(`/courses/${courseId}/chat/${mode}`, {
      method: "POST",
      body: JSON.stringify({ message: input }),
    });
    setMessages((items) => [...items, { role: "assistant", text: result.response, sources: result.sources }]);
    setInput("");
  }

  return (
    <AppShell role="student">
      <div className="grid gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-2xl font-semibold">Course Chat</h2>
          <CourseSelect value={courseId} onChange={setCourseId} />
        </div>
        <Card className="min-h-[420px]">
          <div className="grid gap-4">
            {messages.map((message, index) => (
              <div key={index} className={message.role === "student" ? "ml-auto max-w-2xl rounded-md bg-primary p-4 text-primary-foreground" : "max-w-3xl rounded-md border border-border p-4"}>
                <p className="whitespace-pre-wrap text-sm leading-7">{message.text}</p>
                {!!message.sources?.length && (
                  <div className="mt-4 grid gap-2">
                    {message.sources.map((source) => (
                      <p key={source.citation} className="rounded border border-border p-2 text-xs text-muted">
                        [{source.citation}] {source.document} {source.page ? `page ${source.page}` : ""}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
        <div className="grid gap-2">
          <Textarea placeholder="Ask a question grounded in this course..." value={input} onChange={(event) => setInput(event.target.value)} />
          <div className="flex gap-2">
            <Button onClick={() => ask("ask")}><Send className="size-4" /> Ask</Button>
            <Button variant="secondary" onClick={() => ask("summary")}><Sparkles className="size-4" /> Summary</Button>
            <Button variant="secondary" onClick={() => ask("flashcards")}><Sparkles className="size-4" /> Flashcards</Button>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

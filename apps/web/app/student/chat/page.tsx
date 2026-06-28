"use client";

import { useEffect, useRef, useState } from "react";

import ReactMarkdown from "react-markdown";

import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { api, type Source } from "@/lib/api";
import { Loader2, Send } from "lucide-react";

type Message = {
  role: "student" | "assistant";
  text: string;
  sources?: Source[];
  time?: string;
};

export default function ChatPage() {
  const [courseId, setCourseId] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  async function ask(
    mode: "ask" | "summary" | "flashcards" = "ask"
  ) {
    if (!input.trim() || !courseId || loading) return;

    const question = input;

    setMessages((items) => [
      ...items,
      {
        role: "student",
        text: question,
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ]);

    setInput("");
    setLoading(true);

    try {
      const result = await api<{
        response: string;
        sources: Source[];
      }>(`/courses/${courseId}/chat/${mode}`, {
        method: "POST",
        body: JSON.stringify({
          message: question,
        }),
      });

      setMessages((items) => [
        ...items,
        {
  role: "assistant",
  text: result.response,
  sources: result.sources,
  time: new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  }),
},
      ]);
    } catch (err) {
      console.error(err);

      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          text:
            "⚠️ Something went wrong while contacting the AI. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell role="student">
      <div className="flex h-[calc(100vh-80px)] flex-col overflow-hidden bg-background">

        {/* Header */}
       <div className="sticky top-0 z-20 bg-[#0f0f0f]/95 backdrop-blur-md">
  <div className="mx-auto flex max-w-7xl items-center px-6 py-2">
    <CourseSelect
      value={courseId}
      onChange={setCourseId}
    />
  </div>
</div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-5 py-6">

          {messages.length === 0 ? (

            <div className="flex h-full items-center justify-center">

              <div className="max-w-2xl text-center">

                <h1 className="text-xl font-bold tracking-tight">
                  What would you like to know about this subject?
                </h1>

                <p className="mt-5 text-lg text-muted-foreground">
                  Select a course and ask questions based on your uploaded PDFs,
                  PPTs, notes and videos.
                </p>

                {!courseId && (
                  <div className="mt-8 rounded-2xl border border-dashed p-8">

                    <p className="text-lg font-medium">
                      Select a course first
                    </p>

                    <p className="mt-2 text-sm text-muted-foreground">
                      CourseGPT will answer using only the materials uploaded
                      for the selected course.
                    </p>

                  </div>
                )}

              </div>

            </div>

          ) : (

            <div className="mx-auto max-w-4xl">

              <div className="space-y-8">

                {messages.map((message, index) => (

                  <div
                  key={index}
                    className={`flex ${
                      message.role === "student"
                      ? "justify-end"
                      : "justify-start"
                    }`}
                  >

                    {message.role === "student" ? (
                      <div className="max-w-[75%] rounded-3xl bg-primary px-5 py-4 text-primary-foreground shadow">
                        <div className="prose prose-sm max-w-none text-primary-foreground">
                          <ReactMarkdown>
                            {message.text}
                            </ReactMarkdown>
                            </div>
                            {message.time && (
                              <p className="mt-2 text-right text-xs opacity-70">
                                {message.time}
                                </p>
                              )}
                              </div>
                              ) : (

                      <div className="max-w-full rounded-2xl border bg-card px-6 py-5 shadow-sm">

                        <div className="mb-4 flex items-center gap-3">

                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                            CG
                          </div>

                          <span className="font-semibold">
                            CourseGPT
                          </span>

                        </div>
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown>
                            {message.text}
                            </ReactMarkdown>
                            </div>
                            {message.time && (
                              <p className="mt-3 text-xs text-muted-foreground">
                                {message.time}
                                </p>)}
                                {!!message.sources?.length && (

                          <div className="mt-5 flex flex-wrap gap-3">

                            {message.sources.map((source) => (

                              <div
                                key={source.citation}
                                className="rounded-xl border bg-background px-4 py-3 text-xs shadow-sm"
                              >

                                <div className="font-semibold text-primary">
                                  [{source.citation}]
                                </div>

                                <div className="mt-1">

                                  {source.document}

                                  {source.page
                                    ? ` • Page ${source.page}`
                                    : ""}

                                </div>

                              </div>

                            ))}

                          </div>

                        )}

                      </div>

                    )}

                  </div>

                ))}

                {loading && (

                  <div className="flex justify-start">

                    <div className="rounded-2xl border bg-card px-5 py-4 shadow-sm">

                      <div className="flex items-center gap-3">

                        <Loader2 className="h-5 w-5 animate-spin text-primary" />

                        <span className="text-sm text-muted-foreground">
                          CourseGPT is thinking...
                        </span>

                      </div>

                    </div>

                  </div>

                )}

                <div ref={bottomRef} />

              </div>

            </div>

          )}

        </div>

        {/* Input */}
               
<div className="bg-background px-4 py-4">
  <div className="mx-auto max-w-5xl">

    <div className="relative rounded-3xl border border-zinc-700 bg-card shadow-sm">

      <Textarea
        placeholder="Ask anything about your course..."
        value={input}
        disabled={loading}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            ask("ask");
          }
        }}
        className="min-h-[130px] resize-none border-0 bg-transparent px-5 pt-5 pb-16 text-base shadow-none focus-visible:ring-0"
      />

      <Button
        onClick={() => ask("ask")}
        disabled={!input.trim() || !courseId || loading}
        className="absolute bottom-4 right-4 h-11 rounded-full px-5"
      >
        {loading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Thinking...
          </>
        ) : (
          <>
            <Send className="mr-2 h-4 w-4" />
            Ask
          </>
        )}
      </Button>

    </div>

    <div className="mt-3 flex items-center justify-between px-2">
      <p className="text-xs text-muted-foreground">
        Press <kbd className="rounded border px-1">Enter</kbd> to send ·{" "}
        <kbd className="rounded border px-1">Shift + Enter</kbd> for a new line
      </p>

      <p className="text-xs text-muted-foreground">
        AI answers are based on this course's materials.
      </p>
    </div>

  </div>
</div>
</div>
    </AppShell>
  );
}
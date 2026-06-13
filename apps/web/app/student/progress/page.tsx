"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";

type Result = { id: string; score: number; created_at: string };

export default function ProgressPage() {
  const [results, setResults] = useState<Result[]>([]);
  useEffect(() => {
    api<Result[]>("/quiz-results").then(setResults);
  }, []);
  const average = results.length ? Math.round(results.reduce((sum, item) => sum + item.score, 0) / results.length) : 0;
  return (
    <AppShell role="student">
      <div className="grid gap-5">
        <h2 className="text-2xl font-semibold">Progress Tracking</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <Card><p className="text-sm text-muted">Quiz history</p><p className="mt-2 text-3xl font-semibold">{results.length}</p></Card>
          <Card><p className="text-sm text-muted">Average score</p><p className="mt-2 text-3xl font-semibold">{average}%</p></Card>
          <Card><p className="text-sm text-muted">Completed lessons</p><p className="mt-2 text-3xl font-semibold">0</p></Card>
        </div>
        <Card>
          <h3 className="font-semibold">Quiz history</h3>
          <div className="mt-4 grid gap-3">
            {results.map((result) => (
              <div key={result.id} className="flex justify-between rounded-md border border-border p-3 text-sm">
                <span>{new Date(result.created_at).toLocaleDateString()}</span>
                <span className="font-medium">{result.score}%</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

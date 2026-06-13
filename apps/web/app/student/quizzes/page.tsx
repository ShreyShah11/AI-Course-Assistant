"use client";

import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, type Quiz } from "@/lib/api";

export default function QuizPage() {
  const [courseId, setCourseId] = useState("");
  const [topic, setTopic] = useState("Practice quiz from current course materials");
  const [quiz, setQuiz] = useState<Quiz | null>(null);

  return (
    <AppShell role="student">
      <div className="grid gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-2xl font-semibold">Quiz Section</h2>
          <CourseSelect value={courseId} onChange={setCourseId} />
        </div>
        <Card>
          <div className="flex gap-2">
            <Input value={topic} onChange={(event) => setTopic(event.target.value)} />
            <Button onClick={async () => setQuiz(await api<Quiz>(`/courses/${courseId}/quizzes/generate`, { method: "POST", body: JSON.stringify({ topic, question_count: 8 }) }))}>
              Generate
            </Button>
          </div>
        </Card>
        {quiz && (
          <Card>
            <pre className="whitespace-pre-wrap text-sm leading-7">{quiz.generated_content.content}</pre>
          </Card>
        )}
      </div>
    </AppShell>
  );
}

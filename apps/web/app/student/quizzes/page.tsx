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
    <div className="mx-auto w-full max-w-6xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">
            Quiz Generator
          </h1>

          <p className="mt-1 text-muted-foreground">
            Generate quizzes directly from your course materials.
          </p>
        </div>

        <CourseSelect
          value={courseId}
          onChange={setCourseId}
        />
      </div>

      {/* Generator */}
      <Card className="p-6">
        <div className="space-y-4">
          <div>
            <label className="mb-2 block text-sm font-medium">
              Quiz Topic
            </label>

            <Input
              value={topic}
              onChange={(event) =>
                setTopic(event.target.value)
              }
              placeholder="Enter a topic..."
            />
          </div>

          <Button
            className="w-full"
            disabled={!courseId}
            onClick={async () =>
              setQuiz(
                await api<Quiz>(
                  `/courses/${courseId}/quizzes/generate`,
                  {
                    method: "POST",
                    body: JSON.stringify({
                      topic,
                      question_count: 8,
                    }),
                  }
                )
              )
            }
          >
            Generate Quiz
          </Button>
        </div>
      </Card>

      {/* Results */}
      {!quiz ? (
        <Card className="flex min-h-[350px] items-center justify-center">
          <div className="text-center">
            <h3 className="text-xl font-semibold">
              No Quiz Generated Yet
            </h3>

            <p className="mt-2 text-muted-foreground">
              Select a course and generate a quiz to begin.
            </p>
          </div>
        </Card>
      ) : (
        <Card className="p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">
              Generated Quiz
            </h2>
          </div>

          <div className="max-h-[600px] overflow-y-auto rounded-lg border bg-muted/20 p-4">
            <pre className="whitespace-pre-wrap text-sm leading-7">
              {quiz.generated_content.content}
            </pre>
          </div>
        </Card>
      )}
    </div>
  </AppShell>
);


}

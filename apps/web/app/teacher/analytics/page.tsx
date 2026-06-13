"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";

export default function AnalyticsPage() {
  const [courseId, setCourseId] = useState("");
  const [data, setData] = useState<Record<string, number>>({});

  useEffect(() => {
    if (courseId) api<Record<string, number>>(`/courses/${courseId}/analytics`).then(setData);
  }, [courseId]);

  return (
    <AppShell role="teacher">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-2xl font-semibold">Analytics</h2>
        <CourseSelect value={courseId} onChange={setCourseId} />
      </div>
      <div className="mt-5 grid gap-4 md:grid-cols-5">
        {Object.entries({
          Students: data.enrolled_students ?? 0,
          Materials: data.materials ?? 0,
          Chats: data.chat_messages ?? 0,
          Quizzes: data.quizzes ?? 0,
          "Avg score": `${data.average_score ?? 0}%`,
        }).map(([label, value]) => (
          <Card key={label}>
            <p className="text-sm text-muted">{label}</p>
            <p className="mt-2 text-3xl font-semibold">{value}</p>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}

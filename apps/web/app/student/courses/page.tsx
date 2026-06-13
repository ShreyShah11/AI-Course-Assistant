"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, type Course } from "@/lib/api";

export default function StudentCourses() {
  const [catalog, setCatalog] = useState<Course[]>([]);
  const [mine, setMine] = useState<Course[]>([]);

  async function load() {
    setCatalog(await api<Course[]>("/courses/catalog"));
    setMine(await api<Course[]>("/courses"));
  }

  useEffect(() => {
    load();
  }, []);

  const enrolled = new Set(mine.map((course) => course.id));
  return (
    <AppShell role="student">
      <h2 className="text-2xl font-semibold">My Courses</h2>
      <div className="mt-5 grid gap-3">
        {catalog.map((course) => (
          <Card key={course.id} className="flex items-start justify-between gap-4">
            <div>
              <h3 className="font-semibold">{course.title}</h3>
              <p className="text-sm text-muted">{course.description}</p>
            </div>
            <Button
              disabled={enrolled.has(course.id)}
              onClick={async () => {
                await api(`/courses/${course.id}/enroll`, { method: "POST" });
                load();
              }}
            >
              {enrolled.has(course.id) ? "Enrolled" : "Enroll"}
            </Button>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}

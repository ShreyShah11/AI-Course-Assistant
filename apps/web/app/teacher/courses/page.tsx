"use client";

import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { api, type Course } from "@/lib/api";

export default function TeacherCourses() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [form, setForm] = useState({ title: "", description: "" });

  async function load() {
    setCourses(await api<Course[]>("/courses"));
  }

  useEffect(() => {
    load();
  }, []);

  async function createCourse(event: React.FormEvent) {
    event.preventDefault();
    await api<Course>("/courses", { method: "POST", body: JSON.stringify(form) });
    setForm({ title: "", description: "" });
    load();
  }

  return (
    <AppShell role="teacher">
      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        <Card>
          <h2 className="text-xl font-semibold">Create Course</h2>
          <form onSubmit={createCourse} className="mt-5 grid gap-3">
            <Input placeholder="Course title" value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} />
            <Textarea placeholder="Description" value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} />
            <Button><Plus className="size-4" /> Create</Button>
          </form>
        </Card>
        <div className="grid gap-3">
          {courses.map((course) => (
            <Card key={course.id} className="flex items-start justify-between gap-4">
              <div>
                <h3 className="font-semibold">{course.title}</h3>
                <p className="text-sm leading-6 text-muted">{course.description}</p>
                <p className="mt-2 text-xs text-muted">Course ID: {course.id}</p>
              </div>
              <Button
                variant="ghost"
                onClick={async () => {
                  await api(`/courses/${course.id}`, { method: "DELETE" });
                  load();
                }}
              >
                <Trash2 className="size-4" />
              </Button>
            </Card>
          ))}
        </div>
      </div>
    </AppShell>
  );
}

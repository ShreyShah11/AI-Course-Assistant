"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Card } from "@/components/ui/card";
import { api, type Course } from "@/lib/api";

export default function TeacherDashboard() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [analytics, setAnalytics] = useState<Record<string, number>>({});

  useEffect(() => {
    api<Course[]>("/courses").then(async (items) => {
      setCourses(items);
      if (items[0]) setAnalytics(await api<Record<string, number>>(`/courses/${items[0].id}/analytics`));
    });
  }, []);

  return (
    <AppShell role="teacher">
      <div className="grid gap-5">
        <div>
          <h2 className="text-2xl font-semibold">Teacher Dashboard</h2>
          <p className="text-muted">Create courses, upload materials, and monitor learning activity.</p>
        </div>
        <div className="grid gap-4 md:grid-cols-4">
          {[
            ["Courses", courses.length],
            ["Students", analytics.enrolled_students ?? 0],
            ["Materials", analytics.materials ?? 0],
            ["Average score", `${analytics.average_score ?? 0}%`],
          ].map(([label, value]) => (
            <Card key={label}>
              <p className="text-sm text-muted">{label}</p>
              <p className="mt-2 text-3xl font-semibold">{value}</p>
            </Card>
          ))}
        </div>
        <Card>
          <h3 className="font-semibold">Recent courses</h3>
          <div className="mt-4 grid gap-3">
            {courses.map((course) => (
              <div key={course.id} className="rounded-md border border-border p-4">
                <p className="font-medium">{course.title}</p>
                <p className="text-sm text-muted">{course.description}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

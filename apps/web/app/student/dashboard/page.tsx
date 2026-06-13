"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Card } from "@/components/ui/card";
import { api, type Course } from "@/lib/api";

export default function StudentDashboard() {
  const [courses, setCourses] = useState<Course[]>([]);
  useEffect(() => {
    api<Course[]>("/courses").then(setCourses);
  }, []);
  return (
    <AppShell role="student">
      <div className="grid gap-5">
        <h2 className="text-2xl font-semibold">Student Dashboard</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {["Enrolled courses", "Quiz attempts", "Learning streak"].map((label, index) => (
            <Card key={label}>
              <p className="text-sm text-muted">{label}</p>
              <p className="mt-2 text-3xl font-semibold">{index === 0 ? courses.length : 0}</p>
            </Card>
          ))}
        </div>
        <Card>
          <h3 className="font-semibold">My courses</h3>
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

"use client";

import { useEffect, useState } from "react";
import { api, type Course } from "@/lib/api";

export function CourseSelect({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const [courses, setCourses] = useState<Course[]>([]);
  useEffect(() => {
    api<Course[]>("/courses").then((items) => {
      setCourses(items);
      if (!value && items[0]) onChange(items[0].id);
    });
  }, [onChange, value]);
  return (
    <select
      className="h-10 rounded-md border border-border bg-card px-3 text-sm"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {courses.map((course) => (
        <option key={course.id} value={course.id}>
          {course.title}
        </option>
      ))}
    </select>
  );
}

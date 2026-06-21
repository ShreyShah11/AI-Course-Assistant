"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  BookOpen,
  Brain,
  Sparkles,
  ArrowRight,
} from "lucide-react";

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
      <div className="space-y-6">
        {/* Hero Section */}
        <Card className="overflow-hidden">
          <div className="flex flex-col gap-4 p-6 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm uppercase tracking-wider text-muted">
                Student Workspace
              </p>

              <h1 className="mt-2 text-3xl font-bold">
                Welcome back 👋
              </h1>

              <p className="mt-2 max-w-2xl text-muted">
                Learn from course materials, ask questions,
                generate summaries, and explore concepts
                through AI-powered learning.
              </p>
            </div>

            <div className="flex items-center gap-3 rounded-xl border border-border px-5 py-4">
              <Brain className="h-5 w-5" />

              <div>
                <p className="text-xs text-muted">
                  AI Learning Assistant
                </p>

                <p className="font-semibold">
                  Ready to Learn
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Quick Overview */}
        <div className="grid gap-4 md:grid-cols-2">
          <Card className="p-5">
            <BookOpen className="h-5 w-5 text-muted" />

            <p className="mt-3 text-sm text-muted">
              Enrolled Courses
            </p>

            <p className="mt-1 text-3xl font-bold">
              {courses.length}
            </p>
          </Card>

          <Card className="p-5">
            <Sparkles className="h-5 w-5 text-muted" />

            <p className="mt-3 text-sm text-muted">
              AI Learning Status
            </p>

            <p className="mt-1 text-3xl font-bold">
              Active
            </p>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Continue Learning */}
          <div className="lg:col-span-2">
            <Card className="p-6">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">
                    Continue Learning
                  </h2>

                  <p className="text-sm text-muted">
                    Open a course to access materials,
                    videos, notes, summaries, and AI chat.
                  </p>
                </div>

                <span className="text-sm text-muted">
                  {courses.length} enrolled
                </span>
              </div>

              {courses.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-10 text-center">
                  <BookOpen className="mx-auto mb-3 h-8 w-8 text-muted" />

                  <h3 className="font-medium">
                    No courses joined yet
                  </h3>

                  <p className="mt-2 text-sm text-muted">
                    Join a course to start learning with
                    CourseGPT.
                  </p>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {courses.map((course) => (
                    <Link
                      key={course.id}
                      href={`/student/courses/${course.id}`}
                      className="group rounded-xl border border-border p-5 transition-all hover:border-primary/50 hover:shadow-lg"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <BookOpen className="h-4 w-4" />

                          <h3 className="font-semibold">
                            {course.title}
                          </h3>
                        </div>

                        <ArrowRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" />
                      </div>

                      <p className="mt-3 line-clamp-3 text-sm text-muted">
                        {course.description}
                      </p>

                      <div className="mt-4 text-sm text-primary">
                        Open Course →
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </Card>
          </div>

          {/* AI Learning Tools */}
          <div>
            <Card className="p-6">
              <div className="mb-4 flex items-center gap-2">
                <Brain className="h-5 w-5" />

                <h2 className="text-lg font-semibold">
                  AI Learning Tools
                </h2>
              </div>

              <div className="space-y-3">
                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Ask questions directly from uploaded
                    PDFs, notes, slides, and videos.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Generate summaries for faster revision.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Create quizzes automatically from
                    course content.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Learn concepts conversationally using
                    CourseGPT AI.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Card } from "@/components/ui/card";
import { api, type Course } from "@/lib/api";
import {
  BookOpen,
  Users,
  FileText,
  Brain,
  Sparkles,
} from "lucide-react";

export default function TeacherDashboard() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [analytics, setAnalytics] = useState<Record<string, number>>({});

  useEffect(() => {
    api<Course[]>("/courses").then(async (items) => {
      setCourses(items);

      if (items[0]) {
        setAnalytics(
          await api<Record<string, number>>(
            `/courses/${items[0].id}/analytics`
          )
        );
      }
    });
  }, []);

  return (
    <AppShell role="teacher">
      <div className="space-y-6">
        {/* Hero Section */}
        <Card className="overflow-hidden">
          <div className="flex flex-col gap-4 p-6 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm uppercase tracking-wider text-muted">
                Teacher Workspace
              </p>

              <h1 className="mt-2 text-3xl font-bold">
                Welcome back 👋
              </h1>

              <p className="mt-2 max-w-2xl text-muted">
                Upload learning materials, organize courses,
                and let AI help students understand concepts
                faster through intelligent learning support.
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-xl border border-border px-4 py-3">
              <Brain className="h-5 w-5" />
              <div>
                <p className="text-xs text-muted">
                  AI Learning Platform
                </p>
                <p className="font-semibold">
                  CourseGPT Active
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <BookOpen className="h-5 w-5 text-muted" />
            </div>

            <p className="mt-3 text-sm text-muted">
              Active Courses
            </p>

            <p className="mt-1 text-3xl font-bold">
              {courses.length}
            </p>
          </Card>

          <Card className="p-5">
            <Users className="h-5 w-5 text-muted" />

            <p className="mt-3 text-sm text-muted">
              Learners Reached
            </p>

            <p className="mt-1 text-3xl font-bold">
              {analytics.enrolled_students ?? 0}
            </p>
          </Card>

          <Card className="p-5">
            <FileText className="h-5 w-5 text-muted" />

            <p className="mt-3 text-sm text-muted">
              Learning Materials
            </p>

            <p className="mt-1 text-3xl font-bold">
              {analytics.materials ?? 0}
            </p>
          </Card>

          <Card className="p-5">
            <Sparkles className="h-5 w-5 text-muted" />

            <p className="mt-3 text-sm text-muted">
              AI Readiness
            </p>

            <p className="mt-1 text-3xl font-bold">
              {courses.length > 0 ? "Ready" : "Setup"}
            </p>
          </Card>
        </div>

        {/* Main Layout */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Courses */}
          <div className="lg:col-span-2">
            <Card className="p-6">
              <div className="mb-5 flex items-center justify-between">
                <h2 className="text-xl font-semibold">
                  Your Courses
                </h2>

                <span className="text-sm text-muted">
                  {courses.length} courses
                </span>
              </div>

              {courses.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border p-10 text-center">
                  <BookOpen className="mx-auto mb-3 h-8 w-8 text-muted" />

                  <h3 className="font-medium">
                    No courses yet
                  </h3>

                  <p className="mt-2 text-sm text-muted">
                    Create your first course and start
                    uploading learning materials.
                  </p>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {courses.map((course) => (
                    <div
                      key={course.id}
                      className="rounded-xl border border-border p-5 transition-all hover:border-primary/50 hover:shadow-lg"
                    >
                      <div className="mb-3 flex items-center gap-2">
                        <BookOpen className="h-4 w-4" />

                        <p className="font-semibold">
                          {course.title}
                        </p>
                      </div>

                      <p className="line-clamp-3 text-sm text-muted">
                        {course.description}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>

          {/* AI Insights */}
          <div>
            <Card className="p-6">
              <div className="mb-4 flex items-center gap-2">
                <Brain className="h-5 w-5" />
                <h2 className="text-lg font-semibold">
                  AI Insights
                </h2>
              </div>

              <div className="space-y-3">
                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Upload PDFs, videos, notes and
                    presentations to make them searchable
                    through AI.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    Students can ask questions directly
                    from uploaded course materials.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    AI-generated summaries and learning
                    assistance become available after
                    materials are processed.
                  </p>
                </div>

                <div className="rounded-lg border border-border p-3">
                  <p className="text-sm">
                    More learning materials generally
                    improve the quality of AI responses.
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
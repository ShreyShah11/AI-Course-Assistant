"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  Trash2,
  BookOpen,
  Sparkles,
  ChevronRight,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { api, type Course } from "@/lib/api";

export default function TeacherCourses() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [form, setForm] = useState({
    title: "",
    description: "",
  });

  async function load() {
    setCourses(await api<Course[]>("/courses"));
  }

  useEffect(() => {
    load();
  }, []);

  async function createCourse(event: React.FormEvent) {
    event.preventDefault();

    await api<Course>("/courses", {
      method: "POST",
      body: JSON.stringify(form),
    });

    setForm({
      title: "",
      description: "",
    });

    load();
  }

  return (
    <AppShell role="teacher">
      <div className="space-y-6">
        {/* Hero Section */}
        <Card className="overflow-hidden">
          <div className="p-8">
            <div className="flex items-center gap-2 text-primary">
              <Sparkles className="h-5 w-5" />
              <span className="text-sm font-medium">
                Course Management
              </span>
            </div>

            <h1 className="mt-3 text-3xl font-bold">
              Build Learning Spaces
            </h1>

            <p className="mt-3 max-w-2xl text-muted">
              Create courses, organize learning materials,
              and prepare content for AI-powered learning.
            </p>

            <div className="mt-6 flex gap-6">
              <div>
                <p className="text-3xl font-bold">
                  {courses.length}
                </p>
                <p className="text-sm text-muted">
                  Active Courses
                </p>
              </div>

              
            </div>
          </div>
        </Card>

        <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
          {/* Create Course */}
          <Card className="h-fit p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg border border-border p-2">
                <Plus className="h-5 w-5" />
              </div>

              <div>
                <h2 className="font-semibold">
                  Create New Course
                </h2>

                <p className="text-sm text-muted">
                  Start a new learning space
                </p>
              </div>
            </div>

            <form
              onSubmit={createCourse}
              className="mt-6 space-y-4"
            >
              <Input
                placeholder="Course title"
                value={form.title}
                onChange={(event) =>
                  setForm({
                    ...form,
                    title: event.target.value,
                  })
                }
              />

              <Textarea
                placeholder="Describe what students will learn..."
                value={form.description}
                onChange={(event) =>
                  setForm({
                    ...form,
                    description: event.target.value,
                  })
                }
              />

              <Button className="w-full">
                <Plus className="h-4 w-4" />
                Create Course
              </Button>
            </form>
          </Card>

          {/* Courses */}
          <div>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                Your Courses
              </h2>

              <span className="text-sm text-muted">
                {courses.length} total
              </span>
            </div>

            {courses.length === 0 ? (
              <Card className="p-10 text-center">
                <BookOpen className="mx-auto h-10 w-10 text-muted" />

                <h3 className="mt-4 text-lg font-semibold">
                  No Courses Yet
                </h3>

                <p className="mt-2 text-muted">
                  Create your first course to start
                  building AI-powered learning experiences.
                </p>
              </Card>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {courses.map((course) => (
                  <Card
                    key={course.id}
                    className="group p-5 transition-all duration-200 hover:border-primary/40 hover:shadow-lg"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <BookOpen className="h-5 w-5 text-primary" />

                        <h3 className="font-semibold">
                          {course.title}
                        </h3>
                      </div>

                      <Button
                        variant="ghost"
                        onClick={async () => {
                          await api(
                            `/courses/${course.id}`,
                            {
                              method: "DELETE",
                            }
                          );

                          load();
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>

                    <p className="mt-3 line-clamp-3 text-sm text-muted">
                      {course.description ||
                        "No description provided."}
                    </p>

                    <div className="mt-5 flex items-center justify-between border-t border-border pt-4">
                      <span className="text-xs text-muted">
                        ID: {course.id.slice(0, 8)}
                      </span>

                      <div className="flex items-center gap-1 text-sm text-primary">
                        <span>Manage</span>
                        <ChevronRight className="h-4 w-4" />
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
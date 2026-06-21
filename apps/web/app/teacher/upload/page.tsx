"use client";

import { useCallback, useEffect, useState } from "react";
import {
  FileUp,
  Link as LinkIcon,
  Trash2,
  FileText,
  Video,
  Brain,
  UploadCloud,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  api,
  API_URL,
  getToken,
  type Material,
} from "@/lib/api";

export default function UploadMaterials() {
  const [courseId, setCourseId] = useState("");
  const [materials, setMaterials] = useState<Material[]>([]);
  const [youtube, setYoutube] = useState("");
  const [progress, setProgress] = useState("");

  const load = useCallback(
    async (id = courseId) => {
      if (id) {
        setMaterials(
          await api<Material[]>(
            `/courses/${id}/materials`
          )
        );
      }
    },
    [courseId]
  );

  useEffect(() => {
    load(courseId);
  }, [courseId, load]);

  async function upload(file: File) {
    try {
      if (!courseId) {
        alert("Please select a course");
        return;
      }

      const form = new FormData();
      form.set("file", file);

      setProgress(`Uploading ${file.name}...`);

      const response = await fetch(
        `${API_URL}/courses/${courseId}/materials/files`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${getToken()}`,
          },
          body: form,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Upload failed (${response.status})`
        );
      }

      setProgress(
        "Material uploaded and queued for AI processing"
      );

      await load();
    } catch (err) {
      console.error(err);

      setProgress(
        err instanceof Error
          ? err.message
          : "Upload failed"
      );
    }
  }

  const videoMaterials = materials.filter(
    (m) =>
      m.file_type?.toLowerCase().includes("video") ||
      m.file_name?.toLowerCase().includes("youtube")
  );

  const documentMaterials = materials.filter(
    (m) =>
      !m.file_type?.toLowerCase().includes("video") &&
      !m.file_name?.toLowerCase().includes("youtube")
  );

  return (
    <AppShell role="teacher">
      <div className="space-y-6">
        {/* Header */}
        <Card className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold">
                Course Content
              </h1>

              <p className="mt-2 text-muted">
                Upload lectures, notes, videos and
                learning resources for AI-powered
                student assistance.
              </p>
            </div>

            <CourseSelect
              value={courseId}
              onChange={setCourseId}
            />
          </div>
        </Card>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="p-5">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-primary" />

              <div>
                <p className="text-sm text-muted">
                  Documents
                </p>

                <p className="text-2xl font-bold">
                  {documentMaterials.length}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-5">
            <div className="flex items-center gap-3">
              <Video className="h-5 w-5 text-primary" />

              <div>
                <p className="text-sm text-muted">
                  Videos
                </p>

                <p className="text-2xl font-bold">
                  {videoMaterials.length}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-5">
            <div className="flex items-center gap-3">
              <Brain className="h-5 w-5 text-primary" />

              <div>
                <p className="text-sm text-muted">
                  AI Knowledge Base
                </p>

                <p className="text-2xl font-bold">
                  {materials.length > 0
                    ? "Ready"
                    : "Empty"}
                </p>
              </div>
            </div>
          </Card>
        </div>

        {/* Upload Area */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card className="p-6">
            <div className="flex items-center gap-2">
              <UploadCloud className="h-5 w-5" />
              <h3 className="font-semibold">
                Upload Lecture Material
              </h3>
            </div>

            <label className="mt-5 flex min-h-52 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-border transition hover:border-primary/50">
              <FileUp className="mb-4 h-10 w-10 text-primary" />

              <p className="font-medium">
                Drag & drop or click to upload
              </p>

              <p className="mt-2 text-sm text-muted">
                PDF, DOCX, PPT, TXT, MP4 and more
              </p>

              <input
                type="file"
                className="hidden"
                onChange={(event) =>
                  event.target.files?.[0] &&
                  upload(event.target.files[0])
                }
              />
            </label>

            {progress && (
              <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm">
                {progress}
              </div>
            )}
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-2">
              <Video className="h-5 w-5" />

              <h3 className="font-semibold">
                Add YouTube Lecture
              </h3>
            </div>

            <p className="mt-2 text-sm text-muted">
              Import lecture videos directly into
              the AI learning pipeline.
            </p>

            <form
              className="mt-6 flex gap-2"
              onSubmit={async (event) => {
                event.preventDefault();

                if (!youtube.trim()) return;

                await api(
                  `/courses/${courseId}/materials/youtube`,
                  {
                    method: "POST",
                    body: JSON.stringify({
                      url: youtube,
                    }),
                  }
                );

                setYoutube("");

                await load();
              }}
            >
              <Input
                placeholder="https://youtube.com/watch?v=..."
                value={youtube}
                onChange={(event) =>
                  setYoutube(event.target.value)
                }
              />

              <Button type="submit">
                <LinkIcon className="h-4 w-4" />
              </Button>
            </form>
          </Card>
        </div>

        {/* Materials */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Documents */}
          <Card className="p-6">
            <h3 className="mb-5 text-lg font-semibold">
              Lecture Notes & Documents
            </h3>

            {documentMaterials.length === 0 ? (
              <p className="text-sm text-muted">
                No documents uploaded yet.
              </p>
            ) : (
              <div className="space-y-3">
                {documentMaterials.map((material) => (
                  <div
                    key={material.id}
                    className="flex items-center justify-between rounded-xl border border-border p-4"
                  >
                    <div>
                      <p className="font-medium">
                        {material.file_name}
                      </p>

                      <p className="text-xs text-muted">
                        {material.file_type}
                      </p>
                    </div>

                    <Button
                      variant="ghost"
                      onClick={async () => {
                        await api(
                          `/courses/${courseId}/materials/${material.id}`,
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
                ))}
              </div>
            )}
          </Card>

          {/* Videos */}
          <Card className="p-6">
            <h3 className="mb-5 text-lg font-semibold">
              Lecture Videos
            </h3>

            {videoMaterials.length === 0 ? (
              <p className="text-sm text-muted">
                No videos uploaded yet.
              </p>
            ) : (
              <div className="space-y-3">
                {videoMaterials.map((material) => (
                  <div
                    key={material.id}
                    className="flex items-center justify-between rounded-xl border border-border p-4"
                  >
                    <div>
                      <p className="font-medium">
                        {material.file_name}
                      </p>

                      <p className="text-xs text-muted">
                        {material.file_type}
                      </p>
                    </div>

                    <Button
                      variant="ghost"
                      onClick={async () => {
                        await api(
                          `/courses/${courseId}/materials/${material.id}`,
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
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
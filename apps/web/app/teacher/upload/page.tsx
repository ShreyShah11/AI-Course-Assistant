"use client";

import { useCallback, useEffect, useState } from "react";
import { FileUp, Link as LinkIcon, Trash2 } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, API_URL, getToken, type Material } from "@/lib/api";

export default function UploadMaterials() {
  const [courseId, setCourseId] = useState("");
  const [materials, setMaterials] = useState<Material[]>([]);
  const [youtube, setYoutube] = useState("");
  const [progress, setProgress] = useState("");

  const load = useCallback(async (id = courseId) => {
    if (id) setMaterials(await api<Material[]>(`/courses/${id}/materials`));
  }, [courseId]);

  useEffect(() => {
    load(courseId);
  }, [courseId, load]);

async function upload(file: File) {
  try {
    console.log("========== UPLOAD DEBUG ==========");
    console.log("Course ID:", courseId);

    if (!courseId) {
      alert("No course selected!");
      return;
    }

    const form = new FormData();
    form.set("file", file);

    setProgress(`Uploading ${file.name}`);

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

    console.log("Status:", response.status);

    const text = await response.text();
    console.log("Response:", text);

    if (!response.ok) {
      throw new Error(`Upload failed (${response.status})`);
    }

    setProgress("Ingestion job queued");
    await load();
  } catch (err) {
    console.error(err);
    setProgress(err instanceof Error ? err.message : "Upload failed");
  }
}

  return (
    <AppShell role="teacher">
      <div className="grid gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold">Upload Materials</h2>
            <p className="text-muted">Files and YouTube links are sent to the existing LangChain ingestion pipeline.</p>
          </div>
          <CourseSelect value={courseId} onChange={setCourseId} />
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <h3 className="font-semibold">File upload</h3>
            <label className="mt-4 grid min-h-36 cursor-pointer place-items-center rounded-md border border-dashed border-border p-6 text-center">
              <FileUp className="mb-2 size-8 text-primary" />
              <span className="text-sm text-muted">PDF, DOCX, PPT, TXT, or video</span>
              <input className="hidden" type="file" onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])} />
            </label>
            {progress && <p className="mt-3 text-sm text-primary">{progress}</p>}
          </Card>
          <Card>
            <h3 className="font-semibold">YouTube link</h3>
            <form
              className="mt-4 flex gap-2"
              onSubmit={async (event) => {
                event.preventDefault();
                await api(`/courses/${courseId}/materials/youtube`, { method: "POST", body: JSON.stringify({ url: youtube }) });
                setYoutube("");
                load();
              }}
            >
              <Input placeholder="https://youtube.com/watch?v=..." value={youtube} onChange={(event) => setYoutube(event.target.value)} />
              <Button><LinkIcon className="size-4" /></Button>
            </form>
          </Card>
        </div>
        <Card>
          <h3 className="font-semibold">Uploaded materials</h3>
          <div className="mt-4 grid gap-3">
            {materials.map((material) => (
              <div key={material.id} className="flex items-center justify-between gap-3 rounded-md border border-border p-3">
                <div>
                  <p className="font-medium">{material.file_name}</p>
                  <p className="text-xs text-muted">{material.file_type} · job {material.ingestion_job_id ?? "queued"}</p>
                </div>
                <Button
                  variant="ghost"
                  onClick={async () => {
                    await api(`/courses/${courseId}/materials/${material.id}`, { method: "DELETE" });
                    load();
                  }}
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

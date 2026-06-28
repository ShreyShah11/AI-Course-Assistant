"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api, type Material } from "@/lib/api";

export default function CoursePage() {
  const params = useParams();
  const courseId = params.id as string;

  const [materials, setMaterials] = useState<Material[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const data = await api<Material[]>(
          `/courses/${courseId}/materials`
        );

        console.log("MATERIALS:", data);
        setMaterials(data);
      } catch (err) {
        console.error(err);
      }
    }

    if (courseId) {
      load();
    }
  }, [courseId]);

return (
  <div className="p-10">
    <h1 className="mb-6 text-3xl font-bold">
      Course Materials
    </h1>

    <div className="grid gap-4">
      {materials.map((material) => (
        <div
          key={material.id}
          className="rounded-xl border border-zinc-800 p-5"
        >
          <h3 className="font-semibold">
            {material.file_name}
          </h3>

          <p className="mt-1 text-sm text-gray-400">
            {material.file_type.toUpperCase()}
          </p>

          {material.file_type === "youtube" ? (
            <a
              href={material.storage_path}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-block rounded bg-cyan-500 px-4 py-2 text-black"
            >
              Watch Video
            </a>
          ) : (
            <a
              href={`http://127.0.0.1:8000/courses/${courseId}/materials/${material.id}/download`}
              target="_blank"
              className="mt-3 inline-block rounded bg-cyan-500 px-4 py-2 text-black"
            >
              Download Material
            </a>
          )}
        </div>
      ))}
    </div>
  </div>
);
}
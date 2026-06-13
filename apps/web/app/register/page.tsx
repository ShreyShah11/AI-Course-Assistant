"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, setSession, type Role, type User } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [role, setRole] = useState<Role>("student");
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const result = await api<{ access_token: string; user: User }>("/auth/register", {
        method: "POST",
        body: JSON.stringify({ ...form, role }),
      });
      setSession(result.access_token, result.user);
      router.push(result.user.role === "teacher" ? "/teacher/dashboard" : "/student/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    }
  }

  return (
    <main className="grid min-h-screen place-items-center px-5">
      <Card className="w-full max-w-md">
        <h1 className="text-2xl font-semibold">Create account</h1>
        <div className="mt-5 grid grid-cols-2 gap-2 rounded-md border border-border p-1">
          {(["student", "teacher"] as Role[]).map((item) => (
            <button
              key={item}
              className={`h-9 rounded text-sm capitalize ${role === item ? "bg-primary text-primary-foreground" : "text-muted"}`}
              onClick={() => setRole(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
        <form onSubmit={submit} className="mt-5 grid gap-4">
          <Input placeholder="Name" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
          <Input placeholder="Email" type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
          <Input placeholder="Password" type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
          {error && <p className="text-sm text-accent">{error}</p>}
          <Button>Create account</Button>
        </form>
        <p className="mt-4 text-sm text-muted">
          Already registered? <Link className="font-medium text-primary" href="/login">Login</Link>
        </p>
      </Card>
    </main>
  );
}

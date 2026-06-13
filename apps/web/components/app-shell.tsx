"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BookOpen, GraduationCap, LayoutDashboard, LogOut, Moon, Upload, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { clearSession, getStoredUser } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

const teacherLinks = [
  { href: "/teacher/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/teacher/courses", label: "Courses", icon: BookOpen },
  { href: "/teacher/upload", label: "Materials", icon: Upload },
  { href: "/teacher/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/teacher/exams", label: "Exams", icon: GraduationCap },
];

const studentLinks = [
  { href: "/student/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/student/courses", label: "My Courses", icon: BookOpen },
  { href: "/student/chat", label: "Course Chat", icon: GraduationCap },
  { href: "/student/quizzes", label: "Quizzes", icon: Upload },
  { href: "/student/progress", label: "Progress", icon: BarChart3 },
];

export function AppShell({ children, role }: { children: React.ReactNode; role: "teacher" | "student" }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<any>(null);

useEffect(() => {
  setUser(getStoredUser());
}, []);
  const links = role === "teacher" ? teacherLinks : studentLinks;
  return (
    <div className="min-h-screen md:grid md:grid-cols-[248px_1fr]">
      <aside className="border-b border-border bg-card md:min-h-screen md:border-b-0 md:border-r">
        <div className="flex h-16 items-center justify-between px-5 md:h-20">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <span className="grid size-9 place-items-center rounded-md bg-primary text-primary-foreground">CG</span>
            CourseGPT
          </Link>
          <Moon className="size-5 text-muted md:hidden" />
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
          {links.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex h-10 shrink-0 items-center gap-2 rounded-md px-3 text-sm text-muted transition hover:bg-black/5 dark:hover:bg-white/10",
                pathname === item.href && "bg-primary/10 text-foreground",
              )}
            >
              <item.icon className="size-4" />
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <main>
        <header className="flex min-h-16 items-center justify-between border-b border-border px-5">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted">{role}</p>
            <h1 className="text-lg font-semibold">{user?.name ?? "CourseGPT"}</h1>
          </div>
          <Button
            variant="secondary"
            onClick={() => {
              clearSession();
              router.push("/login");
            }}
          >
            <LogOut className="size-4" />
            Logout
          </Button>
        </header>
        <div className="mx-auto w-full max-w-7xl p-5 md:p-8">{children}</div>
      </main>
    </div>
  );
}

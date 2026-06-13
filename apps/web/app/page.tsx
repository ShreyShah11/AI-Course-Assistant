import Link from "next/link";
import { ArrowRight, BookOpen, FileText, MessageSquare, ShieldCheck } from "lucide-react";

const features = [
  { title: "Upload materials", body: "PDF, DOCX, PPT, TXT, video, and YouTube ingestion.", Icon: FileText },
  { title: "Ask with citations", body: "Students get grounded answers from course vectors only.", Icon: MessageSquare },
  { title: "Generate assessments", body: "Teachers and students can create quizzes from retrieved context.", Icon: BookOpen },
  { title: "Role protected", body: "JWT auth, password hashing, and protected teacher/student routes.", Icon: ShieldCheck },
];

export default function Home() {
  return (
    <main className="min-h-screen">
      <section className="grid min-h-[92vh] content-between border-b border-border bg-[linear-gradient(120deg,rgba(15,118,110,.16),transparent_38%),linear-gradient(310deg,rgba(194,65,12,.12),transparent_32%)] px-5 py-6">
        <nav className="mx-auto flex w-full max-w-7xl items-center justify-between">
          <div className="flex items-center gap-2 font-semibold">
            <span className="grid size-9 place-items-center rounded-md bg-primary text-primary-foreground">CG</span>
            CourseGPT
          </div>
          <div className="flex items-center gap-2">
            <Link className="rounded-md px-3 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10" href="/login">
              Login
            </Link>
            <Link className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground" href="/register">
              Register
            </Link>
          </div>
        </nav>
        <div className="mx-auto grid w-full max-w-7xl gap-10 py-12 lg:grid-cols-[1fr_520px] lg:items-center">
          <div>
            <p className="mb-4 text-sm font-medium uppercase tracking-[0.18em] text-primary">RAG-native learning platform</p>
            <h1 className="max-w-4xl text-5xl font-semibold leading-tight md:text-7xl">CourseGPT</h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-muted">
              A production-ready educational workspace wrapped around your existing LangChain pipeline for course materials,
              quizzes, summaries, citations, and course-specific AI chat.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link className="inline-flex h-11 items-center gap-2 rounded-md bg-primary px-5 text-sm font-medium text-primary-foreground" href="/register">
                Start teaching <ArrowRight className="size-4" />
              </Link>
              <Link className="inline-flex h-11 items-center rounded-md border border-border bg-card px-5 text-sm font-medium" href="/login">
                Student login
              </Link>
            </div>
          </div>
          <div className="grid gap-3 rounded-lg border border-border bg-card p-4 shadow-sm">
            {features.map(({ title, body, Icon }) => (
              <div key={title} className="flex gap-3 rounded-md border border-border p-4">
                <Icon className="mt-1 size-5 text-primary" />
                <div>
                  <h2 className="font-medium">{title}</h2>
                  <p className="text-sm leading-6 text-muted">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="mx-auto grid w-full max-w-7xl gap-3 md:grid-cols-4">
          {["Course chat", "Flashcards", "Progress", "Analytics"].map((item) => (
            <div key={item} className="border-t border-border py-4 text-sm font-medium text-muted">
              {item}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, setSession, type Role, type User } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();

  const [role, setRole] = useState<Role>("student");

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [confirmPassword, setConfirmPassword] =
    useState("");

  const [showPassword, setShowPassword] =
    useState(false);

  const [otp, setOtp] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [otpVerified, setOtpVerified] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  function passwordStrength() {
    if (form.password.length < 6) {
      return {
        text: "Weak",
        color: "text-red-500",
      };
    }

    if (form.password.length < 10) {
      return {
        text: "Medium",
        color: "text-yellow-500",
      };
    }

    return {
      text: "Strong",
      color: "text-green-500",
    };
  }

  async function sendOtp() {
    setError("");

    if (
      !form.name ||
      !form.email ||
      !form.password
    ) {
      setError("Please fill all fields");
      return;
    }

    if (
      form.password !==
      confirmPassword
    ) {
      setError(
        "Passwords do not match"
      );
      return;
    }

    try {
      setLoading(true);

      await api("/auth/send-otp", {
        method: "POST",
        body: JSON.stringify({
          email: form.email,
        }),
      });

      setOtpSent(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to send OTP"
      );
    } finally {
      setLoading(false);
    }
  }

  async function verifyOtp() {
    setError("");

    try {
      setLoading(true);

      await api("/auth/verify-otp", {
        method: "POST",
        body: JSON.stringify({
          email: form.email,
          otp,
        }),
      });

      setOtpVerified(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Invalid OTP"
      );
    } finally {
      setLoading(false);
    }
  }

  async function submit(
    event: React.FormEvent
  ) {
    event.preventDefault();

    if (!otpVerified) {
      setError(
        "Please verify your email first"
      );
      return;
    }

    setError("");

    try {
      const result = await api<{
        access_token: string;
        user: User;
      }>("/auth/register", {
        method: "POST",
        body: JSON.stringify({
          ...form,
          role,
        }),
      });

      setSession(
        result.access_token,
        result.user
      );

      router.push(
        result.user.role ===
          "teacher"
          ? "/teacher/dashboard"
          : "/student/dashboard"
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Registration failed"
      );
    }
  }

  return (
    <main className="grid min-h-screen place-items-center px-5">
      <Card className="w-full max-w-md p-6">
        <h1 className="text-2xl font-semibold">
          Create Account
        </h1>

        <p className="mt-2 text-sm text-muted">
          Join CourseGPT and start
          learning smarter.
        </p>

        <div className="mt-5 grid grid-cols-2 gap-2 rounded-md border border-border p-1">
          {(
            ["student", "teacher"] as Role[]
          ).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() =>
                setRole(item)
              }
              className={`h-10 rounded text-sm capitalize ${
                role === item
                  ? "bg-primary text-primary-foreground"
                  : "text-muted"
              }`}
            >
              {item}
            </button>
          ))}
        </div>

        <form
          onSubmit={submit}
          className="mt-5 grid gap-4"
        >
          <Input
            placeholder="Name"
            value={form.name}
            onChange={(e) =>
              setForm({
                ...form,
                name: e.target.value,
              })
            }
          />

          <Input
            placeholder="Email"
            type="email"
            disabled={otpVerified}
            value={form.email}
            onChange={(e) =>
              setForm({
                ...form,
                email: e.target.value,
              })
            }
          />

          <div className="relative">
            <Input
              placeholder="Password"
              type={
                showPassword
                  ? "text"
                  : "password"
              }
              value={form.password}
              onChange={(e) =>
                setForm({
                  ...form,
                  password:
                    e.target.value,
                })
              }
            />

            <button
              type="button"
              className="absolute right-3 top-3 text-muted"
              onClick={() =>
                setShowPassword(
                  !showPassword
                )
              }
            >
              {showPassword ? (
                <EyeOff size={18} />
              ) : (
                <Eye size={18} />
              )}
            </button>
          </div>

          {form.password && (
            <p
              className={`text-xs ${
                passwordStrength().color
              }`}
            >
              Password Strength:{" "}
              {passwordStrength().text}
            </p>
          )}

          <Input
            placeholder="Confirm Password"
            type="password"
            value={
              confirmPassword
            }
            onChange={(e) =>
              setConfirmPassword(
                e.target.value
              )
            }
          />

          {!otpSent ? (
            <Button
              type="button"
              onClick={sendOtp}
              disabled={loading}
            >
              {loading
                ? "Sending OTP..."
                : "Send OTP"}
            </Button>
          ) : (
            <>
              <Input
                placeholder="Enter OTP"
                value={otp}
                onChange={(e) =>
                  setOtp(
                    e.target.value
                  )
                }
              />

              <div className="flex gap-2">
                <Button
                  type="button"
                  onClick={
                    verifyOtp
                  }
                  disabled={loading}
                  className="flex-1"
                >
                  {loading
                    ? "Verifying..."
                    : "Verify OTP"}
                </Button>

                <Button
                  type="button"
                  variant="secondary"
                  onClick={sendOtp}
                >
                  Resend
                </Button>
              </div>
            </>
          )}

          {otpVerified && (
            <div className="rounded-md border border-green-500/30 bg-green-500/10 p-2 text-sm text-green-500">
              ✓ Email verified successfully
            </div>
          )}

          {error && (
            <p className="text-sm text-red-500">
              {error}
            </p>
          )}

          <Button
            type="submit"
            disabled={!otpVerified}
          >
            Create Account
          </Button>
        </form>

        <p className="mt-4 text-sm text-muted">
          Already registered?{" "}
          <Link
            className="font-medium text-primary"
            href="/login"
          >
            Login
          </Link>
        </p>
      </Card>
    </main>
  );
}
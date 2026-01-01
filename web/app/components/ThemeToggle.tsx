"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = (localStorage.getItem("theme") as any) || "dark";
    setTheme(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  return (
    <button
      className="px-3 py-2 rounded-xl border border-white/15 bg-white/5 hover:bg-white/10"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      Theme: {theme === "dark" ? "Black" : "Light"}
    </button>
  );
}

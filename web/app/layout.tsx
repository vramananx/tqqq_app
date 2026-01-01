import "./globals.css";

export const metadata = { title: "Strategy Lab" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-white">
        {children}
      </body>
    </html>
  );
}

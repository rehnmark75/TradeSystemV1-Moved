"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

interface ScannerCategoryPageProps {
  params: { category: string };
}

export default function ScannerCategoryPage({ params }: ScannerCategoryPageProps) {
  const router = useRouter();

  useEffect(() => {
    router.replace("/settings/scanner");
  }, [router]);

  return (
    <div className="settings-placeholder">Redirecting to Scanner Settings…</div>
  );
}

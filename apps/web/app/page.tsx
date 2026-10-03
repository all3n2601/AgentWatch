import { Dashboard } from "@/components/dashboard";
import { loadExperiments } from "@/lib/experiments";
export const dynamic = "force-dynamic";
export default async function Home() {
  return <Dashboard initialData={await loadExperiments()} page="overview" />;
}

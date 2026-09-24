import { Button } from "@model-passport/ui";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-8 py-24">
      <p className="mb-4 text-sm font-medium uppercase tracking-[0.2em] text-emerald-400">
        Model Passport
      </p>
      <h1 className="max-w-3xl text-5xl font-semibold tracking-tight sm:text-7xl">
        Evidence-backed model assurance.
      </h1>
      <p className="mt-6 max-w-2xl text-lg leading-8 text-zinc-400">
        Register models, run isolated checks, review signed evidence, and enforce deployment policy
        from one control plane.
      </p>
      <div className="mt-10">
        <Button>Register a model</Button>
      </div>
    </main>
  );
}


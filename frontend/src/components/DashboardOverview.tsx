import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  FileSearch,
  ShieldCheck,
} from 'lucide-react'

import GlassPanel from './GlassPanel'

interface ActionCardProps {
  to: string
  icon: ReactNode
  title: string
  description: string
}

function ActionCard({
  to,
  icon,
  title,
  description,
}: ActionCardProps) {
  return (
    <NavLink
      to={to}
      className="group block rounded-[1.75rem] border border-white/70 bg-white/30 p-6 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-white/50 hover:shadow-[0_18px_45px_rgba(30,41,59,0.08)]"
    >
      <div className="flex items-center gap-5">
        <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-[1.4rem] border border-white/80 bg-white/55 text-slate-700 shadow-sm backdrop-blur-xl transition-transform duration-300 group-hover:scale-105">
          {icon}
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-semibold tracking-tight text-slate-800">
            {title}
          </h3>

          <p className="mt-1 text-sm text-slate-500">
            {description}
          </p>
        </div>

        <ArrowRight className="h-5 w-5 shrink-0 text-slate-400 transition-transform duration-300 group-hover:translate-x-1 group-hover:text-slate-700" />
      </div>
    </NavLink>
  )
}

function DashboardOverview() {
  return (
    <section className="mx-auto max-w-7xl px-6 pb-20 lg:px-8">
      <GlassPanel className="overflow-hidden">
        {/* Overview header */}
        <div className="border-b border-white/60 px-8 py-8 sm:px-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                <ShieldCheck className="h-4 w-4" />
                System Overview
              </div>

              <h2 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
                Inspect. Understand. Respond.
              </h2>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500 sm:text-base">
                Analyze instructions using the connected threat detection
                engine and review the signals behind every decision.
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-full border border-white/70 bg-white/45 px-4 py-2.5 text-xs font-medium text-slate-500 backdrop-blur-xl">
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-slate-400 opacity-30" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-slate-400" />
              </span>
              Detection engine connected
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid border-b border-white/60 sm:grid-cols-3">
          <Metric
            icon={<Activity className="h-5 w-5" />}
            label="Analyses"
            value="Data unavailable"
          />

          <Metric
            icon={<ShieldCheck className="h-5 w-5" />}
            label="Threats detected"
            value="Data unavailable"
          />

          <Metric
            icon={<BrainCircuit className="h-5 w-5" />}
            label="Detection model"
            value="Unavailable"
          />
        </div>

        {/* Quick actions */}
        <div className="px-8 py-10 sm:px-10">
          <div className="mb-7">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
              Quick Actions
            </p>

            <h3 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900">
              Continue working.
            </h3>
          </div>

          <div className="space-y-4">
            <ActionCard
              to="/analyze"
              icon={<Activity className="h-8 w-8" />}
              title="Analyze Prompt"
              description="Inspect a suspicious instruction"
            />

            <ActionCard
              to="/documents"
              icon={<FileSearch className="h-8 w-8" />}
              title="Scan Document"
              description="Analyze a supported document"
            />

            <ActionCard
              to="/explainability"
              icon={<BrainCircuit className="h-8 w-8" />}
              title="Explainability"
              description="Understand a detection decision"
            />
          </div>
        </div>
      </GlassPanel>
    </section>
  )
}

function Metric({
  icon,
  label,
  value,
}: {
  icon: ReactNode
  label: string
  value: string
}) {
  return (
    <div className="border-b border-white/50 px-8 py-7 last:border-b-0 sm:border-b-0 sm:border-r sm:last:border-r-0">
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl border border-white/70 bg-white/50 text-slate-600 shadow-sm">
        {icon}
      </div>

      <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-sm font-medium text-slate-600">
        {value}
      </p>
    </div>
  )
}

export default DashboardOverview
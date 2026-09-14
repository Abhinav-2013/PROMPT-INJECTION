import { motion } from 'framer-motion'
import {
  ArrowRight,
  FileSearch,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import DashboardOverview from '../components/DashboardOverview'

function Dashboard() {
  return (
    <>
      <section className="relative z-10 mx-auto flex min-h-[calc(100vh-90px)] max-w-7xl items-center px-6 pb-20 pt-12 lg:px-8">
        <div className="grid w-full items-center gap-16 lg:grid-cols-[1.1fr_0.9fr]">
          <motion.div
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/50 px-4 py-2 text-xs font-medium text-slate-600 shadow-sm backdrop-blur-xl">
              <Sparkles className="h-3.5 w-3.5" />
              Intelligent threat detection
            </div>

            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.05] tracking-[-0.04em] sm:text-6xl lg:text-7xl">
              Detect hostile instructions
              <span className="block text-slate-400">
                before they execute.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-base leading-7 text-slate-500 sm:text-lg">
              Analyze prompts and documents using machine learning,
              behavioral rules, semantic analysis, threat fusion,
              and explainable AI.
            </p>

            <div className="mt-9 flex flex-wrap gap-3">
              <Link
                to="/analyze"
                className="group flex items-center gap-2 rounded-full bg-[#111318] px-6 py-3.5 text-sm font-medium text-white shadow-lg shadow-slate-900/10 transition hover:-translate-y-0.5"
              >
                Analyze Prompt
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>

              <Link
                to="/documents"
                className="flex items-center gap-2 rounded-full border border-white/80 bg-white/45 px-6 py-3.5 text-sm font-medium text-slate-700 shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5"
              >
                <FileSearch className="h-4 w-4" />
                Scan Document
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9, delay: 0.15 }}
            className="relative mx-auto h-[420px] w-full max-w-[480px]"
          >
            <motion.div
              className="absolute left-1/2 top-1/2 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/80 bg-white/30 shadow-[0_30px_100px_rgba(80,100,140,0.18)] backdrop-blur-2xl"
              animate={{
                y: [-10, 10, -10],
                rotate: [0, 3, 0],
              }}
              transition={{
                duration: 7,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            >
              <div className="absolute inset-6 rounded-full border border-white/70 bg-white/20" />

              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex h-24 w-24 items-center justify-center rounded-[2rem] border border-white/80 bg-white/50 shadow-xl backdrop-blur-xl">
                  <ShieldCheck className="h-11 w-11 text-slate-800" />
                </div>
              </div>
            </motion.div>

            <FloatingCard
              label="Threat Analysis"
              value="Ready"
              className="left-0 top-16"
            />

            <FloatingCard
              label="Detection Engine"
              value="Active"
              className="right-0 top-44"
            />

            <FloatingCard
              label="Semantic Engine"
              value="Connected"
              className="bottom-12 left-12"
            />
          </motion.div>
        </div>
      </section>

      <DashboardOverview />
    </>
  )
}

function FloatingCard({
  label,
  value,
  className,
}: {
  label: string
  value: string
  className: string
}) {
  return (
    <motion.div
      className={`absolute ${className} w-48 rounded-2xl border border-white/80 bg-white/40 p-4 shadow-lg shadow-slate-900/5 backdrop-blur-2xl`}
      animate={{ y: [-5, 5, -5] }}
      transition={{
        duration: 5,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    >
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-medium text-slate-700">{value}</p>
    </motion.div>
  )
}

export default Dashboard
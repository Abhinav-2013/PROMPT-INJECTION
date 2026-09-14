import { useEffect, useState, type ReactNode } from 'react'
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  RefreshCw,
  Server,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

import GlassPanel from '../components/GlassPanel'
import {
  getAPIInfo,
  getHealth,
} from '../api/athsApi'

type StatusValue = 'Connected' | 'Unavailable' | 'Checking'

function SystemStatus() {
  const [apiStatus, setApiStatus] =
    useState<StatusValue>('Checking')

  const [healthMessage, setHealthMessage] =
    useState('Checking backend health...')

  const [lastChecked, setLastChecked] =
    useState<string | null>(null)

  const checkStatus = async () => {
    setApiStatus('Checking')
    setHealthMessage('Checking backend health...')

    try {
      const [info, health] = await Promise.all([
        getAPIInfo(),
        getHealth(),
      ])

      if (health.status === 'healthy') {
        setApiStatus('Connected')
        setHealthMessage(
          info.service ||
            'FastAPI detection service is running.',
        )
      } else {
        setApiStatus('Unavailable')
        setHealthMessage('Backend health check did not report healthy.')
      }
    } catch {
      setApiStatus('Unavailable')
      setHealthMessage(
        'The FastAPI backend could not be reached.',
      )
    } finally {
      setLastChecked(
        new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      )
    }
  }

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void checkStatus()
    }, 0)

    return () => window.clearTimeout(timeoutId)
  }, [])

  const connected = apiStatus === 'Connected'

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      <div className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Infrastructure
        </p>

        <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
          System status.
        </h1>

        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
          View the availability of the connected detection service.
        </p>
      </div>

      <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          {connected ? (
            <CheckCircle2 className="h-5 w-5 text-slate-700" />
          ) : (
            <CircleAlert className="h-5 w-5 text-slate-400" />
          )}

          <div>
            <p className="text-sm font-semibold text-slate-800">
              {connected
                ? 'Backend connected'
                : apiStatus === 'Checking'
                  ? 'Checking backend'
                  : 'Backend unavailable'}
            </p>

            <p className="text-xs text-slate-400">
              {lastChecked
                ? `Last checked at ${lastChecked}`
                : 'Checking now'}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => void checkStatus()}
          disabled={apiStatus === 'Checking'}
          className="inline-flex items-center justify-center gap-2 rounded-full border border-white/80 bg-white/45 px-4 py-2.5 text-xs font-medium text-slate-600 shadow-sm backdrop-blur-xl transition hover:bg-white/70 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw
            className={`h-4 w-4 ${
              apiStatus === 'Checking' ? 'animate-spin' : ''
            }`}
          />
          Refresh
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <StatusCard
          icon={<Server className="h-5 w-5" />}
          title="API"
          value={apiStatus}
        />

        <StatusCard
          icon={<Activity className="h-5 w-5" />}
          title="Detection Engine"
          value={apiStatus}
        />

        <StatusCard
          icon={<BrainCircuit className="h-5 w-5" />}
          title="Model"
          value={apiStatus}
        />

        <StatusCard
          icon={<Sparkles className="h-5 w-5" />}
          title="Semantic Engine"
          value={apiStatus}
        />
      </div>

      <GlassPanel className="mt-5 p-7">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-slate-400" />

          <div>
            <h2 className="font-semibold">
              Backend health
            </h2>

            <p className="mt-1 text-sm leading-6 text-slate-400">
              {healthMessage}
            </p>
          </div>
        </div>
      </GlassPanel>
    </div>
  )
}

function StatusCard({
  icon,
  title,
  value,
}: {
  icon: ReactNode
  title: string
  value: StatusValue
}) {
  return (
    <GlassPanel className="p-7">
      <div className="flex items-center justify-between">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/60">
          {icon}
        </div>

        <span className="rounded-full border border-white/80 bg-white/50 px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-slate-400">
          {value === 'Connected'
            ? 'Live'
            : value === 'Checking'
              ? 'Checking'
              : 'Offline'}
        </span>
      </div>

      <p className="mt-7 text-sm text-slate-500">
        {title}
      </p>

      <p className="mt-1 text-xl font-semibold">
        {value}
      </p>
    </GlassPanel>
  )
}

export default SystemStatus
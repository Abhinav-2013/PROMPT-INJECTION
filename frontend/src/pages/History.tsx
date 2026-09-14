import { useState } from 'react'
import {
  AlertCircle,
  Clock3,
  ShieldCheck,
  Trash2,
} from 'lucide-react'

import GlassPanel from '../components/GlassPanel'
import type {
  AnalyzeResponse,
  DocumentHistoryEntry,
} from '../api/types'

const HISTORY_KEY = 'promptDetectionHistory'

function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    const stored = localStorage.getItem(HISTORY_KEY)

    if (!stored) {
      return []
    }

    try {
      return JSON.parse(stored) as HistoryEntry[]
    } catch {
      localStorage.removeItem(HISTORY_KEY)
      return []
    }
  })

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY)
    setHistory([])
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      <div className="mb-10 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Analysis records
          </p>

          <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
            Detection history.
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
            Review prompts analyzed during this browser session.
          </p>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            onClick={clearHistory}
            className="inline-flex items-center justify-center gap-2 self-start rounded-full border border-white/80 bg-white/45 px-4 py-2.5 text-xs font-medium text-slate-600 shadow-sm backdrop-blur-xl transition hover:bg-white/70 sm:self-auto"
          >
            <Trash2 className="h-4 w-4" />
            Clear History
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <GlassPanel className="flex min-h-[330px] flex-col items-center justify-center p-8 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-3xl border border-white/80 bg-white/60">
            <Clock3 className="h-7 w-7 text-slate-500" />
          </div>

          <h2 className="mt-6 text-xl font-semibold text-slate-800">
            No analyses yet
          </h2>

          <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
            Analyze a prompt and completed analyses will appear here.
          </p>
        </GlassPanel>
      ) : (
        <div className="space-y-4">
          {history.map((item, index) => (
            isDocumentHistoryEntry(item) ? (
              <DocumentHistoryItem key={item.history_id} item={item} />
            ) : (
              <HistoryItem
                key={`${item.prediction_id ?? 'analysis'}-${index}`}
                item={item}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

type HistoryEntry = AnalyzeResponse | DocumentHistoryEntry

function isDocumentHistoryEntry(
  item: HistoryEntry,
): item is DocumentHistoryEntry {
  return 'record_type' in item && item.record_type === 'document'
}

function DocumentHistoryItem({
  item,
}: {
  item: DocumentHistoryEntry
}) {
  const result = item.result
  const decision = result?.document_decision
  const threatPages = result
    ? [...new Set(
        result.results
          .filter((chunk) => chunk.decision === 'BLOCK' || chunk.decision === 'REVIEW')
          .map((chunk) => chunk.page_number)
          .filter((page): page is number => typeof page === 'number'),
      )]
    : []

  return (
    <GlassPanel className="p-6 sm:p-7">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-slate-200/70">
          <Clock3 className="h-5 w-5 text-slate-700" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-white/80 bg-white/55 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              Document
            </span>
            <span className="rounded-full border border-white/80 bg-white/40 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.14em] text-slate-400">
              {decision || item.status}
            </span>
          </div>

          <p className="mt-4 break-words text-sm font-semibold text-slate-700">
            {item.filename}
          </p>

          <p className="mt-2 text-xs text-slate-400">
            {item.status === 'success'
              ? `${result?.threat_chunks ?? 0} threatening chunk(s)`
              : item.error || 'Document scanning failed.'}
            {threatPages.length > 0
              ? ` | Pages: ${threatPages.join(', ')}`
              : ''}
          </p>
        </div>

        <div className="shrink-0 text-xs text-slate-400">
          {new Date(item.scanned_at).toLocaleString()}
        </div>
      </div>
    </GlassPanel>
  )
}

function HistoryItem({
  item,
}: {
  item: AnalyzeResponse
}) {
  const score = Math.min(
    Math.max(item.threat_score * 100, 0),
    100,
  )

  const isThreat =
    item.decision === 'BLOCK' ||
    item.decision === 'REVIEW'

  return (
    <GlassPanel className="p-6 sm:p-7">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center">
        <div
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ${
            isThreat
              ? 'bg-slate-200/70'
              : 'bg-white/60'
          }`}
        >
          {isThreat ? (
            <AlertCircle className="h-5 w-5 text-slate-700" />
          ) : (
            <ShieldCheck className="h-5 w-5 text-slate-500" />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-white/80 bg-white/55 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              {item.decision}
            </span>

            <span className="rounded-full border border-white/80 bg-white/40 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.14em] text-slate-400">
              {item.severity}
            </span>

            {item.prediction_id !== undefined && (
              <span className="text-[10px] uppercase tracking-[0.14em] text-slate-300">
                ID {item.prediction_id}
              </span>
            )}
          </div>

          <p className="mt-4 line-clamp-3 text-sm leading-6 text-slate-600">
            {item.text}
          </p>
        </div>

        <div className="w-full shrink-0 lg:w-40">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-slate-400">
              Threat score
            </span>

            <span className="text-sm font-semibold text-slate-700">
              {score.toFixed(1)}%
            </span>
          </div>

          <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200/60">
            <div
              className="h-full rounded-full bg-slate-700 transition-all duration-500"
              style={{
                width: `${score}%`,
              }}
            />
          </div>
        </div>
      </div>
    </GlassPanel>
  )
}

export default HistoryPage
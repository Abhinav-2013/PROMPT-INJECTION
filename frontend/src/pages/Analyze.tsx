import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  LoaderCircle,
  RotateCcw,
  Send,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import GlassPanel from '../components/GlassPanel'
import {
  analyzePrompt,
  submitFeedback,
} from '../api/athsApi'
import type {
  AnalyzeResponse,
  FeedbackRequest,
} from '../api/types'

const HISTORY_KEY = 'promptDetectionHistory'
const MAX_HISTORY_ITEMS = 50
const MAX_CHARS = 10000

function Analyze() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<AnalyzeResponse | null>(() => {
    const stored = sessionStorage.getItem('latestAnalysis')

    if (!stored) {
      return null
    }

    try {
      return JSON.parse(stored) as AnalyzeResponse
    } catch {
      sessionStorage.removeItem('latestAnalysis')
      return null
    }
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [feedbackLabel, setFeedbackLabel] =
    useState<number | null>(null)

  const [attackCategory, setAttackCategory] =
    useState('')

  const [feedbackMessage, setFeedbackMessage] =
    useState('')

  const [feedbackLoading, setFeedbackLoading] =
    useState(false)

  const [showSemanticNeighbors, setShowSemanticNeighbors] =
    useState(false)

  const characterCount = text.length

  const canAnalyze =
    text.trim().length > 0 &&
    characterCount <= MAX_CHARS &&
    !loading

  const semanticNeighbors = useMemo(() => {
    if (!result?.semantic) {
      return []
    }

    const possible = result.semantic.top_neighbors

    return Array.isArray(possible) ? possible : []
  }, [result])

  const handleAnalyze = async () => {
    const cleanText = text.trim()

    if (!cleanText) {
      setError('Enter a prompt before analyzing.')
      return
    }

    if (cleanText.length > MAX_CHARS) {
      setError(
        `Prompt must be ${MAX_CHARS.toLocaleString()} characters or fewer.`,
      )
      return
    }

    setLoading(true)
    setError('')
    setFeedbackMessage('')
    setFeedbackLabel(null)

    try {
      const response = await analyzePrompt({
        text: cleanText,
      })

      setResult(response)

      sessionStorage.setItem(
        'latestAnalysis',
        JSON.stringify(response),
      )

      saveToHistory(response)
    } catch (error) {
      const message =
        typeof error === 'object' &&
        error !== null &&
        'message' in error &&
        typeof error.message === 'string'
          ? error.message
          : 'The detection service is unavailable.'

      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setText('')
    setResult(null)
    setError('')
    setFeedbackLabel(null)
    setFeedbackMessage('')
    setAttackCategory('')

    sessionStorage.removeItem('latestAnalysis')
  }

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (
      (event.metaKey || event.ctrlKey) &&
      event.key === 'Enter'
    ) {
      event.preventDefault()

      if (canAnalyze) {
        void handleAnalyze()
      }
    }
  }

  const handleFeedback = async (humanLabel: number) => {
    if (
      result?.prediction_id === undefined ||
      feedbackLoading
    ) {
      return
    }

    setFeedbackLabel(humanLabel)
    setFeedbackLoading(true)
    setFeedbackMessage('')
    setError('')

    const payload: FeedbackRequest = {
      prediction_id: result.prediction_id,
      human_label: humanLabel,
      attack_category:
        attackCategory.trim() || null,
    }

    try {
      await submitFeedback(payload)

      setFeedbackMessage(
        'Feedback stored successfully.',
      )
    } catch (error) {
      const message =
        typeof error === 'object' &&
        error !== null &&
        'message' in error &&
        typeof error.message === 'string'
          ? error.message
          : 'Could not save feedback.'

      setError(message)
      setFeedbackLabel(null)
    } finally {
      setFeedbackLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      <div className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Threat analysis
        </p>

        <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
          Analyze a prompt.
        </h1>

        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
          Submit an instruction to the connected detection engine
          and inspect the signals behind its decision.
        </p>
      </div>

      <GlassPanel className="p-6 sm:p-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-slate-800">
              Prompt input
            </p>

            <p className="mt-1 text-xs text-slate-400">
              Press Ctrl+Enter or Cmd+Enter to analyze.
            </p>
          </div>

          <span className="text-xs text-slate-400">
            {characterCount.toLocaleString()} /{' '}
            {MAX_CHARS.toLocaleString()}
          </span>
        </div>

        <textarea
          value={text}
          onChange={(event) => {
            setText(event.target.value)
            setError('')
          }}
          onKeyDown={handleKeyDown}
          maxLength={MAX_CHARS}
          placeholder="Enter a prompt to analyze…"
          className="mt-5 min-h-[260px] w-full resize-y rounded-[1.5rem] border border-white/80 bg-white/45 p-5 text-sm leading-7 text-slate-700 outline-none placeholder:text-slate-300 focus:border-slate-300 focus:bg-white/60"
          aria-label="Prompt to analyze"
        />

        <div className="mt-5 flex flex-col gap-3 sm:flex-row">
          <button
            type="button"
            onClick={() => void handleAnalyze()}
            disabled={!canAnalyze}
            className="inline-flex items-center justify-center gap-2 rounded-full bg-[#111318] px-6 py-3.5 text-sm font-medium text-white transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <>
                <LoaderCircle className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <ShieldCheck className="h-4 w-4" />
                Analyze Prompt
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleClear}
            disabled={loading && !result}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-white/80 bg-white/45 px-6 py-3.5 text-sm font-medium text-slate-700 transition hover:bg-white/70 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RotateCcw className="h-4 w-4" />
            Clear
          </button>
        </div>

        {error && (
          <div className="mt-5 flex items-start gap-3 rounded-2xl border border-slate-200 bg-white/50 p-4 text-sm text-slate-600">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </GlassPanel>

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 space-y-5"
        >
          <ResultSummary result={result} />

          <DetectorBreakdown result={result} />

          <SemanticSection
            result={result}
            neighbors={semanticNeighbors}
            expanded={showSemanticNeighbors}
            onToggle={() =>
              setShowSemanticNeighbors((value) => !value)
            }
          />

          <RulesSection result={result} />

          <FusionSection result={result} />

          <FeedbackSection
            result={result}
            feedbackLabel={feedbackLabel}
            attackCategory={attackCategory}
            feedbackMessage={feedbackMessage}
            loading={feedbackLoading}
            onCategoryChange={setAttackCategory}
            onFeedback={(label) =>
              void handleFeedback(label)
            }
          />

          <div className="flex flex-wrap gap-3">
            <Link
              to="/explainability"
              className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/45 px-5 py-3 text-sm font-medium text-slate-700 backdrop-blur-xl transition hover:bg-white/70"
            >
              <ShieldAlert className="h-4 w-4" />
              View Explainability
            </Link>

            <Link
              to="/history"
              className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/45 px-5 py-3 text-sm font-medium text-slate-700 backdrop-blur-xl transition hover:bg-white/70"
            >
              View History
            </Link>
          </div>
        </motion.div>
      )}
    </div>
  )
}

function ResultSummary({
  result,
}: {
  result: AnalyzeResponse
}) {
  const score = clamp(result.threat_score * 100)

  const isThreat =
    result.decision === 'BLOCK' ||
    result.decision === 'REVIEW'

  return (
    <GlassPanel className="overflow-hidden p-7 sm:p-9">
      <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Detection result
          </p>

          <div className="mt-3 flex items-center gap-3">
            {isThreat ? (
              <AlertCircle className="h-7 w-7 text-slate-700" />
            ) : (
              <CheckCircle2 className="h-7 w-7 text-slate-500" />
            )}

            <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
              {result.decision}
            </h2>
          </div>

          <p className="mt-2 text-sm text-slate-400">
            Severity: {result.severity}
          </p>
        </div>

        <div className="relative flex h-44 w-44 shrink-0 items-center justify-center rounded-full border border-white/80 bg-white/40 shadow-[0_20px_60px_rgba(30,41,59,0.08)]">
          <svg
            className="absolute inset-0 h-full w-full -rotate-90"
            viewBox="0 0 120 120"
          >
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="rgba(148,163,184,0.15)"
              strokeWidth="8"
            />

            <motion.circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="currentColor"
              className="text-slate-700"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={2 * Math.PI * 50}
              initial={{
                strokeDashoffset: 2 * Math.PI * 50,
              }}
              animate={{
                strokeDashoffset:
                  2 * Math.PI * 50 -
                  (score / 100) * 2 * Math.PI * 50,
              }}
              transition={{
                duration: 1,
                ease: 'easeOut',
              }}
            />
          </svg>

          <div className="relative text-center">
            <p className="text-3xl font-semibold tracking-tight">
              {score.toFixed(1)}%
            </p>

            <p className="mt-1 text-[10px] uppercase tracking-[0.16em] text-slate-400">
              Threat score
            </p>
          </div>
        </div>
      </div>
    </GlassPanel>
  )
}

function DetectorBreakdown({
  result,
}: {
  result: AnalyzeResponse
}) {
  return (
    <GlassPanel className="p-7 sm:p-9">
      <div className="mb-7">
        <h2 className="text-lg font-semibold">
          Detector breakdown
        </h2>

        <p className="mt-1 text-sm text-slate-400">
          Individual signals returned by the detection engine.
        </p>
      </div>

      <div className="space-y-6">
        <DetectorBar
          label="Machine learning"
          value={result.ml.probability}
          threat={result.ml.label === 1}
        />

        <DetectorBar
          label="Behavioral rules"
          value={result.rules.score}
          threat={result.rules.is_threat}
        />

        <DetectorBar
          label="Semantic similarity"
          value={result.semantic.score}
          threat={result.semantic.is_threat}
        />
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        <Vote
          label="ML"
          value={result.detector_votes.ml}
        />

        <Vote
          label="Rules"
          value={result.detector_votes.rules}
        />

        <Vote
          label="Semantic"
          value={result.detector_votes.semantic}
        />
      </div>

      <div className="mt-4 rounded-2xl border border-white/70 bg-white/35 p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs uppercase tracking-[0.15em] text-slate-400">
            Detector votes
          </span>

          <span className="font-semibold text-slate-700">
            {result.detector_votes.total}/3
          </span>
        </div>
      </div>
    </GlassPanel>
  )
}

function DetectorBar({
  label,
  value,
  threat,
}: {
  label: string
  value: number
  threat: boolean
}) {
  const percentage = clamp(value * 100)

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4">
        <span className="text-sm font-medium text-slate-600">
          {label}
        </span>

        <span className="text-sm font-semibold text-slate-700">
          {percentage.toFixed(1)}%
        </span>
      </div>

      <div className="h-2.5 overflow-hidden rounded-full bg-slate-200/60">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.7 }}
          className="h-full rounded-full bg-slate-700"
        />
      </div>

      <p className="mt-2 text-xs text-slate-400">
        {threat ? 'Threat' : 'Non-threat'}
      </p>
    </div>
  )
}

function SemanticSection({
  result,
  neighbors,
  expanded,
  onToggle,
}: {
  result: AnalyzeResponse
  neighbors: unknown[]
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <GlassPanel className="p-7 sm:p-9">
      <div className="flex items-start justify-between gap-5">
        <div>
          <h2 className="text-lg font-semibold">
            Semantic evidence
          </h2>

          <p className="mt-1 text-sm text-slate-400">
            Similarity-based evidence returned by the semantic engine.
          </p>
        </div>

        {neighbors.length > 0 && (
          <button
            type="button"
            onClick={onToggle}
            className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/45 px-4 py-2 text-xs font-medium text-slate-600"
          >
            {expanded ? (
              <>
                Hide neighbors
                <ChevronUp className="h-4 w-4" />
              </>
            ) : (
              <>
                Show neighbors
                <ChevronDown className="h-4 w-4" />
              </>
            )}
          </button>
        )}
      </div>

      <div className="mt-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <InfoBox
          label="Similarity"
          value={`${(
            result.semantic.similarity * 100
          ).toFixed(1)}%`}
        />

        <InfoBox
          label="Matched label"
          value={String(
            result.semantic.matched_label,
          )}
        />

        <InfoBox
          label="Malicious neighbors"
          value={String(
            result.semantic.malicious_neighbor_count,
          )}
        />

        <InfoBox
          label="High confidence"
          value={
            result.semantic.high_confidence
              ? 'Yes'
              : 'No'
          }
        />
      </div>

      {result.semantic.matched_text && (
        <div className="mt-5 rounded-2xl border border-white/70 bg-white/35 p-5">
          <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
            Matched semantic text
          </p>

          <p className="mt-3 text-sm leading-6 text-slate-600">
            {result.semantic.matched_text}
          </p>
        </div>
      )}

      {expanded && neighbors.length > 0 && (
        <div className="mt-5 space-y-3">
          {neighbors.map((neighbor, index) => (
            <div
              key={index}
              className="rounded-2xl border border-white/70 bg-white/30 p-5"
            >
              <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
                Neighbor {index + 1}
              </p>

              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-slate-600">
                {formatUnknownValue(neighbor)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </GlassPanel>
  )
}

function RulesSection({
  result,
}: {
  result: AnalyzeResponse
}) {
  return (
    <GlassPanel className="p-7 sm:p-9">
      <h2 className="text-lg font-semibold">
        Rule analysis
      </h2>

      <div className="mt-6">
        <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
          Categories
        </p>

        <div className="mt-3 flex flex-wrap gap-2">
          {result.rules.categories.length > 0 ? (
            result.rules.categories.map((category, index) => (
              <span
                key={`${formatUnknownValue(category)}-${index}`}
                className="rounded-full border border-white/80 bg-white/45 px-3 py-1.5 text-xs text-slate-600"
              >
                {formatUnknownValue(category)}
              </span>
            ))
          ) : (
            <span className="text-sm text-slate-400">
              No rule categories returned.
            </span>
          )}
        </div>
      </div>

      <div className="mt-7">
        <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
          Matched rules
        </p>

        <div className="mt-3 space-y-3">
          {result.rules.matched_rules.length > 0 ? (
            result.rules.matched_rules.map((rule, index) => (
              <RuleCard
                key={`${getRuleName(rule)}-${index}`}
                rule={rule}
              />
            ))
          ) : (
            <div className="rounded-xl border border-white/70 bg-white/30 px-4 py-3 text-sm text-slate-400">
              No matched rules returned.
            </div>
          )}
        </div>
      </div>
    </GlassPanel>
  )
}

function RuleCard({
  rule,
}: {
  rule: unknown
}) {
  if (
    typeof rule === 'object' &&
    rule !== null
  ) {
    const record =
      rule as Record<string, unknown>

    const ruleName =
      typeof record.rule === 'string'
        ? record.rule
        : 'Matched rule'

    const category =
      typeof record.category === 'string'
        ? record.category
        : null

    const weight =
      typeof record.weight === 'number'
        ? record.weight
        : null

    const matchedPatterns =
      Array.isArray(record.matched_patterns)
        ? record.matched_patterns
        : []

    return (
      <div className="rounded-xl border border-white/70 bg-white/30 px-4 py-4 text-sm text-slate-600">
        <div className="font-medium text-slate-700">
          {ruleName}
        </div>

        {category && (
          <div className="mt-1 text-xs text-slate-400">
            Category: {category}
          </div>
        )}

        {weight !== null && (
          <div className="mt-1 text-xs text-slate-400">
            Weight: {weight}
          </div>
        )}

        {matchedPatterns.length > 0 && (
          <div className="mt-3">
            <p className="text-xs uppercase tracking-[0.12em] text-slate-400">
              Matched patterns
            </p>

            <div className="mt-2 flex flex-wrap gap-2">
              {matchedPatterns.map(
                (pattern, patternIndex) => (
                  <span
                    key={patternIndex}
                    className="rounded-full border border-white/80 bg-white/50 px-3 py-1 text-xs text-slate-600"
                  >
                    {formatUnknownValue(pattern)}
                  </span>
                ),
              )}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-white/70 bg-white/30 px-4 py-3 text-sm text-slate-600">
      {formatUnknownValue(rule)}
    </div>
  )
}

function FusionSection({
  result,
}: {
  result: AnalyzeResponse
}) {
  return (
    <GlassPanel className="p-7 sm:p-9">
      <h2 className="text-lg font-semibold">
        Threat fusion
      </h2>

      <p className="mt-1 text-sm text-slate-400">
        Combined threat score and detector weights returned by the backend.
      </p>

      <div className="mt-7 grid gap-4 sm:grid-cols-3">
        <InfoBox
          label="ML weight"
          value={formatWeight(result.fusion.weights?.ml)}
        />

        <InfoBox
          label="Rules weight"
          value={formatWeight(result.fusion.weights?.rules)}
        />

        <InfoBox
          label="Semantic weight"
          value={formatWeight(result.fusion.weights?.semantic)}
        />
      </div>

      <div className="mt-5 rounded-2xl border border-white/70 bg-white/35 p-5">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-500">
            Backend fusion score
          </span>

          <span className="text-lg font-semibold text-slate-800">
            {result.fusion.threat_score.toFixed(4)}
          </span>
        </div>
      </div>
    </GlassPanel>
  )
}

function FeedbackSection({
  result,
  feedbackLabel,
  attackCategory,
  feedbackMessage,
  loading,
  onCategoryChange,
  onFeedback,
}: {
  result: AnalyzeResponse
  feedbackLabel: number | null
  attackCategory: string
  feedbackMessage: string
  loading: boolean
  onCategoryChange: (value: string) => void
  onFeedback: (label: number) => void
}) {
  const canSendFeedback =
    result.prediction_id !== undefined

  return (
    <GlassPanel className="p-7 sm:p-9">
      <div>
        <h2 className="text-lg font-semibold">
          Human feedback
        </h2>

        <p className="mt-1 text-sm leading-6 text-slate-400">
          Help improve future detection by marking whether this
          prediction was correct.
        </p>
      </div>

      {!canSendFeedback ? (
        <div className="mt-6 rounded-2xl border border-white/70 bg-white/35 p-5 text-sm text-slate-400">
          Feedback is unavailable because the backend did not return
          a prediction ID for this analysis.
        </div>
      ) : (
        <>
          <div className="mt-6">
            <label
              htmlFor="attack-category"
              className="text-xs font-medium uppercase tracking-[0.15em] text-slate-400"
            >
              Attack category
            </label>

            <select
              id="attack-category"
              value={attackCategory}
              onChange={(event) =>
                onCategoryChange(event.target.value)
              }
              className="mt-2 w-full rounded-2xl border border-white/80 bg-white/45 px-4 py-3 text-sm text-slate-700 outline-none focus:border-slate-300 sm:max-w-md"
            >
              <option value="">
                Select category (optional)
              </option>

              <option value="instruction_override">
                Instruction override
              </option>

              <option value="system_prompt_extraction">
                System prompt extraction
              </option>

              <option value="jailbreak">
                Jailbreak
              </option>

              <option value="data_exfiltration">
                Data exfiltration
              </option>

              <option value="role_manipulation">
                Role manipulation
              </option>

              <option value="other">
                Other
              </option>
            </select>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              type="button"
              disabled={loading}
              onClick={() => onFeedback(1)}
              className={`inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium transition ${
                feedbackLabel === 1
                  ? 'bg-[#111318] text-white'
                  : 'border border-white/80 bg-white/45 text-slate-700 hover:bg-white/70'
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <ShieldAlert className="h-4 w-4" />
              Threat
            </button>

            <button
              type="button"
              disabled={loading}
              onClick={() => onFeedback(0)}
              className={`inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium transition ${
                feedbackLabel === 0
                  ? 'bg-[#111318] text-white'
                  : 'border border-white/80 bg-white/45 text-slate-700 hover:bg-white/70'
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <CheckCircle2 className="h-4 w-4" />
              Not a Threat
            </button>
          </div>

          {loading && (
            <div className="mt-4 flex items-center gap-2 text-sm text-slate-400">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Saving feedback...
            </div>
          )}

          {feedbackMessage && (
            <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
              <Send className="h-4 w-4" />
              {feedbackMessage}
            </div>
          )}
        </>
      )}
    </GlassPanel>
  )
}

function Vote({
  label,
  value,
}: {
  label: string
  value: boolean
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-sm font-semibold text-slate-700">
        {value ? 'Threat' : 'Non-threat'}
      </p>
    </div>
  )
}

function InfoBox({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-lg font-semibold text-slate-800">
        {value}
      </p>
    </div>
  )
}

function formatWeight(value?: number) {
  if (typeof value !== 'number') {
    return 'Unavailable'
  }

  return `${(value * 100).toFixed(0)}%`
}

function formatUnknownValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  if (
    typeof value === 'number' ||
    typeof value === 'boolean' ||
    typeof value === 'bigint'
  ) {
    return String(value)
  }

  if (value === null || value === undefined) {
    return 'Unavailable'
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function getRuleName(rule: unknown): string {
  if (
    typeof rule === 'object' &&
    rule !== null
  ) {
    const record =
      rule as Record<string, unknown>

    if (typeof record.rule === 'string') {
      return record.rule
    }
  }

  return formatUnknownValue(rule)
}

function clamp(value: number) {
  return Math.min(Math.max(value, 0), 100)
}

function saveToHistory(result: AnalyzeResponse) {
  try {
    const stored = localStorage.getItem(HISTORY_KEY)

    const history = stored
      ? (JSON.parse(stored) as AnalyzeResponse[])
      : []

    const updated = [
      result,
      ...history.filter(
        (item) =>
          item.prediction_id !== result.prediction_id,
      ),
    ].slice(0, MAX_HISTORY_ITEMS)

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(updated),
    )
  } catch {
    // History storage should never prevent an analysis from succeeding.
  }
}

export default Analyze
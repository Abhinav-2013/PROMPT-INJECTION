import { useState, type ReactNode } from 'react'
import {
  BrainCircuit,
  ChevronDown,
  FileSearch,
  Files,
  GitBranch,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'

import GlassPanel from '../components/GlassPanel'
import type {
  AnalyzeResponse,
  DocumentChunkResult,
  DocumentScanResponse,
  DocumentScanResult,
} from '../api/types'

function ChunkExplainability() {
  const [scan] = useState<DocumentScanResponse | null>(() => {
    const stored = sessionStorage.getItem('latestDocumentScan')
    if (!stored) return null

    try {
      return JSON.parse(stored) as DocumentScanResponse
    } catch {
      sessionStorage.removeItem('latestDocumentScan')
      return null
    }
  })

  if (!scan) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
        <GlassPanel className="flex min-h-[300px] flex-col items-center justify-center p-8 text-center">
          <FileSearch className="h-10 w-10 text-slate-500" />
          <h1 className="mt-5 text-2xl font-semibold">No document scan available</h1>
          <p className="mt-2 text-sm text-slate-400">Scan a document first to inspect every analyzed chunk.</p>
        </GlassPanel>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      <div className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Document explainability</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">Inspect every chunk.</h1>
        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">Review the page-aware evidence returned by the document scanner.</p>
      </div>

      <ScanSummary scan={scan} />

      <div className="space-y-6">
        {scan.documents.map((document) => (
          <DocumentExplanation key={document.filename} document={document.result} filename={document.filename} />
        ))}
      </div>
    </div>
  )
}

function ScanSummary({ scan }: { scan: DocumentScanResponse }) {
  const successful = scan.documents.filter((document) => document.result)
  const blocked = successful.filter((document) => document.result?.document_decision === 'BLOCK').length
  const reviews = successful.filter((document) => document.result?.document_decision === 'REVIEW').length
  const chunks = successful.reduce((total, document) => total + (document.result?.chunk_count ?? 0), 0)
  const threats = successful.reduce((total, document) => total + (document.result?.threat_chunks ?? 0), 0)

  return (
    <GlassPanel className="mb-6 overflow-hidden p-6 sm:p-8">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-900 text-white">
            <Files className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Scan overview</p>
            <h2 className="mt-1 text-xl font-semibold text-slate-900">{scan.document_count} document{scan.document_count === 1 ? '' : 's'} analyzed</h2>
          </div>
        </div>
        <DecisionBadge decision={scan.overall_decision} />
      </div>
      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
        <SummaryMetric label="Documents" value={successful.length} />
        <SummaryMetric label="Blocked" value={blocked} />
        <SummaryMetric label="Review" value={reviews} />
        <SummaryMetric label="Chunks" value={chunks} />
        <SummaryMetric label="Threat chunks" value={threats} />
      </div>
    </GlassPanel>
  )
}

function SummaryMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-800">{value}</p>
    </div>
  )
}

function DocumentExplanation({
  document,
  filename,
}: {
  document?: DocumentScanResult
  filename: string
}) {
  if (!document) {
    return <GlassPanel className="p-6 text-sm text-slate-600">{filename}: scan failed.</GlassPanel>
  }

  return (
    <section className="space-y-4">
      <GlassPanel className="p-7 sm:p-9">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Document assessment</p>
            <h2 className="mt-2 break-words text-2xl font-semibold">{filename}</h2>
            <p className="mt-2 text-sm text-slate-500">{document.chunk_count} chunks, {document.threat_chunks} threatening chunks</p>
          </div>
          <DecisionBadge decision={document.document_decision} />
        </div>

        {document.document_explanation && (
          <div className="mt-7 border-t border-white/70 pt-6">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-slate-500" />
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-400">Primary behavior</p>
            </div>
            <p className="mt-2 text-base font-semibold text-slate-800">{document.document_explanation.primary_hypothesis}</p>
            {document.document_explanation.supporting_evidence.length > 0 && (
              <ul className="mt-3 grid gap-2 sm:grid-cols-2">
                {document.document_explanation.supporting_evidence.map((item) => (
                  <li key={item} className="rounded-xl border border-white/70 bg-white/35 px-3 py-2 text-sm leading-5 text-slate-600">{item}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </GlassPanel>

      <div className="space-y-4">
        {document.results.map((chunk, index) => (
          <ChunkExplanation key={`${filename}-${chunk.chunk_index ?? index}`} chunk={chunk} index={index} />
        ))}
      </div>
    </section>
  )
}

function ChunkExplanation({ chunk, index }: { chunk: DocumentChunkResult; index: number }) {
  const [open, setOpen] = useState(chunk.decision !== 'ALLOW')
  const analysis = normalizeChunkAnalysis(chunk.full_result)

  return (
    <GlassPanel className={`overflow-hidden border-l-4 ${chunk.decision === 'BLOCK' ? 'border-l-red-400' : chunk.decision === 'REVIEW' ? 'border-l-amber-400' : 'border-l-emerald-400'}`}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-4 p-6 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        aria-expanded={open}
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Chunk {chunk.chunk_index ?? index + 1}</span>
            {typeof chunk.page_number === 'number' && <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">Page {chunk.page_number}</span>}
            <DecisionBadge decision={chunk.decision ?? 'ALLOW'} />
          </div>
          <p className="mt-3 line-clamp-2 text-sm leading-6 text-slate-600">{chunk.text}</p>
        </div>
        <ChevronDown className={`h-5 w-5 shrink-0 text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && <div className="border-t border-white/70 p-6"><AnalysisDetails analysis={analysis} /></div>}
    </GlassPanel>
  )
}

function normalizeChunkAnalysis(
  response?: AnalyzeResponse,
): AnalyzeResponse | undefined {
  if (!response) return undefined
  const rules = response.rules ?? ({} as AnalyzeResponse['rules'])
  const semantic = response.semantic ?? ({} as AnalyzeResponse['semantic'])
  const rawRules = rules as AnalyzeResponse['rules'] & { rule_score?: number }
  const rawSemantic = semantic as AnalyzeResponse['semantic'] & {
    similarity_score?: number
    semantic_threat_score?: number
  }
  const matchedRules = rawRules.matched_rules?.length ? rawRules.matched_rules : response.explainability?.rules?.matched_rules ?? []
  const unifiedSemantic = response.explainability?.semantic
  const categories = rawRules.categories?.length ? rawRules.categories : response.explainability?.rules?.categories ?? matchedRules.map((rule) => rule.category)

  return {
    ...response,
    rules: {
      ...rawRules,
      score: rawRules.score ?? rawRules.rule_score ?? 0,
      categories: [...new Set(categories)],
      matched_rules: matchedRules,
    },
    semantic: {
      ...rawSemantic,
      similarity: rawSemantic.similarity ?? rawSemantic.similarity_score ?? unifiedSemantic?.similarity ?? 0,
      score: rawSemantic.score ?? rawSemantic.semantic_threat_score ?? unifiedSemantic?.score ?? 0,
      threat_score: rawSemantic.threat_score ?? rawSemantic.semantic_threat_score ?? unifiedSemantic?.threat_score ?? 0,
    },
  }
}

function AnalysisDetails({ analysis }: { analysis?: AnalyzeResponse }) {
  if (!analysis) return <p className="text-sm text-slate-500">No detailed analysis was returned for this chunk.</p>

  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-3">
        <Evidence label="Random Forest" value={formatScore(analysis.ml?.probability)} icon={<BrainCircuit className="h-4 w-4" />} />
        <Evidence label="Rules" value={formatScore(analysis.rules?.score)} icon={<ShieldAlert className="h-4 w-4" />} />
        <Evidence label="Semantic" value={formatScore(analysis.semantic?.score)} icon={<GitBranch className="h-4 w-4" />} />
      </div>
      <EvidenceBlock title="Behavior rules" value={formatRules(analysis)} />
      <div className="grid gap-3 sm:grid-cols-2">
        <EvidenceBlock title="Rule categories" value={analysis.rules?.categories?.join(', ') || 'No rule categories detected'} />
        <EvidenceBlock title="Detector votes" value={formatVotes(analysis)} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <EvidenceBlock title="Semantic similarity" value={formatSemantic(analysis)} />
        <EvidenceBlock title="Threat fusion" value={formatFusion(analysis)} />
      </div>
      <EvidenceBlock title="Semantic neighbors" value={formatNeighbors(analysis)} />
      <EvidenceBlock title="Attack hypothesis" value={analysis.hypothesis || 'No hypothesis returned'} />
      <EvidenceBlock title="Evidence" value={formatEvidence(analysis)} />
      <LogisticRegressionDetails analysis={analysis} />
      <div className="flex items-center gap-2 text-xs text-slate-500"><ShieldCheck className="h-4 w-4" /> Final fusion: {analysis.decision} at {formatScore(analysis.threat_score)}</div>
    </div>
  )
}

function formatEvidence(analysis: AnalyzeResponse) {
  const evidence = analysis.evidence ?? []
  if (evidence.length === 0) return 'No additional evidence returned'
  return evidence.map((item) => {
    if (typeof item === 'string') return item
    return item.title || item.description || String(item.value ?? '')
  }).filter(Boolean).join('; ')
}

function LogisticRegressionDetails({ analysis }: { analysis: AnalyzeResponse }) {
  const explanation = analysis.explainability?.logistic_regression
  if (!explanation) {
    return <EvidenceBlock title="Logistic Regression" value="Explanation was not returned for this chunk." />
  }

  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-4">
      <p className="text-xs uppercase tracking-[0.12em] text-slate-400">Logistic Regression</p>
      <p className="mt-2 text-sm text-slate-600">
        {explanation.prediction === 1 ? 'Malicious' : 'Benign'} prediction, {(explanation.probability_malicious * 100).toFixed(1)}% malicious probability, intercept {explanation.intercept.toFixed(4)}.
      </p>
      <div className="mt-3 space-y-2">
        {explanation.top_features.map((feature) => (
          <p key={feature.feature} className="text-xs text-slate-500">
            {feature.feature}: contribution {feature.contribution >= 0 ? '+' : ''}{feature.contribution.toFixed(4)} ({feature.direction.replace('_', ' ')})
          </p>
        ))}
      </div>
    </div>
  )
}

function formatRules(analysis: AnalyzeResponse) {
  const rules = analysis.rules?.matched_rules ?? []
  if (rules.length === 0) return 'No matched behavior rules'
  return rules.map((rule) => {
    const patterns = rule.matched_patterns?.join(', ') || 'pattern unavailable'
    return `${rule.rule} (${rule.category}): ${patterns}`
  }).join(' | ')
}

function formatSemantic(analysis: AnalyzeResponse) {
  const semantic = analysis.semantic
  if (!semantic) return 'No semantic result returned'
  const similarity = typeof semantic.similarity === 'number' ? `${(semantic.similarity * 100).toFixed(1)}%` : 'N/A'
  const score = typeof semantic.score === 'number' ? `${(semantic.score * 100).toFixed(1)}%` : 'N/A'
  return `${similarity} similarity, ${score} threat score, ${semantic.malicious_neighbor_count ?? 0} malicious neighbors`
}

function formatNeighbors(analysis: AnalyzeResponse) {
  const neighbors = analysis.semantic?.top_neighbors ?? []
  if (neighbors.length === 0) return 'No semantic neighbors returned'
  return neighbors.slice(0, 5).map((neighbor, index) => {
    const similarity = typeof neighbor.similarity === 'number' ? `${(neighbor.similarity * 100).toFixed(1)}%` : 'N/A'
    const label = neighbor.label === 1 ? 'malicious' : 'benign'
    return `${index + 1}. ${similarity} (${label}): ${neighbor.text || 'text unavailable'}`
  }).join(' | ')
}

function formatVotes(analysis: AnalyzeResponse) {
  const votes = analysis.detector_votes
  return `${votes.total}/3 threat votes: ML ${votes.ml ? 'threat' : 'clear'}, rules ${votes.rules ? 'threat' : 'clear'}, semantic ${votes.semantic ? 'threat' : 'clear'}`
}

function formatFusion(analysis: AnalyzeResponse) {
  const fusion = analysis.fusion
  return `Final ${fusion.decision} at ${(analysis.threat_score * 100).toFixed(1)}%; weights ML ${fusion.weights?.ml ?? 'N/A'}, rules ${fusion.weights?.rules ?? 'N/A'}, semantic ${fusion.weights?.semantic ?? 'N/A'}`
}

function DecisionBadge({ decision }: { decision: string }) {
  const style = decision === 'BLOCK' ? 'border-red-200 bg-red-50 text-red-700' : decision === 'REVIEW' ? 'border-amber-200 bg-amber-50 text-amber-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'
  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${style}`}>{decision}</span>
}

function Evidence({ label, value, icon }: { label: string; value: string; icon: ReactNode }) {
  return <div className="rounded-2xl border border-white/70 bg-white/35 p-4"><div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-slate-400">{icon}{label}</div><p className="mt-2 text-lg font-semibold text-slate-800">{value}</p></div>
}

function EvidenceBlock({ title, value }: { title: string; value: string }) {
  return <div className="rounded-2xl border border-white/70 bg-white/35 p-4"><p className="text-xs uppercase tracking-[0.12em] text-slate-400">{title}</p><p className="mt-2 break-words text-sm leading-6 text-slate-600">{value}</p></div>
}

function formatScore(value: number | undefined) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'N/A'
}

export default ChunkExplainability

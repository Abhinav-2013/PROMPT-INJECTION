import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  GitBranch,
  Loader2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from 'lucide-react'

import GlassPanel from '../components/GlassPanel'
import { explainPrompt } from '../api/athsApi'
import type {
  AnalyzeResponse,
  ExplainResponse,
  SHAPFeatureContribution,
} from '../api/types'

function Explainability() {
  const [analysis] = useState<AnalyzeResponse | null>(() => {
    const stored = sessionStorage.getItem('latestAnalysis')

    if (!stored) {
      return null
    }

    try {
      return normalizeAnalysisResponse(
        JSON.parse(stored) as AnalyzeResponse,
      )
    } catch {
      sessionStorage.removeItem('latestAnalysis')
      return null
    }
  })
  const [explanation, setExplanation] =
    useState<ExplainResponse | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!analysis?.text) {
      return
    }

    const timeoutId = window.setTimeout(() => {
      setLoading(true)
      setError(null)

      explainPrompt(analysis.text)
        .then((response) => {
          setExplanation(response)
        })
        .catch((err) => {
          const message =
            typeof err === 'object' &&
            err !== null &&
            'message' in err
              ? String(
                  (err as { message?: unknown }).message ??
                    'Explainability request failed.',
                )
              : 'Explainability request failed.'

          setError(message)
        })
        .finally(() => {
          setLoading(false)
        })
    }, 0)

    return () => window.clearTimeout(timeoutId)
  }, [analysis])

  if (!analysis) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
        <div className="mb-10">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Explainable detection
          </p>

          <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
            Understand the decision.
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
            Run a prompt analysis first. The detection signals and
            model attribution from the latest backend analysis will
            appear here.
          </p>
        </div>

        <GlassPanel className="flex min-h-[300px] flex-col items-center justify-center p-8 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-3xl border border-white/80 bg-white/60">
            <BrainCircuit className="h-7 w-7 text-slate-500" />
          </div>

          <h2 className="mt-6 text-xl font-semibold text-slate-800">
            No analysis available
          </h2>

          <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
            Analyze a prompt first to populate this page with real
            ML, rule-based, semantic, fusion, and SHAP signals.
          </p>
        </GlassPanel>
      </div>
    )
  }

  const score = analysis.threat_score * 100
  const semanticEvidence = analysis.semantic_evidence

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      {/* ------------------------------------------------------------------ */}
      {/* Header                                                             */}
      {/* ------------------------------------------------------------------ */}

      <div className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Explainable detection
        </p>

        <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
          Understand the decision.
        </h1>

        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
          Inspect the real detection signals and Random Forest
          feature attribution returned by the backend.
        </p>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Final decision                                                      */}
      {/* ------------------------------------------------------------------ */}

      <GlassPanel className="p-7 sm:p-9">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3">
              {analysis.decision === 'BLOCK' ? (
                <AlertCircle className="h-6 w-6 text-slate-700" />
              ) : analysis.decision === 'REVIEW' ? (
                <TriangleAlert className="h-6 w-6 text-slate-600" />
              ) : (
                <CheckCircle2 className="h-6 w-6 text-slate-500" />
              )}

              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Final decision
                </p>

                <p className="mt-1 text-3xl font-semibold tracking-tight text-slate-900">
                  {analysis.decision}
                </p>
              </div>
            </div>

            <div className="mt-7 rounded-2xl border border-white/70 bg-white/35 p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Analyzed prompt
              </p>

              <p className="mt-3 text-sm leading-7 text-slate-600">
                {analysis.text}
              </p>
            </div>
          </div>

          <div className="flex h-44 w-44 shrink-0 items-center justify-center rounded-full border border-white/80 bg-white/45 shadow-[0_20px_60px_rgba(30,41,59,0.08)]">
            <div className="text-center">
              <p className="text-4xl font-semibold tracking-tight text-slate-900">
                {score.toFixed(1)}%
              </p>

              <p className="mt-1 text-xs uppercase tracking-[0.15em] text-slate-400">
                Threat score
              </p>
            </div>
          </div>
        </div>
      </GlassPanel>

      {/* ------------------------------------------------------------------ */}
      {/* Detector signals                                                    */}
      {/* ------------------------------------------------------------------ */}

      <div className="mt-5 grid gap-5 md:grid-cols-3">
        <SignalCard
          icon={<BrainCircuit className="h-5 w-5" />}
          title="Machine learning"
          score={analysis.ml.probability}
          description={
            analysis.ml.label === 1
              ? 'The Random Forest detector classified this prompt as a threat.'
              : 'The Random Forest detector classified this prompt as non-threatening.'
          }
        />

        <SignalCard
          icon={<ShieldCheck className="h-5 w-5" />}
          title="Behavioral rules"
          score={analysis.rules.score}
          description={
            analysis.rules.is_threat
              ? 'Security rules identified suspicious behavior.'
              : 'No rule-based threat condition was triggered.'
          }
        />

        <SignalCard
          icon={<GitBranch className="h-5 w-5" />}
          title="Semantic analysis"
          score={analysis.semantic.score}
          description={
            analysis.semantic.is_threat
              ? 'Semantic similarity indicates potentially malicious intent.'
              : 'Semantic similarity did not cross the threat threshold.'
          }
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Threat fusion                                                       */}
      {/* ------------------------------------------------------------------ */}

      <GlassPanel className="mt-5 p-7 sm:p-9">
        <div className="flex items-center gap-3">
          <GitBranch className="h-5 w-5 text-slate-500" />

          <div>
            <h2 className="text-lg font-semibold">
              Detection fusion
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              The final threat score combines the detector signals
              returned by the backend.
            </p>
          </div>
        </div>

        <div className="mt-7 grid gap-4 sm:grid-cols-3">
          <FusionValue
            label="ML"
            value={analysis.ml.probability}
            weight={analysis.fusion.weights?.ml}
          />

          <FusionValue
            label="Rules"
            value={analysis.rules.score}
            weight={analysis.fusion.weights?.rules}
          />

          <FusionValue
            label="Semantic"
            value={analysis.semantic.score}
            weight={analysis.fusion.weights?.semantic}
          />
        </div>

        <div className="mt-7 rounded-2xl border border-white/70 bg-white/35 p-5">
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm text-slate-500">
              Final threat score
            </span>

            <span className="text-lg font-semibold text-slate-800">
              {analysis.threat_score.toFixed(4)}
            </span>
          </div>

          <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200/60">
            <div
              className="h-full rounded-full bg-slate-700 transition-all duration-700"
              style={{
                width: `${Math.min(
                  analysis.threat_score * 100,
                  100,
                )}%`,
              }}
            />
          </div>
        </div>
      </GlassPanel>

      {/* ------------------------------------------------------------------ */}
      {/* Detector voting                                                     */}
      {/* ------------------------------------------------------------------ */}

      <GlassPanel className="mt-5 p-7 sm:p-9">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-5 w-5 text-slate-500" />

          <div>
            <h2 className="text-lg font-semibold">
              Detector voting
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Each detector contributes one threat/non-threat vote.
            </p>
          </div>
        </div>

        <div className="mt-7 grid gap-4 sm:grid-cols-4">
          <Vote
            label="ML"
            value={analysis.detector_votes.ml}
          />

          <Vote
            label="Rules"
            value={analysis.detector_votes.rules}
          />

          <Vote
            label="Semantic"
            value={analysis.detector_votes.semantic}
          />

          <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
            <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
              Total
            </p>

            <p className="mt-2 text-2xl font-semibold text-slate-800">
              {analysis.detector_votes.total}/3
            </p>
          </div>
        </div>
      </GlassPanel>

      {/* ------------------------------------------------------------------ */}
      {/* Semantic evidence                                                   */}
      {/* ------------------------------------------------------------------ */}

      <GlassPanel className="mt-5 p-7 sm:p-9">
        <div className="flex items-center gap-3">
          <GitBranch className="h-5 w-5 text-slate-500" />

          <div>
            <h2 className="text-lg font-semibold">
              Semantic evidence
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Evidence returned by the semantic similarity detector.
            </p>
          </div>
        </div>

        {semanticEvidence ? (
          <>
            <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <EvidenceValue
                label="Similarity"
                value={`${(
                  semanticEvidence.similarity * 100
                ).toFixed(1)}%`}
              />

              <EvidenceValue
                label="Matched label"
                value={String(
                  semanticEvidence.matched_label,
                )}
              />

              <EvidenceValue
                label="Malicious neighbors"
                value={String(
                  semanticEvidence.malicious_neighbor_count,
                )}
              />

              <EvidenceValue
                label="High confidence"
                value={
                  semanticEvidence.high_confidence
                    ? 'Yes'
                    : 'No'
                }
              />
            </div>
          </>
        ) : (
          <p className="mt-6 text-sm text-slate-400">
            Semantic evidence was not included in this analysis
            response.
          </p>
        )}
      </GlassPanel>

      <LogisticRegressionPanel
        analysis={analysis}
        fallback={explanation?.logistic_regression}
      />

      <GlassPanel className="mt-5 p-7 sm:p-9">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-5 w-5 text-slate-500" />
          <div>
            <h2 className="text-lg font-semibold">Behavior rule analysis</h2>
            <p className="mt-1 text-sm text-slate-400">
              Exact rule categories and matched patterns returned by the backend.
            </p>
          </div>
        </div>

        {analysis.rules.matched_rules.length > 0 ? (
          <div className="mt-6 space-y-3">
            {analysis.rules.matched_rules.map((rule) => (
              <div key={rule.rule} className="rounded-2xl border border-white/70 bg-white/35 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="font-medium text-slate-800">{rule.rule}</p>
                  <span className="rounded-full bg-white/60 px-3 py-1 text-xs text-slate-500">{rule.category} · weight {rule.weight.toFixed(2)}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-600">Matched: {rule.matched_patterns.join(', ')}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-6 text-sm text-slate-400">No behavior rules matched this prompt.</p>
        )}
      </GlassPanel>

      <GlassPanel className="mt-5 p-7 sm:p-9">
        <div className="flex items-center gap-3">
          <GitBranch className="h-5 w-5 text-slate-500" />
          <div>
            <h2 className="text-lg font-semibold">Semantic similarity evidence</h2>
            <p className="mt-1 text-sm text-slate-400">Nearest dataset examples used by the semantic detector.</p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <EvidenceValue label="Similarity" value={`${(analysis.semantic.similarity * 100).toFixed(1)}%`} />
          <EvidenceValue label="Threat score" value={`${(analysis.semantic.score * 100).toFixed(1)}%`} />
          <EvidenceValue label="Malicious neighbors" value={String(analysis.semantic.malicious_neighbor_count ?? 0)} />
        </div>

        <div className="mt-6 space-y-3">
          {(analysis.semantic.top_neighbors ?? []).map((neighbor, index) => (
            <div key={`${neighbor.index ?? index}-${neighbor.text ?? ''}`} className="rounded-2xl border border-white/70 bg-white/35 p-4">
              <p className="text-xs uppercase tracking-[0.12em] text-slate-400">
                Neighbor {index + 1} · {typeof neighbor.similarity === 'number' ? `${(neighbor.similarity * 100).toFixed(1)}%` : 'N/A'} · {neighbor.label === 1 ? 'malicious' : 'benign'}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-600">{neighbor.text || 'Matching text unavailable.'}</p>
            </div>
          ))}
        </div>
      </GlassPanel>

      {/* ------------------------------------------------------------------ */}
      {/* SHAP explainability                                                 */}
      {/* ------------------------------------------------------------------ */}

      <GlassPanel className="mt-5 overflow-hidden p-7 sm:p-9">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/60">
                <Sparkles className="h-5 w-5 text-slate-600" />
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Model explainability
                </p>

                <h2 className="mt-1 text-xl font-semibold text-slate-900">
                  Random Forest feature attribution
                </h2>
              </div>
            </div>

            <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-500">
              SHAP shows which of the 15 backend security features
              pushed the Random Forest prediction toward or away
              from a malicious classification.
            </p>
          </div>

          {explanation && (
            <div className="shrink-0 rounded-2xl border border-white/70 bg-white/40 px-5 py-4">
              <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
                Model prediction
              </p>

              <p className="mt-1 text-lg font-semibold text-slate-800">
                {explanation.xai.prediction === 1
                  ? 'Malicious'
                  : 'Benign'}
              </p>

              <p className="mt-1 text-sm text-slate-500">
                {(
                  explanation.xai.probability_malicious * 100
                ).toFixed(1)}% malicious probability
              </p>
            </div>
          )}
        </div>

        {loading && (
          <div className="mt-8 flex min-h-[180px] flex-col items-center justify-center rounded-3xl border border-white/70 bg-white/25">
            <Loader2 className="h-7 w-7 animate-spin text-slate-500" />

            <p className="mt-4 text-sm font-medium text-slate-600">
              Calculating SHAP attribution...
            </p>

            <p className="mt-1 text-xs text-slate-400">
              Using the backend Random Forest explainer.
            </p>
          </div>
        )}

        {!loading && error && (
          <div className="mt-8 rounded-3xl border border-white/70 bg-white/30 p-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-slate-600" />

              <div>
                <p className="font-medium text-slate-800">
                  Explainability unavailable
                </p>

                <p className="mt-1 text-sm leading-6 text-slate-500">
                  {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && explanation && (
          <SHAPContent explanation={explanation} />
        )}
      </GlassPanel>
    </div>
  )
}

function LogisticRegressionPanel({
  analysis,
  fallback,
}: {
  analysis: AnalyzeResponse
  fallback?: NonNullable<ExplainResponse['logistic_regression']>
}) {
  const explanation = analysis.explainability?.logistic_regression ?? fallback

  return (
    <GlassPanel className="mt-5 p-7 sm:p-9">
      <div className="flex items-center gap-3">
        <BrainCircuit className="h-5 w-5 text-slate-500" />
        <div>
          <h2 className="text-lg font-semibold">Logistic Regression explanation</h2>
          <p className="mt-1 text-sm text-slate-400">
            Coefficients multiplied by scaled feature values show the linear model&apos;s contribution.
          </p>
        </div>
      </div>

      {explanation ? (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <EvidenceValue label="Prediction" value={explanation.prediction === 1 ? 'Malicious' : 'Benign'} />
            <EvidenceValue label="Malicious probability" value={`${(explanation.probability_malicious * 100).toFixed(1)}%`} />
            <EvidenceValue label="Intercept" value={explanation.intercept.toFixed(4)} />
          </div>
          <div className="mt-6 space-y-3">
            {explanation.top_features.map((feature) => (
              <div key={feature.feature} className="rounded-2xl border border-white/70 bg-white/35 p-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-slate-800">{formatFeatureName(feature.feature)}</span>
                  <span className="text-sm font-semibold text-slate-600">{feature.contribution >= 0 ? '+' : ''}{feature.contribution.toFixed(4)}</span>
                </div>
                <p className="mt-1 text-xs text-slate-400">Coefficient {feature.coefficient.toFixed(4)} · scaled value {feature.scaled_value.toFixed(4)} · {feature.direction.replace('_', ' ')}</p>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="mt-6 text-sm text-slate-400">Logistic Regression explanation was not returned for this stored analysis. Run the prompt again to refresh it.</p>
      )}
    </GlassPanel>
  )
}

function normalizeAnalysisResponse(
  response: AnalyzeResponse,
): AnalyzeResponse {
  const unifiedRules = response.explainability?.rules
  const rawRules = (response.rules ?? {}) as AnalyzeResponse['rules'] & {
    rule_score?: number
    score?: number
  }
  const matchedRules = rawRules.matched_rules?.length
    ? rawRules.matched_rules
    : unifiedRules?.matched_rules ?? []
  const categories = rawRules.categories?.length
    ? rawRules.categories
    : unifiedRules?.categories?.length
      ? unifiedRules.categories
    : matchedRules
        .map((rule) => rule.category)
        .filter((category): category is string => Boolean(category))
  const unifiedSemantic = response.explainability?.semantic
  const rawSemantic = (response.semantic ?? {}) as AnalyzeResponse['semantic'] & {
    similarity_score?: number
    semantic_threat_score?: number
  }

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

/* ========================================================================== */
/* SHAP content                                                               */
/* ========================================================================== */

function SHAPContent({
  explanation,
}: {
  explanation: ExplainResponse
}) {
  const topFeatures = explanation.xai.top_features

  const maxMagnitude = useMemo(() => {
    if (topFeatures.length === 0) {
      return 1
    }

    return Math.max(
      ...topFeatures.map((item) =>
        Math.abs(item.shap_value),
      ),
      0.000001,
    )
  }, [topFeatures])

  return (
    <div className="mt-8">
      {/* Model probabilities */}

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard
          label="Prediction"
          value={
            explanation.xai.prediction === 1
              ? 'Malicious'
              : 'Benign'
          }
        />

        <MetricCard
          label="Benign probability"
          value={`${(
            explanation.xai.probability_benign * 100
          ).toFixed(1)}%`}
        />

        <MetricCard
          label="Malicious probability"
          value={`${(
            explanation.xai.probability_malicious * 100
          ).toFixed(1)}%`}
        />
      </div>

      {/* Top feature attribution */}

      <div className="mt-7 rounded-3xl border border-white/70 bg-white/25 p-6">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-5 w-5 text-slate-500" />

          <div>
            <h3 className="font-semibold text-slate-800">
              Top contributing features
            </h3>

            <p className="mt-1 text-xs text-slate-400">
              Ranked by absolute SHAP value.
            </p>
          </div>
        </div>

        <div className="mt-6 space-y-5">
          {topFeatures.map((feature) => (
            <SHAPBar
              key={feature.feature}
              feature={feature}
              maxMagnitude={maxMagnitude}
            />
          ))}
        </div>
      </div>

      {/* All features */}

      <div className="mt-7 rounded-3xl border border-white/70 bg-white/25 p-6">
        <div>
          <h3 className="font-semibold text-slate-800">
            All feature contributions
          </h3>

          <p className="mt-1 text-xs leading-5 text-slate-400">
            Complete 15-feature attribution returned by the
            backend XAI module.
          </p>
        </div>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-white/70 text-left">
                <th className="pb-3 pr-4 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
                  Feature
                </th>

                <th className="pb-3 pr-4 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
                  Value
                </th>

                <th className="pb-3 pr-4 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
                  SHAP value
                </th>

                <th className="pb-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
                  Effect
                </th>
              </tr>
            </thead>

            <tbody>
              {explanation.xai.feature_contributions.map(
                (feature) => (
                  <SHAPTableRow
                    key={feature.feature}
                    feature={feature}
                  />
                ),
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

/* ========================================================================== */
/* SHAP bar                                                                   */
/* ========================================================================== */

function SHAPBar({
  feature,
  maxMagnitude,
}: {
  feature: SHAPFeatureContribution
  maxMagnitude: number
}) {
  const magnitude = Math.abs(feature.shap_value)

  const width = Math.max(
    magnitude / maxMagnitude,
    magnitude === 0 ? 0 : 0.04,
  )

  const increasesThreat =
    feature.direction === 'increases_threat'

  const decreasesThreat =
    feature.direction === 'decreases_threat'

  return (
    <div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-medium text-slate-800">
            {formatFeatureName(feature.feature)}
          </p>

          <p className="mt-0.5 text-xs text-slate-400">
            Input value: {formatFeatureValue(feature.value)}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-700">
            {feature.shap_value >= 0 ? '+' : ''}
            {feature.shap_value.toFixed(4)}
          </span>

          <span className="text-xs text-slate-400">
            {increasesThreat
              ? 'increases threat'
              : decreasesThreat
                ? 'decreases threat'
                : 'neutral'}
          </span>
        </div>
      </div>

      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200/60">
        <div
          className="h-full rounded-full bg-slate-700 transition-all duration-700"
          style={{
            width: `${width * 100}%`,
            opacity: increasesThreat
              ? 1
              : decreasesThreat
                ? 0.4
                : 0.2,
          }}
        />
      </div>
    </div>
  )
}

/* ========================================================================== */
/* SHAP table                                                                 */
/* ========================================================================== */

function SHAPTableRow({
  feature,
}: {
  feature: SHAPFeatureContribution
}) {
  return (
    <tr className="border-b border-white/50 last:border-0">
      <td className="py-4 pr-4 font-medium text-slate-700">
        {formatFeatureName(feature.feature)}
      </td>

      <td className="py-4 pr-4 text-slate-500">
        {formatFeatureValue(feature.value)}
      </td>

      <td className="py-4 pr-4 font-mono text-xs text-slate-600">
        {feature.shap_value >= 0 ? '+' : ''}
        {feature.shap_value.toFixed(6)}
      </td>

      <td className="py-4">
        <span className="rounded-full border border-white/70 bg-white/50 px-3 py-1 text-xs text-slate-600">
          {feature.direction === 'increases_threat'
            ? 'Increases threat'
            : feature.direction === 'decreases_threat'
              ? 'Decreases threat'
              : 'Neutral'}
        </span>
      </td>
    </tr>
  )
}

/* ========================================================================== */
/* General components                                                         */
/* ========================================================================== */

function SignalCard({
  icon,
  title,
  score,
  description,
}: {
  icon: React.ReactNode
  title: string
  score: number
  description: string
}) {
  return (
    <GlassPanel className="p-7">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/60 text-slate-600">
        {icon}
      </div>

      <p className="mt-6 text-sm font-semibold text-slate-800">
        {title}
      </p>

      <p className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
        {(score * 100).toFixed(1)}%
      </p>

      <p className="mt-3 text-sm leading-6 text-slate-400">
        {description}
      </p>
    </GlassPanel>
  )
}

function FusionValue({
  label,
  value,
  weight,
}: {
  label: string
  value: number
  weight?: number
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold text-slate-800">
        {(value * 100).toFixed(1)}%
      </p>

      <p className="mt-2 text-xs text-slate-400">
        Weight:{' '}
        {typeof weight === 'number'
          ? `${(weight * 100).toFixed(0)}%`
          : 'Unavailable'}
      </p>
    </div>
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

      <p className="mt-2 text-lg font-semibold text-slate-800">
        {value ? 'Threat' : 'Non-threat'}
      </p>
    </div>
  )
}

function EvidenceValue({
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

function MetricCard({
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

      <p className="mt-2 text-xl font-semibold tracking-tight text-slate-800">
        {value}
      </p>
    </div>
  )
}

/* ========================================================================== */
/* Formatting                                                                 */
/* ========================================================================== */

function formatFeatureName(feature: string): string {
  return feature
    .split('_')
    .map(
      (part) =>
        part.charAt(0).toUpperCase() + part.slice(1),
    )
    .join(' ')
}

function formatFeatureValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value)
  }

  return value.toFixed(4)
}

export default Explainability
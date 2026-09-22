export interface AnalyzeRequest {
  text: string
}

export interface AnalyzeResponse {
  text: string
  prediction_id?: number

  ml: MLResult
  rules: RuleResult
  semantic: SemanticResult
  semantic_evidence?: SemanticEvidence

  fusion: FusionResult

  threat_score: number
  decision: Decision
  severity: Severity

  detector_votes: DetectorVotes

  hypothesis?: string
  description?: string
  confidence?: number
  recommended_action?: string
  hypotheses?: HypothesisResult[]
  evidence?: Array<EvidenceItem | string>
  explainability?: UnifiedExplainability
}

export type Decision = 'ALLOW' | 'REVIEW' | 'BLOCK'
export type Severity = 'LOW' | 'MEDIUM' | 'HIGH'

export interface MLResult {
  label: number
  probability: number
}

export interface MatchedRule {
  rule: string
  category: string
  weight: number
  matched_patterns: string[]
}

export interface RuleResult {
  score: number
  is_threat: boolean
  categories: string[]
  matched_rules: MatchedRule[]
}

export interface SemanticNeighbor {
  index?: number
  similarity?: number
  label?: number
  text?: string
  [key: string]: unknown
}

export interface SemanticResult {
  similarity: number
  score: number
  threshold: number
  is_threat: boolean
  matched_label: number
  matched_text: string
  is_malicious_match: boolean
  threat_score: number

  malicious_neighbor_count?: number
  strong_malicious_neighbor_count?: number
  high_confidence?: boolean
  top_neighbors?: SemanticNeighbor[]

  [key: string]: unknown
}

export interface SemanticEvidence {
  similarity: number
  matched_label: number
  malicious_match: boolean
  threat_score: number
  malicious_neighbor_count: number
  strong_malicious_neighbor_count: number
  high_confidence: boolean
}

export interface FusionResult {
  threat_score: number
  decision: Decision
  weights?: FusionWeights
  [key: string]: unknown
}

export interface FusionWeights {
  ml?: number
  rules?: number
  semantic?: number
}

export interface DetectorVotes {
  ml: boolean
  rules: boolean
  semantic: boolean
  total: number
}

export interface HypothesisResult {
  hypothesis?: string
  description?: string
  confidence?: number
  recommended_action?: string
}

export interface EvidenceItem {
  type?: string
  title?: string
  description?: string
  value?: string | number | boolean
}

export interface FeedbackRequest {
  prediction_id: number
  human_label: number
  attack_category?: string | null
}

export interface FeedbackResponse {
  status?: string
  prediction_id?: number
  human_label?: number
  attack_category?: string | null
  message?: string
}

export interface HealthResponse {
  status: string
  [key: string]: unknown
}

export interface APIInfoResponse {
  system?: string
  status?: string
  service?: string
  [key: string]: unknown
}

/* -------------------------------------------------------------------------- */
/* SHAP / Explainability                                                      */
/* -------------------------------------------------------------------------- */

export type SHAPDirection =
  | 'increases_threat'
  | 'decreases_threat'
  | 'neutral'

export interface SHAPFeatureContribution {
  feature: string
  value: number
  shap_value: number
  direction: SHAPDirection
}

export interface SHAPExplanation {
  prediction: number
  probability_benign: number
  probability_malicious: number
  feature_contributions: SHAPFeatureContribution[]
  top_features: SHAPFeatureContribution[]
}

export interface ExplainResponse {
  status: string
  prompt: string
  feature_count: number
  embedding_dimension: number
  xai: SHAPExplanation
  logistic_regression?: LogisticExplanation
}

export interface UnifiedExplainability {
  status: 'success' | 'partial' | 'failed' | string
  random_forest?: SHAPExplanation | null
  logistic_regression?: LogisticExplanation | null
  rules?: RuleResult
  semantic?: SemanticResult
  threat_fusion?: FusionResult
  hypothesis?: HypothesisResult | null
  errors?: Array<{ source?: string; message?: string }>
}

export interface LogisticContribution {
  feature: string
  value: number
  scaled_value: number
  coefficient: number
  contribution: number
  direction: SHAPDirection
}

export interface LogisticExplanation {
  prediction: number
  probability_benign: number
  probability_malicious: number
  intercept: number
  feature_contributions: LogisticContribution[]
  top_features: LogisticContribution[]
}

/* -------------------------------------------------------------------------- */
/* Document scanning                                                          */
/* -------------------------------------------------------------------------- */

export interface DocumentChunkResult {
  chunk_index?: number
  page_number?: number
  page_chunk_index?: number
  text?: string
  full_result?: AnalyzeResponse

  threat_score?: number
  decision?: Decision
  severity?: Severity

  [key: string]: unknown
}

export interface DocumentScanResult {
  file_name: string
  file_type: string
  success: boolean

  page_count: number
  extracted_pages: number
  character_count: number
  chunk_count: number

  document_decision: Decision
  document_severity: Severity

  threat_chunks: number

  results: DocumentChunkResult[]
  document_explanation?: DocumentExplanation
}

export interface DocumentExplanation {
  primary_hypothesis: string
  decision: Decision
  supporting_evidence: string[]
  threat_chunk_count: number
}

export interface DocumentScanResponse {
  status: string
  document_count: number
  successful_documents: number
  failed_documents: number
  overall_decision: Decision
  documents: DocumentScanItem[]
}

export interface DocumentScanItem {
  filename: string
  file_type?: string
  file_size_bytes?: number
  status: 'success' | 'error'
  result?: DocumentScanResult
  error?: string
}

export interface DocumentHistoryEntry {
  record_type: 'document'
  history_id: string
  scanned_at: string
  filename: string
  file_type?: string
  file_size_bytes?: number
  status: 'success' | 'error'
  result?: DocumentScanResult
  error?: string
}

export interface APIError {
  message: string
  status?: number
}
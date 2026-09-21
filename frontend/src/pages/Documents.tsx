import { useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  FileText,
  LoaderCircle,
  ShieldAlert,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import GlassPanel from '../components/GlassPanel'
import {
  scanDocument,
  type DocumentScanResponse,
} from '../api/athsApi'
import type { DocumentScanResult } from '../api/types'
import type {
  DocumentChunkResult,
  DocumentHistoryEntry,
} from '../api/types'

const MAX_FILE_SIZE = 20 * 1024 * 1024
const ACCEPTED_TYPES = ['.pdf', '.docx', '.txt']
const HISTORY_KEY = 'promptDetectionHistory'
const MAX_HISTORY_ITEMS = 50

function Documents() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isScanning, setIsScanning] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] =
    useState<DocumentScanResponse | null>(null)

  const validateFile = (selectedFile: File) => {
    const extension =
      `.${selectedFile.name.split('.').pop()?.toLowerCase() || ''}`

    if (!ACCEPTED_TYPES.includes(extension)) {
      setError('Please select a PDF, DOCX, or TXT file.')
      return false
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('The selected file must be smaller than 20 MB.')
      return false
    }

    return true
  }

  const selectFiles = (selectedFiles: File[]) => {
    setError('')
    setMessage('')
    setResult(null)

    const validFiles = selectedFiles.filter(validateFile)

    if (validFiles.length !== selectedFiles.length) {
      setError('Some files were skipped. Select only PDF, DOCX, or TXT files smaller than 20 MB.')
    }

    if (validFiles.length === 0) {
      setFiles([])
      return
    }

    setFiles(validFiles)
  }

  const handleFileInput = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const selectedFiles = Array.from(event.target.files ?? [])

    if (selectedFiles.length > 0) {
      selectFiles(selectedFiles)
    }

    event.target.value = ''
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)

    const droppedFiles = Array.from(event.dataTransfer.files ?? [])

    if (droppedFiles.length > 0) {
      selectFiles(droppedFiles)
    }
  }

  const handleRemove = () => {
    setFiles([])
    setError('')
    setMessage('')
    setResult(null)
  }

  const handleScan = async () => {
    if (files.length === 0) {
      setError('Choose a document before scanning.')
      return
    }

    setError('')
    setMessage('')
    setResult(null)
    setIsScanning(true)

    try {
      const scanResult = await scanDocument(files)

      setResult(scanResult)
      saveDocumentsToHistory(scanResult.documents)
      setMessage(`${scanResult.successful_documents} document(s) analyzed.`)
    } catch (error) {
      const errorMessage =
        typeof error === 'object' &&
        error !== null &&
        'message' in error &&
        typeof error.message === 'string'
          ? error.message
          : 'Document scanning failed.'

      setError(errorMessage)
    } finally {
      setIsScanning(false)
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
      <div className="mb-10">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Document intelligence
        </p>

        <h1 className="mt-3 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
          Scan a document.
        </h1>

        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-500">
          Upload a supported document for extraction and document-level
          threat analysis.
        </p>
      </div>

      <GlassPanel className="p-6 sm:p-10">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={handleFileInput}
          className="hidden"
        />

        {files.length === 0 ? (
          <div
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`flex min-h-[360px] flex-col items-center justify-center rounded-[1.75rem] border border-dashed px-6 text-center transition ${
              isDragging
                ? 'border-slate-500 bg-white/50'
                : 'border-slate-300/70 bg-white/25'
            }`}
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-3xl border border-white/80 bg-white/60 shadow-sm">
              <UploadCloud className="h-7 w-7 text-slate-600" />
            </div>

            <h2 className="mt-6 text-xl font-semibold">
              Drop your document here
            </h2>

            <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
              Supported formats: PDF, DOCX, and TXT.
              <br />
              Maximum file size: 20 MB.
            </p>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="mt-6 rounded-full bg-[#111318] px-6 py-3 text-sm font-medium text-white transition hover:-translate-y-0.5"
            >
              Choose Files
            </button>
          </div>
        ) : (
          <div className="rounded-[1.75rem] border border-white/70 bg-white/35 p-6 backdrop-blur-xl">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex min-w-0 items-start gap-4">
                <FileText className="mt-1 h-6 w-6 shrink-0 text-slate-600" />

                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-800">
                    {files.length} file(s) selected
                  </p>
                  <div className="mt-2 space-y-1 text-xs text-slate-400">
                    {files.map((selectedFile) => (
                      <p key={`${selectedFile.name}-${selectedFile.lastModified}`} className="truncate">
                        {selectedFile.name} ({formatFileSize(selectedFile.size)})
                      </p>
                    ))}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={handleRemove}
                disabled={isScanning}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/80 bg-white/50 px-4 py-2.5 text-xs font-medium text-slate-600 transition hover:bg-white/75 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                Remove
              </button>
            </div>

            <div className="mt-7 flex flex-col gap-3 sm:flex-row">
              <button
                type="button"
                onClick={handleScan}
                disabled={isScanning}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-[#111318] px-6 py-3 text-sm font-medium text-white transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isScanning ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <FileSearch className="h-4 w-4" />
                    Scan Document
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isScanning}
                className="rounded-full border border-white/80 bg-white/45 px-6 py-3 text-sm font-medium text-slate-700 transition hover:bg-white/70 disabled:opacity-50"
              >
                Choose More Files
              </button>
            </div>
          </div>
        )}

        {message && (
          <div className="mt-5 flex items-center gap-3 rounded-2xl border border-white/70 bg-white/40 p-4 text-sm text-slate-600">
            <CheckCircle2 className="h-5 w-5 shrink-0" />
            {message}
          </div>
        )}

        {error && (
          <div className="mt-5 flex items-start gap-3 rounded-2xl border border-slate-200 bg-white/50 p-4 text-sm text-slate-600">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </GlassPanel>

      {result && (
        <div className="mt-5 space-y-5">
          {result.documents.map((document) => (
            document.result ? (
              <DocumentResults key={document.filename} result={document.result} />
            ) : (
              <GlassPanel key={document.filename} className="p-6 text-sm text-red-700">
                {document.filename}: {document.error || 'Document scanning failed.'}
              </GlassPanel>
            )
          ))}
        </div>
      )}
    </div>
  )
}

function saveDocumentsToHistory(
  documents: DocumentHistoryEntry[] | DocumentScanResponse['documents'],
) {
  try {
    const stored = localStorage.getItem(HISTORY_KEY)
    const history = stored ? (JSON.parse(stored) as unknown[]) : []
    const scannedAt = new Date().toISOString()
    const entries: DocumentHistoryEntry[] = documents.map(
      (document, index) => ({
        ...document,
        record_type: 'document',
        history_id: `${scannedAt}-${index}-${document.filename}`,
        scanned_at: scannedAt,
      }),
    )

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify([...entries, ...history].slice(0, MAX_HISTORY_ITEMS)),
    )
  } catch {
    // History storage should never prevent document scanning from succeeding.
  }
}

function DocumentResults({
  result,
}: {
  result: DocumentScanResult
}) {
  const navigate = useNavigate()
  const decision = result.document_decision
  const severity = result.document_severity

  const isBlocked = decision === 'BLOCK'
  const isReview = decision === 'REVIEW'
  const explainableChunk = getMostRelevantChunk(result.results)

  const openExplainability = () => {
    const analysis = getChunkAnalysis(explainableChunk)

    if (!analysis) {
      return
    }

    sessionStorage.setItem(
      'latestAnalysis',
      JSON.stringify(analysis),
    )

    navigate('/explainability')
  }

  return (
    <div className="mt-5 space-y-5">
      <GlassPanel className="p-6 sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              Document threat assessment
            </p>

            <h2 className="mt-2 break-words text-2xl font-semibold tracking-tight">
              {result.file_name}
            </h2>

            <p className="mt-2 text-sm text-slate-500">
              {result.file_type.toUpperCase()} document analysis
            </p>
          </div>

          <div
            className={`inline-flex items-center gap-2 self-start rounded-full border px-5 py-2.5 text-sm font-semibold ${
              isBlocked
                ? 'border-red-200 bg-red-50 text-red-700'
                : isReview
                  ? 'border-amber-200 bg-amber-50 text-amber-700'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700'
            }`}
          >
            <ShieldAlert className="h-4 w-4" />
            {decision}
          </div>
        </div>

        {explainableChunk && getChunkAnalysis(explainableChunk) && (
          <button
            type="button"
            onClick={openExplainability}
            className="mt-6 inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/55 px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-white/75"
          >
            <ShieldAlert className="h-4 w-4" />
            View Explainability
          </button>
        )}

        <div className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric
            label="Severity"
            value={severity}
          />

          <Metric
            label="Pages"
            value={result.page_count}
          />

          <Metric
            label="Chunks analyzed"
            value={result.chunk_count}
          />

          <Metric
            label="Threat chunks"
            value={result.threat_chunks}
          />
        </div>
      </GlassPanel>

      <GlassPanel className="p-6 sm:p-8">
        <div className="flex items-center gap-3">
          <FileSearch className="h-5 w-5 text-slate-500" />

          <div>
            <h2 className="font-semibold">
              Extraction summary
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Information returned directly by the document scanner.
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <InfoItem
            label="File"
            value={result.file_name}
          />

          <InfoItem
            label="Type"
            value={result.file_type.toUpperCase()}
          />

          <InfoItem
            label="Characters extracted"
            value={result.character_count.toLocaleString()}
          />

          <InfoItem
            label="Pages extracted"
            value={result.extracted_pages}
          />

          <InfoItem
            label="Chunks"
            value={result.chunk_count}
          />

          <InfoItem
            label="Threat chunks"
            value={result.threat_chunks}
          />
        </div>
      </GlassPanel>

      {result.threat_chunks > 0 && (
        <GlassPanel className="p-6 sm:p-8">
          <div className="flex items-center gap-3">
            <ShieldAlert className="h-5 w-5 text-slate-500" />

            <div>
              <h2 className="font-semibold">
                Threatening document chunks
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                These results come from the backend chunk-level
                analysis.
              </p>
            </div>
          </div>

          <div className="mt-6 space-y-4">
            {getThreatResults(result.results).map(
              (chunk, index) => (
                <ThreatChunk
                  key={index}
                  chunk={chunk}
                  index={index}
                />
              ),
            )}
          </div>
        </GlassPanel>
      )}

      {result.threat_chunks === 0 && (
        <GlassPanel className="p-6 sm:p-8">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-slate-500" />

            <div>
              <h2 className="font-semibold">
                No threatening chunks detected
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                The document scanner did not report any threatening
                chunks.
              </p>
            </div>
          </div>

          {explainableChunk && getChunkAnalysis(explainableChunk) && (
            <button
              type="button"
              onClick={openExplainability}
              className="mt-5 inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/55 px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-white/75"
            >
              <ShieldAlert className="h-4 w-4" />
              Explain the document result
            </button>
          )}
        </GlassPanel>
      )}
    </div>
  )
}

function getMostRelevantChunk(
  results: DocumentScanResult['results'],
): Record<string, unknown> | null {
  const chunks = results.filter(
    (item): item is Record<string, unknown> =>
      typeof item === 'object' &&
      item !== null &&
      !Array.isArray(item),
  )

  return (
    [...chunks].sort(
      (left, right) =>
        (getNumber(right, 'threat_score') ?? 0) -
        (getNumber(left, 'threat_score') ?? 0),
    )[0] ?? null
  )
}

function getChunkAnalysis(
  chunk: Record<string, unknown> | null,
) {
  const analysis = chunk?.full_result

  return isAnalyzeResponse(analysis) ? analysis : null
}

function isAnalyzeResponse(
  value: unknown,
): value is NonNullable<DocumentChunkResult['full_result']> {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const analysis = value as Record<string, unknown>

  return (
    typeof analysis.text === 'string' &&
    typeof analysis.threat_score === 'number' &&
    typeof analysis.decision === 'string' &&
    typeof analysis.ml === 'object' &&
    typeof analysis.rules === 'object' &&
    typeof analysis.semantic === 'object' &&
    typeof analysis.fusion === 'object'
  )
}

function Metric({
  label,
  value,
}: {
  label: string
  value: string | number
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
      <p className="text-xs uppercase tracking-[0.12em] text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-xl font-semibold text-slate-800">
        {value}
      </p>
    </div>
  )
}

function InfoItem({
  label,
  value,
}: {
  label: string
  value: string | number
}) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/30 p-4">
      <p className="text-xs text-slate-400">
        {label}
      </p>

      <p className="mt-1 break-words text-sm font-medium text-slate-700">
        {value}
      </p>
    </div>
  )
}

function ThreatChunk({
  chunk,
  index,
}: {
  chunk: Record<string, unknown>
  index: number
}) {
  const decision = getString(chunk, 'decision')
  const severity = getString(chunk, 'severity')
  const threatScore = getNumber(chunk, 'threat_score')
  const pageNumber = getNumber(chunk, 'page_number')
  const pageChunkIndex = getNumber(chunk, 'page_chunk_index')

  const text =
    getString(chunk, 'text') ||
    getString(chunk, 'chunk_text') ||
    getString(chunk, 'content')

  const categories = getStringArray(
    chunk,
    'categories',
  )

  return (
    <div className="rounded-2xl border border-white/70 bg-white/35 p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.12em] text-slate-400">
            Threat chunk {index + 1}
          </p>

          <div className="mt-2 flex flex-wrap gap-2">
            {typeof pageNumber === 'number' && (
              <span className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                Page {pageNumber}
                {typeof pageChunkIndex === 'number'
                  ? `, chunk ${pageChunkIndex}`
                  : ''}
              </span>
            )}

            {decision && (
              <span className="rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs font-semibold text-red-700">
                {decision}
              </span>
            )}

            {severity && (
              <span className="rounded-full border border-white/80 bg-white/60 px-3 py-1 text-xs font-medium text-slate-600">
                {severity}
              </span>
            )}

            {typeof threatScore === 'number' && (
              <span className="rounded-full border border-white/80 bg-white/60 px-3 py-1 text-xs font-medium text-slate-600">
                Score {threatScore.toFixed(3)}
              </span>
            )}
          </div>
        </div>
      </div>

      {categories.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {categories.map((category) => (
            <span
              key={category}
              className="rounded-full bg-slate-100/70 px-3 py-1 text-xs text-slate-600"
            >
              {category}
            </span>
          ))}
        </div>
      )}

      {text && (
        <div className="mt-4 rounded-xl border border-white/70 bg-white/45 p-4">
          <p className="whitespace-pre-wrap text-sm leading-6 text-slate-600">
            {text}
          </p>
        </div>
      )}
    </div>
  )
}

function getThreatResults(
  results: unknown[],
): Record<string, unknown>[] {
  return results.filter((item): item is Record<string, unknown> => {
    if (
      typeof item !== 'object' ||
      item === null ||
      Array.isArray(item)
    ) {
      return false
    }

    const record = item as Record<string, unknown>

    const decision = getString(record, 'decision')

    return (
      decision === 'BLOCK' ||
      decision === 'REVIEW' ||
      getBoolean(record, 'is_threat') === true ||
      getBoolean(record, 'threat') === true
    )
  })
}

function getString(
  object: Record<string, unknown>,
  key: string,
): string | null {
  const value = object[key]

  return typeof value === 'string' ? value : null
}

function getNumber(
  object: Record<string, unknown>,
  key: string,
): number | null {
  const value = object[key]

  return typeof value === 'number' ? value : null
}

function getBoolean(
  object: Record<string, unknown>,
  key: string,
): boolean {
  return object[key] === true
}

function getStringArray(
  object: Record<string, unknown>,
  key: string,
): string[] {
  const value = object[key]

  if (!Array.isArray(value)) {
    return []
  }

  return value.filter(
    (item): item is string => typeof item === 'string',
  )
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default Documents
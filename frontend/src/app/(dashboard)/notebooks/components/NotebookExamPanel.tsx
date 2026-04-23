'use client'

import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { chatApi } from '@/lib/api/chat'
import { useTranslation } from '@/lib/hooks/use-translation'
import { SourceListResponse, NoteResponse } from '@/lib/types/api'
import { getApiErrorMessage } from '@/lib/utils/error-handler'
import { ContextSelections } from '../[id]/page'

interface NotebookExamPanelProps {
  notebookId: string
  sources: SourceListResponse[]
  notes: NoteResponse[]
  contextSelections: ContextSelections
}

export function NotebookExamPanel({
  notebookId,
  sources,
  notes,
  contextSelections
}: NotebookExamPanelProps) {
  const { t } = useTranslation()
  const [prompt, setPrompt] = useState('')
  const [exam, setExam] = useState('')
  const [submission, setSubmission] = useState('')
  const [gradingResult, setGradingResult] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isGrading, setIsGrading] = useState(false)

  const contextConfig = useMemo(() => {
    const config: { sources: Record<string, string>, notes: Record<string, string> } = {
      sources: {},
      notes: {}
    }

    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') {
        config.sources[source.id] = 'insights'
      } else if (mode === 'full') {
        config.sources[source.id] = 'full content'
      } else {
        config.sources[source.id] = 'not in'
      }
    })

    notes.forEach(note => {
      const mode = contextSelections.notes[note.id]
      config.notes[note.id] = mode === 'full' ? 'full content' : 'not in'
    })

    return config
  }, [sources, notes, contextSelections])

  const handleGenerateExam = async () => {
    if (!prompt.trim()) {
      toast.error(t('exam.promptRequired'))
      return
    }

    setIsGenerating(true)
    setGradingResult('')

    try {
      const response = await chatApi.generateExam({
        notebook_id: notebookId,
        prompt,
        context_config: contextConfig
      })
      setExam(response.exam)
      toast.success(t('exam.generated'))
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(
        getApiErrorMessage(
          error.response?.data?.detail || error.message,
          (key) => t(key),
          'exam.generateFailed'
        )
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const handleGradeExam = async () => {
    if (!prompt.trim() || !exam.trim() || !submission.trim()) {
      toast.error(t('exam.missingForGrading'))
      return
    }

    setIsGrading(true)

    try {
      const response = await chatApi.gradeExam({
        notebook_id: notebookId,
        prompt,
        exam,
        submission,
        context_config: contextConfig
      })
      setGradingResult(response.result)
      toast.success(t('exam.graded'))
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(
        getApiErrorMessage(
          error.response?.data?.detail || error.message,
          (key) => t(key),
          'exam.gradeFailed'
        )
      )
    } finally {
      setIsGrading(false)
    }
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle>{t('exam.title')}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 space-y-4 overflow-auto">
        <div className="space-y-2">
          <Label htmlFor="exam-prompt">{t('exam.promptLabel')}</Label>
          <Textarea
            id="exam-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={t('exam.promptPlaceholder')}
            rows={3}
          />
          <Button onClick={handleGenerateExam} disabled={isGenerating}>
            {isGenerating && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            {t('exam.generate')}
          </Button>
        </div>

        <div className="space-y-2">
          <Label htmlFor="exam-content">{t('exam.generatedExam')}</Label>
          <Textarea
            id="exam-content"
            value={exam}
            onChange={(e) => setExam(e.target.value)}
            placeholder={t('exam.examPlaceholder')}
            rows={10}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="exam-submission">{t('exam.submissionLabel')}</Label>
          <Textarea
            id="exam-submission"
            value={submission}
            onChange={(e) => setSubmission(e.target.value)}
            placeholder={t('exam.submissionPlaceholder')}
            rows={8}
          />
          <Button onClick={handleGradeExam} disabled={isGrading || !exam.trim()}>
            {isGrading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            {t('exam.grade')}
          </Button>
        </div>

        {gradingResult && (
          <div className="space-y-2">
            <Label>{t('exam.gradingResult')}</Label>
            <div className="rounded-md border p-3 prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {gradingResult}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

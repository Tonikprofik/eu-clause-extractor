"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
    FileText,
    Sparkles,
    Zap,
    ChevronDown,
    ChevronUp,
    Loader2,
    Scale,
    Cpu,
    Database,
    ArrowRight,
    CheckCircle2,
    AlertCircle,
    Github,
    ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";

type ModelsResponse = {
    reader_models: string[];
    embedding_models: string[];
    all_models: string[];
    defaults: { reader_model: string; embedding_model: string };
};

type PredictedAnnotation = { clause_type: string; clause_text: string };

type ExtractionResponse = {
    predicted_annotations: PredictedAnnotation[];
    retrieved_chunks: string[];
    reader_llm_output_raw?: string | null;
    trace_id?: string | null;
    timings?: Record<string, number | null>;
    usage?: Record<string, number | null>;
    model_info?: Record<string, unknown>;
    error?: { message: string; stage?: string };
};

const CLAUSE_TYPES = [
    "Subject Matter & Scope",
    "Definitions",
    "Obligations of Member States",
    "Penalties",
    "Entry into Force & Application",
];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

const SAMPLE_TEXT = `Article 1
Subject matter and scope
This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data.

Article 2
Definitions
(1) 'personal data' means any information relating to an identified or identifiable natural person ('data subject');
(2) 'processing' means any operation or set of operations which is performed on personal data or on sets of personal data.

Article 3
Penalties
Member States shall lay down the rules on penalties applicable to infringements of this Regulation and shall take all measures necessary to ensure that they are implemented.`;

export default function Home() {
    const [documentId, setDocumentId] = useState("demo-doc-1");
    const [documentText, setDocumentText] = useState(SAMPLE_TEXT);
    const [userQuery, setUserQuery] = useState("Extract definitions, obligations, and penalties.");
    const [selectedClauseTypes, setSelectedClauseTypes] = useState<string[]>(CLAUSE_TYPES);
    const [language, setLanguage] = useState("en");
    const [useHyde, setUseHyde] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const [models, setModels] = useState<ModelsResponse | null>(null);
    const [readerModel, setReaderModel] = useState("claude-3-7");
    const [embeddingModel, setEmbeddingModel] = useState("text-embedding-3-small");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<ExtractionResponse | null>(null);

    useEffect(() => {
        const controller = new AbortController();
        const loadModels = async () => {
            try {
                const res = await fetch(`${API_BASE}/api/v1/models`, {
                    signal: controller.signal,
                });
                if (!res.ok) throw new Error(`Failed to fetch models (${res.status})`);
                const data = (await res.json()) as ModelsResponse;
                setModels(data);
                if (data.defaults?.reader_model) setReaderModel(data.defaults.reader_model);
                if (data.defaults?.embedding_model) setEmbeddingModel(data.defaults.embedding_model);
            } catch (e) {
                if (e instanceof Error && e.name === "AbortError") return;
                console.warn("Models fetch failed:", e);
            }
        };
        loadModels();
        return () => controller.abort();
    }, []);

    const groupedByType = useMemo(() => {
        if (!result?.predicted_annotations) return {} as Record<string, PredictedAnnotation[]>;
        return result.predicted_annotations.reduce(
            (acc: Record<string, PredictedAnnotation[]>, ann: PredictedAnnotation) => {
                (acc[ann.clause_type] = acc[ann.clause_type] || []).push(ann);
                return acc;
            },
            {}
        );
    }, [result]);

    const toggleClauseType = useCallback((ct: string) => {
        setSelectedClauseTypes((prev) =>
            prev.includes(ct) ? prev.filter((x) => x !== ct) : [...prev, ct]
        );
    }, []);

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const res = await fetch(`${API_BASE}/api/v1/extract-clauses`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    document_id: documentId,
                    document_text: documentText,
                    user_query: userQuery,
                    clause_types: selectedClauseTypes,
                    language,
                    options: {
                        use_hyde: useHyde,
                        top_k: 5,
                        reader_model: readerModel,
                        embedding_model: embeddingModel,
                    },
                }),
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || `Request failed (${res.status})`);
            }

            const data = (await res.json()) as ExtractionResponse;
            setResult(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Extraction failed");
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="min-h-screen">
            {/* Hero Section */}
            <div className="relative overflow-hidden border-b border-border/50 bg-gradient-to-b from-primary/5 via-background to-background">
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent" />
                <div className="relative mx-auto max-w-6xl px-6 py-12 md:py-16">
                    <div className="flex flex-col items-center text-center">
                        <div className="mb-4 flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-1.5 text-sm text-primary">
                            <Scale className="h-4 w-4" />
                            <span>MSc Thesis Project — LLMOps Demo</span>
                        </div>
                        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
                            EU Clause <span className="gradient-text">Extractor</span>
                        </h1>
                        <p className="mb-6 max-w-2xl text-lg text-muted-foreground">
                            Extract legal clauses from EU regulations using RAG-powered LLM pipelines.
                            Powered by ChromaDB for semantic search, LiteLLM for model routing,
                            and Langfuse for observability.
                        </p>
                        <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
                            <div className="flex items-center gap-2">
                                <Database className="h-4 w-4 text-primary" />
                                <span>ChromaDB</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Cpu className="h-4 w-4 text-accent" />
                                <span>LiteLLM</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Sparkles className="h-4 w-4 text-primary" />
                                <span>Langfuse</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="mx-auto max-w-6xl space-y-8 px-6 py-8 md:py-12">
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid gap-6 lg:grid-cols-2">
                        {/* Left Panel - Document Input */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <FileText className="h-4 w-4" />
                                    Document Input
                                </CardTitle>
                                <CardDescription>Paste regulation text to extract clauses from</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="documentId">Document ID</Label>
                                    <Input
                                        id="documentId"
                                        value={documentId}
                                        onChange={(e) => setDocumentId(e.target.value)}
                                        placeholder="document-id"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="userQuery">Extraction Query</Label>
                                    <Input
                                        id="userQuery"
                                        value={userQuery}
                                        onChange={(e) => setUserQuery(e.target.value)}
                                        placeholder="What clauses to extract?"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="documentText">Document Text</Label>
                                    <Textarea
                                        id="documentText"
                                        value={documentText}
                                        onChange={(e) => setDocumentText(e.target.value)}
                                        placeholder="Paste EU regulation text..."
                                        className="min-h-[200px] font-mono text-sm"
                                    />
                                    <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setDocumentText(SAMPLE_TEXT)}
                                    >
                                        Load Sample
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Right Panel - Settings */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Sparkles className="h-4 w-4" />
                                    Extraction Settings
                                </CardTitle>
                                <CardDescription>Configure models and clause types</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid gap-4 sm:grid-cols-2">
                                    <div className="space-y-2">
                                        <Label htmlFor="language">Language</Label>
                                        <Select
                                            id="language"
                                            value={language}
                                            onChange={(e) => setLanguage(e.target.value)}
                                        >
                                            <option value="en">English</option>
                                            <option value="de">German</option>
                                        </Select>
                                    </div>

                                    <div className="space-y-2">
                                        <Label htmlFor="readerModel">Reader Model</Label>
                                        <Select
                                            id="readerModel"
                                            value={readerModel}
                                            onChange={(e) => setReaderModel(e.target.value)}
                                        >
                                            {(models?.reader_models || [readerModel]).map((m) => (
                                                <option key={m} value={m}>
                                                    {m}
                                                </option>
                                            ))}
                                        </Select>
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <Label>Clause Types</Label>
                                    <div className="space-y-2">
                                        {CLAUSE_TYPES.map((ct) => (
                                            <Checkbox
                                                key={ct}
                                                id={`ct-${ct}`}
                                                checked={selectedClauseTypes.includes(ct)}
                                                onChange={() => toggleClauseType(ct)}
                                                label={ct}
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Advanced Settings Collapsible */}
                                <div className="border-t pt-4">
                                    <button
                                        type="button"
                                        className="flex w-full items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground"
                                        onClick={() => setShowAdvanced(!showAdvanced)}
                                    >
                                        Advanced Settings
                                        {showAdvanced ? (
                                            <ChevronUp className="h-4 w-4" />
                                        ) : (
                                            <ChevronDown className="h-4 w-4" />
                                        )}
                                    </button>

                                    {showAdvanced && (
                                        <div className="mt-4 space-y-4">
                                            <div className="space-y-2">
                                                <Label htmlFor="embeddingModel">Embedding Model</Label>
                                                <Select
                                                    id="embeddingModel"
                                                    value={embeddingModel}
                                                    onChange={(e) => setEmbeddingModel(e.target.value)}
                                                >
                                                    {(models?.embedding_models || [embeddingModel]).map((m) => (
                                                        <option key={m} value={m}>
                                                            {m}
                                                        </option>
                                                    ))}
                                                </Select>
                                            </div>

                                            <Checkbox
                                                id="useHyde"
                                                checked={useHyde}
                                                onChange={(e) => setUseHyde(e.target.checked)}
                                                label="Enable HyDE (Hypothetical Document Embedding)"
                                            />
                                        </div>
                                    )}
                                </div>

                                <Button type="submit" className="w-full bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70" disabled={loading}>
                                    {loading ? (
                                        <>
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            Extracting clauses...
                                        </>
                                    ) : (
                                        <>
                                            <ArrowRight className="mr-2 h-4 w-4" />
                                            Extract Clauses
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>
                    </div>
                </form>

                {/* Error Display */}
                {error && (
                    <Card className="animate-fade-in border-destructive bg-destructive/10">
                        <CardContent className="flex items-center gap-3 py-4">
                            <AlertCircle className="h-5 w-5 text-destructive" />
                            <p className="text-sm text-destructive">{error}</p>
                        </CardContent>
                    </Card>
                )}

                {/* Results */}
                {result && (
                    <div className="animate-slide-up space-y-6">
                        <Card className="glass-card">
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <CardTitle className="flex items-center gap-2">
                                            <CheckCircle2 className="h-5 w-5 text-accent" />
                                            Extracted Clauses
                                        </CardTitle>
                                        <CardDescription>
                                            {result.predicted_annotations?.length || 0} clauses identified
                                        </CardDescription>
                                    </div>
                                    {result.trace_id && (
                                        <Badge variant="outline" className="font-mono text-xs">
                                            Trace: {result.trace_id.slice(0, 8)}...
                                        </Badge>
                                    )}
                                </div>
                            </CardHeader>
                            <CardContent>
                                {Object.keys(groupedByType).length === 0 ? (
                                    <p className="text-sm text-muted-foreground">No clauses extracted</p>
                                ) : (
                                    <div className="space-y-6">
                                        {Object.entries(groupedByType).map(([type, clauses]) => (
                                            <div key={type} className="space-y-3">
                                                <div className="flex items-center gap-2">
                                                    <Badge className="bg-primary/20 text-primary hover:bg-primary/30">
                                                        {type}
                                                    </Badge>
                                                    <span className="text-xs text-muted-foreground">
                                                        {clauses.length} clause{clauses.length !== 1 ? "s" : ""}
                                                    </span>
                                                </div>
                                                <ul className="space-y-2 pl-4">
                                                    {clauses.map((clause, idx) => (
                                                        <li
                                                            key={idx}
                                                            className="rounded-lg border border-border/50 bg-muted/30 p-4 text-sm leading-relaxed"
                                                        >
                                                            {clause.clause_text}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Metadata Footer */}
                        <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-muted-foreground">
                            <div className="flex items-center gap-1.5">
                                <Cpu className="h-3.5 w-3.5" />
                                <span>Reader: <code className="rounded bg-muted px-1.5 py-0.5">{(result.model_info?.reader_model as string) || readerModel}</code></span>
                            </div>
                            <span className="text-border">•</span>
                            <div className="flex items-center gap-1.5">
                                <Database className="h-3.5 w-3.5" />
                                <span>Chunks: <code className="rounded bg-muted px-1.5 py-0.5">{result.retrieved_chunks?.length || 0}</code></span>
                            </div>
                            {result.timings?.total_time && (
                                <>
                                    <span className="text-border">•</span>
                                    <div className="flex items-center gap-1.5">
                                        <Zap className="h-3.5 w-3.5" />
                                        <span>Time: <code className="rounded bg-muted px-1.5 py-0.5">{result.timings.total_time.toFixed(2)}s</code></span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* Footer */}
                <footer className="border-t border-border/50 pt-8 text-center">
                    <p className="text-sm text-muted-foreground">
                        EU Clause Extractor — MSc Thesis Demo
                    </p>
                    <div className="mt-3 flex items-center justify-center gap-4">
                        <a
                            href="https://github.com/Tonikprofik/eu-clause-extractor"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
                        >
                            <Github className="h-3.5 w-3.5" />
                            GitHub
                        </a>
                        <a
                            href="https://langfuse.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
                        >
                            <ExternalLink className="h-3.5 w-3.5" />
                            Langfuse
                        </a>
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground/60">
                        Aalborg University • 2025
                    </p>
                </footer>
            </div>
        </main>
    );
}



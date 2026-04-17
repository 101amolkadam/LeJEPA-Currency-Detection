import { useParams, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useAnalysis } from "@/hooks/useAnalysis";
import { AnalysisTable } from "@/components/AnalysisTable";
import { InteractiveImage } from "@/components/InteractiveImage";

export default function ResultsPage() {
  const { id } = useParams({ strict: false }) as { id: string };
  const [highlightedFeature, setHighlightedFeature] = useState<string | null>(null);

  const { data: analysis, isLoading, error } = useAnalysis(id);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⏳</div>
          <p className="text-gray-600">Loading analysis results...</p>
        </div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">❌</div>
          <p className="text-gray-600">Failed to load results.</p>
          <Link
            to="/"
            className="mt-4 inline-block bg-blue-600 text-white px-6 py-2 rounded-lg"
          >
            Go Home
          </Link>
        </div>
      </div>
    );
  }

  const isReal = analysis.result === "REAL";

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link
            to="/"
            className="text-gray-600 hover:text-gray-900 cursor-pointer"
          >
            ← Back
          </Link>
          <div
            className={`px-6 py-3 rounded-xl text-xl font-bold ${
              isReal ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
            }`}
          >
            {isReal ? "✅ REAL" : "❌ FAKE"} — {(analysis.confidence * 100).toFixed(1)}%
          </div>
          {analysis.currency_denomination && (
            <div className="text-lg font-medium text-gray-700">
              {analysis.currency_denomination} Note
            </div>
          )}
        </div>

        {/* Confidence Bar */}
        <div className="bg-white rounded-lg p-4 mb-6 shadow-sm">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-gray-700">Confidence</span>
            <div className="flex-1 bg-gray-200 rounded-full h-4">
              <div
                className={`h-4 rounded-full transition-all ${
                  isReal
                    ? "bg-gradient-to-r from-green-400 to-green-600"
                    : "bg-gradient-to-r from-red-400 to-red-600"
                }`}
                style={{ width: `${analysis.confidence * 100}%` }}
              />
            </div>
            <span className="text-sm font-bold">{(analysis.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Interactive Hint */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-6 text-center text-sm text-blue-700">
          💡 <strong>Interactive:</strong> Hover over the image regions or table rows to see corresponding highlights. 
          Click any region to inspect the security feature.
        </div>

        {/* Images + Table Layout */}
        <div className="grid lg:grid-cols-2 gap-6 mb-8">
          {/* Left: Interactive Image */}
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <h3 className="font-semibold mb-4 text-gray-700">Annotated Result</h3>
            <InteractiveImage
              analysis={analysis}
              highlightedFeature={highlightedFeature}
              onFeatureHover={setHighlightedFeature}
            />
          </div>

          {/* Right: Interactive Table */}
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <h2 className="text-lg font-bold mb-4 text-gray-700">Detailed Analysis</h2>
            <AnalysisTable
              analysis={analysis}
              highlightedFeature={highlightedFeature}
              onFeatureHover={setHighlightedFeature}
            />
          </div>
        </div>

        {/* Processing Time */}
        <div className="text-center text-sm text-gray-500 mb-6">
          Processing time: {analysis.processing_time_ms}ms
        </div>

        {/* Actions */}
        <div className="flex gap-4 justify-center">
          <Link
            to="/"
            className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700"
          >
            Analyze Another
          </Link>
          <Link
            to="/history"
            className="bg-gray-200 text-gray-700 px-8 py-3 rounded-lg font-medium hover:bg-gray-300"
          >
            View History
          </Link>
        </div>
      </div>
    </div>
  );
}

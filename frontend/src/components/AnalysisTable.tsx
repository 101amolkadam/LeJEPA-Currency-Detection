import type { AnalysisResult } from "@/types";

interface AnalysisTableProps {
  analysis: AnalysisResult;
  highlightedFeature: string | null;
  onFeatureHover: (feature: string | null) => void;
}

const StatusBadge = ({ status }: { status: string }) => {
  const isGood = ["present", "match", "normal", "valid", "correct", "REAL"].includes(status);
  const isUnknown = status === "unknown";
  return (
    <span
      className={`px-2 py-1 rounded-full text-xs font-semibold transition-all ${
        isUnknown
          ? "bg-gray-100 text-gray-600"
          : isGood
          ? "bg-green-100 text-green-700"
          : "bg-red-100 text-red-700"
      }`}
    >
      {isGood ? "✅" : isUnknown ? "❓" : "❌"} {status.toUpperCase()}
    </span>
  );
};

export function AnalysisTable({ analysis, highlightedFeature, onFeatureHover }: AnalysisTableProps) {
  const rows = [
    {
      key: "overall",
      feature: "Overall Result",
      status: analysis.result,
      confidence: analysis.confidence,
      details: `Ensemble score: ${analysis.ensemble_score.toFixed(4)}`,
    },
    {
      key: "cnn_classification",
      feature: "CNN Classification",
      status: analysis.analysis.cnn_classification.result,
      confidence: analysis.analysis.cnn_classification.confidence,
      details: `Model: ${analysis.analysis.cnn_classification.model} (${analysis.analysis.cnn_classification.processing_time_ms}ms)`,
    },
    {
      key: "watermark",
      feature: "Watermark",
      status: analysis.analysis.watermark.status,
      confidence: analysis.analysis.watermark.confidence,
      details: analysis.analysis.watermark.ssim_score
        ? `SSIM: ${analysis.analysis.watermark.ssim_score.toFixed(2)}`
        : "Template matching",
    },
    {
      key: "security_thread",
      feature: "Security Thread",
      status: analysis.analysis.security_thread.status,
      confidence: analysis.analysis.security_thread.confidence,
      details: analysis.analysis.security_thread.position
        ? `Position: ${analysis.analysis.security_thread.position}`
        : "Detection pending",
    },
    {
      key: "color_analysis",
      feature: "Color Analysis",
      status: analysis.analysis.color_analysis.status,
      confidence: analysis.analysis.color_analysis.confidence,
      details: analysis.analysis.color_analysis.bhattacharyya_distance
        ? `Distance: ${analysis.analysis.color_analysis.bhattacharyya_distance.toFixed(4)}`
        : "Color profile check",
    },
    {
      key: "texture_analysis",
      feature: "Texture Quality",
      status: analysis.analysis.texture_analysis.status,
      confidence: analysis.analysis.texture_analysis.confidence,
      details: `Sharpness: ${(analysis.analysis.texture_analysis.sharpness_score || 0).toFixed(2)}`,
    },
    {
      key: "serial_number",
      feature: "Serial Number",
      status: analysis.analysis.serial_number.status,
      confidence: analysis.analysis.serial_number.confidence,
      details: analysis.analysis.serial_number.extracted_text
        ? `Extracted: "${analysis.analysis.serial_number.extracted_text}"`
        : "Not detected",
    },
    {
      key: "dimensions",
      feature: "Note Dimensions",
      status: analysis.analysis.dimensions.status,
      confidence: analysis.analysis.dimensions.confidence,
      details: analysis.analysis.dimensions.aspect_ratio
        ? `Ratio: ${analysis.analysis.dimensions.aspect_ratio} (expected: ${analysis.analysis.dimensions.expected_aspect_ratio})`
        : "Aspect ratio check",
    },
  ];

  const getHighlightStyle = (key: string) => {
    if (highlightedFeature !== key) return {};
    return {
      backgroundColor: "#eff6ff",
      borderLeft: "3px solid #3b82f6",
      boxShadow: "0 0 8px rgba(59, 130, 246, 0.15)",
    };
  };

  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left font-semibold text-gray-700">Security Feature</th>
            <th className="px-4 py-3 text-left font-semibold text-gray-700">Status</th>
            <th className="px-4 py-3 text-left font-semibold text-gray-700">Confidence</th>
            <th className="px-4 py-3 text-left font-semibold text-gray-700">Details</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.key}
              className={`border-t transition-all duration-200 cursor-pointer ${
                i % 2 === 0 ? "bg-white" : "bg-gray-50"
              } ${highlightedFeature === row.key ? "ring-1 ring-blue-400 bg-blue-50" : ""}`}
              style={getHighlightStyle(row.key)}
              onMouseEnter={() => onFeatureHover(row.key)}
              onMouseLeave={() => onFeatureHover(null)}
            >
              <td className="px-4 py-3 font-medium">{row.feature}</td>
              <td className="px-4 py-3">
                <StatusBadge status={row.status} />
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        row.confidence >= 0.7
                          ? "bg-green-500"
                          : row.confidence >= 0.4
                          ? "bg-yellow-500"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${row.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs">{(row.confidence * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="px-4 py-3 text-gray-600">{row.details}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

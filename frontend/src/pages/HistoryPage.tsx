import { useState } from "react";
import { useHistory, useDeleteAnalysis } from "@/hooks/useAnalysis";

export default function HistoryPage() {
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<"all" | "real" | "fake">("all");
  const limit = 20;

  const { data, isLoading } = useHistory({ page, limit, filter });
  const deleteMutation = useDeleteAnalysis();

  const handleDelete = (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Delete this analysis?")) {
      deleteMutation.mutate(id);
    }
  };

  const handleView = (id: number) => {
    window.location.href = `/results/${id}`;
  };

  const handleNew = () => {
    window.location.href = "/";
  };

  // Format timestamp as YYYY:MM:DD:hh:mm:ss
  const formatTimestamp = (dateStr: string): string => {
    const d = new Date(dateStr);
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mi = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${yyyy}:${mm}:${dd}:${hh}:${mi}:${ss}`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analysis History</h1>
            <p className="text-sm text-gray-500 mt-1">
              All currency analyses with timestamps and results
            </p>
          </div>
          <button
            onClick={handleNew}
            className="bg-blue-600 text-white px-5 py-2.5 rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            + New Analysis
          </button>
        </div>

        {/* Stats Cards */}
        {data && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl p-5 text-center shadow-sm border">
              <div className="text-3xl font-bold text-gray-900">
                {data.pagination.total}
              </div>
              <div className="text-sm text-gray-500 mt-1">Total Analyses</div>
            </div>
            <div className="bg-white rounded-xl p-5 text-center shadow-sm border">
              <div className="text-3xl font-bold text-green-600">
                {data.data.filter((d) => d.result === "REAL").length}
              </div>
              <div className="text-sm text-gray-500 mt-1">Real Notes</div>
            </div>
            <div className="bg-white rounded-xl p-5 text-center shadow-sm border">
              <div className="text-3xl font-bold text-red-600">
                {data.data.filter((d) => d.result === "FAKE").length}
              </div>
              <div className="text-sm text-gray-500 mt-1">Fake Notes</div>
            </div>
            <div className="bg-white rounded-xl p-5 text-center shadow-sm border">
              <div className="text-3xl font-bold text-blue-600">
                {data.data.length > 0
                  ? (data.data.reduce((sum, d) => sum + d.confidence, 0) / data.data.length * 100).toFixed(0)
                  : 0}%
              </div>
              <div className="text-sm text-gray-500 mt-1">Avg Confidence</div>
            </div>
          </div>
        )}

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-6">
          {(["all", "real", "fake"] as const).map((f) => (
            <button
              key={f}
              onClick={() => {
                setFilter(f);
                setPage(1);
              }}
              className={`px-5 py-2.5 rounded-lg font-medium transition-colors ${
                filter === f
                  ? "bg-blue-600 text-white shadow-sm"
                  : "bg-white text-gray-600 hover:bg-gray-100 border"
              }`}
            >
              {f === "all" ? "📋 All" : f === "real" ? "✅ Real" : "❌ Fake"}
              {data && f !== "all" && (
                <span className="ml-2 text-xs opacity-75">
                  ({data.data.filter((d) => d.result === f.toUpperCase()).length})
                </span>
              )}
            </button>
          ))}
        </div>

        {/* History List */}
        {isLoading ? (
          <div className="text-center py-16 bg-white rounded-xl border">
            <div className="animate-spin text-3xl mb-3">⏳</div>
            <p className="text-gray-500">Loading history...</p>
          </div>
        ) : !data || data.data.length === 0 ? (
          <div className="text-center py-16 bg-white rounded-xl border">
            <div className="text-5xl mb-4">📋</div>
            <h3 className="text-lg font-semibold text-gray-700">No analyses yet</h3>
            <p className="text-gray-500 mt-2">
              Upload a currency note to get started
            </p>
            <button
              onClick={handleNew}
              className="mt-4 bg-blue-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-blue-700"
            >
              Analyze Your First Note
            </button>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {data.data.map((item) => {
                const timestamp = formatTimestamp(item.analyzed_at);
                const denom = item.denomination || "Unknown";
                const displayName = `${timestamp}_${denom}`;
                const isReal = item.result === "REAL";

                return (
                  <div
                    key={item.id}
                    onClick={() => handleView(item.id)}
                    className="bg-white rounded-xl border shadow-sm hover:shadow-md transition-all cursor-pointer overflow-hidden"
                  >
                    <div className="flex items-stretch">
                      {/* Thumbnail */}
                      <div className="w-24 h-24 flex-shrink-0 bg-gray-100">
                        <img
                          src={item.thumbnail}
                          alt="Thumbnail"
                          className="w-full h-full object-cover"
                        />
                      </div>

                      {/* Info */}
                      <div className="flex-1 p-4">
                        {/* Name */}
                        <div className="flex items-center gap-3 mb-2">
                          <code className="text-sm font-mono bg-gray-100 px-2 py-1 rounded text-gray-700">
                            {displayName}
                          </code>
                          <span
                            className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                              isReal
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
                            }`}
                          >
                            {isReal ? "✅ REAL" : "❌ FAKE"}
                          </span>
                        </div>

                        {/* Details */}
                        <div className="flex items-center gap-6 text-sm text-gray-500">
                          <div className="flex items-center gap-1.5">
                            <span>Confidence:</span>
                            <div className="w-16 bg-gray-200 rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${
                                  item.confidence >= 0.7
                                    ? "bg-green-500"
                                    : item.confidence >= 0.4
                                    ? "bg-yellow-500"
                                    : "bg-red-500"
                                }`}
                                style={{ width: `${item.confidence * 100}%` }}
                              />
                            </div>
                            <span className="font-medium text-gray-700">
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center pr-4">
                        <button
                          onClick={(e) => handleDelete(item.id, e)}
                          className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete"
                        >
                          🗑️
                        </button>
                        <div className="text-gray-400">→</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {data.pagination.total_pages > 1 && (
              <div className="flex items-center justify-between mt-6 bg-white rounded-xl border p-4">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-4 py-2 rounded-lg border bg-white disabled:opacity-40 hover:bg-gray-50 transition-colors"
                >
                  ← Previous
                </button>
                <div className="text-sm text-gray-600">
                  Page <span className="font-semibold">{page}</span> of{" "}
                  <span className="font-semibold">{data.pagination.total_pages}</span>
                  <span className="text-gray-400 ml-2">
                    ({data.pagination.total} total)
                  </span>
                </div>
                <button
                  onClick={() => setPage((p) => p + 1)}
                  disabled={page >= data.pagination.total_pages}
                  className="px-4 py-2 rounded-lg border bg-white disabled:opacity-40 hover:bg-gray-50 transition-colors"
                >
                  Next →
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

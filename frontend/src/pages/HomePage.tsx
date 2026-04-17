import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { ImageUploader } from "@/components/ImageUploader";
import { CameraCapture } from "@/components/CameraCapture";
import { useAnalyzeImage } from "@/hooks/useAnalysis";

export default function HomePage() {
  const [selectedImage, setSelectedImage] = useState("");
  const [showCamera, setShowCamera] = useState(false);

  const analyzeMutation = useAnalyzeImage();

  const handleAnalyze = () => {
    if (!selectedImage) return;
    analyzeMutation.mutate(
      { base64Image: selectedImage, source: "upload" },
      {
        onSuccess: (data) => {
          window.location.href = `/results/${data.id}`;
        },
      }
    );
  };

  const handleCameraCapture = (base64Image: string) => {
    setSelectedImage(base64Image);
    setShowCamera(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            🔍 Fake Currency Detection
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload or capture a photo of an Indian currency note to verify its
            authenticity using AI-powered analysis.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setShowCamera(false)}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                !showCamera
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              📁 Upload Image
            </button>
            <button
              onClick={() => setShowCamera(true)}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                showCamera
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              📷 Camera Capture
            </button>
          </div>

          {!showCamera ? (
            <ImageUploader onImageSelect={setSelectedImage} />
          ) : (
            <div className="text-center py-8">
              <button
                onClick={() => setShowCamera(true)}
                className="bg-blue-600 text-white px-8 py-4 rounded-lg font-medium hover:bg-blue-700 text-lg"
              >
                📸 Open Camera
              </button>
            </div>
          )}

          {selectedImage && !showCamera && (
            <button
              onClick={handleAnalyze}
              disabled={analyzeMutation.isPending}
              className="w-full mt-6 bg-green-600 text-white py-4 rounded-lg font-semibold text-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {analyzeMutation.isPending ? "⏳ Analyzing..." : "🔍 Analyze Currency Note"}
            </button>
          )}

          {analyzeMutation.isError && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              {(analyzeMutation.error as Error).message}
            </div>
          )}
        </div>

        {/* Camera Modal */}
        {showCamera && (
          <CameraCapture
            onCapture={handleCameraCapture}
            onClose={() => setShowCamera(false)}
          />
        )}

        {/* Features Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[
            { icon: "💧", name: "Watermark" },
            { icon: "🔗", name: "Security Thread" },
            { icon: "🎨", name: "Color Analysis" },
            { icon: "🔬", name: "Texture Quality" },
            { icon: "🔢", name: "Serial Number" },
            { icon: "📐", name: "Dimensions" },
          ].map((f) => (
            <div
              key={f.name}
              className="bg-white rounded-lg p-4 text-center shadow-sm border"
            >
              <div className="text-2xl mb-2">{f.icon}</div>
              <div className="text-sm font-medium text-gray-700">{f.name}</div>
            </div>
          ))}
        </div>

        {/* Disclaimer */}
        <p className="text-center text-xs text-gray-500 mt-8">
          ⚠️ This tool assists in currency verification. Always consult banking
          professionals for definitive authentication.
        </p>

        {/* View History Button */}
        <div className="text-center mt-6">
          <Link
            to="/history"
            className="inline-flex items-center gap-2 bg-gray-100 text-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
          >
            📋 View Analysis History
          </Link>
        </div>
      </div>
    </div>
  );
}

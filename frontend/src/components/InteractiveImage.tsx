import type { AnalysisResult } from "@/types";

interface InteractiveImageProps {
  analysis: AnalysisResult;
  highlightedFeature: string | null;
  onFeatureHover: (feature: string | null) => void;
}

interface FeatureRegion {
  key: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  status: "good" | "bad" | "warning";
  confidence: number;
}

export function InteractiveImage({ analysis, highlightedFeature, onFeatureHover }: InteractiveImageProps) {
  const img = new Image();
  img.src = analysis.annotated_image;
  const imgWidth = img.naturalWidth || 800;
  const imgHeight = img.naturalHeight || 500;

  // Define regions based on analysis data
  const regions: FeatureRegion[] = [];

  // Watermark region
  if (analysis.analysis.watermark.location) {
    const loc = analysis.analysis.watermark.location;
    regions.push({
      key: "watermark",
      label: "Watermark",
      x: loc.x,
      y: loc.y,
      width: loc.width,
      height: loc.height,
      status: analysis.analysis.watermark.status === "present" ? "good" : "bad",
      confidence: analysis.analysis.watermark.confidence,
    });
  } else {
    // Default position for watermark (right-center area)
    regions.push({
      key: "watermark",
      label: "Watermark",
      x: imgWidth * 0.65,
      y: imgHeight * 0.3,
      width: imgWidth * 0.2,
      height: imgHeight * 0.3,
      status: "warning",
      confidence: analysis.analysis.watermark.confidence,
    });
  }

  // Security thread region (vertical line area)
  if (analysis.analysis.security_thread.coordinates) {
    const coords = analysis.analysis.security_thread.coordinates;
    regions.push({
      key: "security_thread",
      label: "Security Thread",
      x: Math.min(coords.x_start, coords.x_end || coords.x_start) - 15,
      y: 0,
      width: 30 + Math.abs((coords.x_end || coords.x_start) - coords.x_start),
      height: imgHeight,
      status: analysis.analysis.security_thread.status === "present" ? "good" : "bad",
      confidence: analysis.analysis.security_thread.confidence,
    });
  } else {
    // Default position (left third of note)
    regions.push({
      key: "security_thread",
      label: "Security Thread",
      x: imgWidth * 0.15,
      y: 0,
      width: imgWidth * 0.05,
      height: imgHeight,
      status: "warning",
      confidence: analysis.analysis.security_thread.confidence,
    });
  }

  // Color analysis region (top area - Gandhi portrait area)
  regions.push({
    key: "color_analysis",
    label: "Color Analysis",
    x: imgWidth * 0.3,
    y: imgHeight * 0.05,
    width: imgWidth * 0.4,
    height: imgHeight * 0.2,
    status: analysis.analysis.color_analysis.status === "match" ? "good" : "bad",
    confidence: analysis.analysis.color_analysis.confidence,
  });

  // Texture region (main body area)
  regions.push({
    key: "texture_analysis",
    label: "Texture Quality",
    x: imgWidth * 0.1,
    y: imgHeight * 0.2,
    width: imgWidth * 0.8,
    height: imgHeight * 0.5,
    status: analysis.analysis.texture_analysis.status === "normal" ? "good" : "bad",
    confidence: analysis.analysis.texture_analysis.confidence,
  });

  // Serial number region (bottom area)
  regions.push({
    key: "serial_number",
    label: "Serial Number",
    x: imgWidth * 0.05,
    y: imgHeight * 0.82,
    width: imgWidth * 0.45,
    height: imgHeight * 0.12,
    status: analysis.analysis.serial_number.status === "valid" ? "good" : "bad",
    confidence: analysis.analysis.serial_number.confidence,
  });

  // Dimensions region (full note area)
  regions.push({
    key: "dimensions",
    label: "Dimensions",
    x: 0,
    y: 0,
    width: imgWidth,
    height: imgHeight,
    status: analysis.analysis.dimensions.status === "correct" ? "good" : "bad",
    confidence: analysis.analysis.dimensions.confidence,
  });

  const getStatusColor = (status: string, isHovered: boolean) => {
    if (!isHovered) return "rgba(0, 0, 0, 0)";
    switch (status) {
      case "good":
        return "rgba(34, 197, 94, 0.3)"; // Green with transparency
      case "bad":
        return "rgba(239, 68, 68, 0.3)"; // Red with transparency
      case "warning":
        return "rgba(234, 179, 8, 0.3)"; // Yellow with transparency
      default:
        return "rgba(100, 100, 100, 0.2)";
    }
  };

  const getStatusBorderColor = (status: string, isHovered: boolean) => {
    if (!isHovered) return "rgba(0, 0, 0, 0)";
    switch (status) {
      case "good":
        return "#22c55e";
      case "bad":
        return "#ef4444";
      case "warning":
        return "#eab308";
      default:
        return "#666";
    }
  };

  return (
    <div className="relative select-none">
      {/* Main annotated image */}
      <img
        src={analysis.annotated_image}
        alt="Analyzed currency note"
        className="w-full rounded-lg"
        style={{ display: "block" }}
      />

      {/* Interactive overlay regions */}
      <div className="absolute inset-0">
        {regions.map((region) => {
          const isHighlighted = highlightedFeature === region.key;
          const scaleX = imgWidth > 0 ? 100 / 1 : 1; // Scale will be handled by percentage
          const scaleY = imgHeight > 0 ? 100 / 1 : 1;

          // Convert pixel coords to percentage
          const left = (region.x / imgWidth) * 100;
          const top = (region.y / imgHeight) * 100;
          const width = (region.width / imgWidth) * 100;
          const height = (region.height / imgHeight) * 100;

          return (
            <div
              key={region.key}
              className="absolute cursor-pointer transition-all duration-200"
              style={{
                left: `${left}%`,
                top: `${top}%`,
                width: `${width}%`,
                height: `${height}%`,
                backgroundColor: getStatusColor(region.status, isHighlighted),
                border: isHighlighted ? `2px solid ${getStatusBorderColor(region.status, true)}` : "2px solid transparent",
                borderRadius: "4px",
                boxShadow: isHighlighted
                  ? `0 0 12px ${getStatusBorderColor(region.status, true)}40`
                  : "none",
              }}
              onMouseEnter={() => onFeatureHover(region.key)}
              onMouseLeave={() => onFeatureHover(null)}
              title={`${region.label}: ${region.status} (${(region.confidence * 100).toFixed(0)}%)`}
            >
              {isHighlighted && (
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap z-10">
                  {region.label}: {region.status} ({(region.confidence * 100).toFixed(0)}%)
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-3 text-xs justify-center">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span className="text-gray-600">Pass</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500" />
          <span className="text-gray-600">Fail</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-500" />
          <span className="text-gray-600">Warning</span>
        </div>
      </div>
    </div>
  );
}

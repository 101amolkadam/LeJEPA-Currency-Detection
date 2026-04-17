export interface AnalysisRequest {
  image: string;
  source: "upload" | "camera";
}

export interface WatermarkAnalysis {
  status: string;
  confidence: number;
  location: { x: number; y: number; width: number; height: number } | null;
  ssim_score: number | null;
}

export interface SecurityThreadAnalysis {
  status: string;
  confidence: number;
  position: string | null;
  coordinates: { x_start: number; x_end: number } | null;
}

export interface ColorAnalysis {
  status: string;
  confidence: number;
  bhattacharyya_distance: number | null;
  dominant_colors: string[] | null;
}

export interface TextureAnalysis {
  status: string;
  confidence: number;
  glcm_contrast: number | null;
  glcm_energy: number | null;
  sharpness_score: number | null;
}

export interface SerialNumberAnalysis {
  status: string;
  confidence: number;
  extracted_text: string | null;
  format_valid: boolean | null;
}

export interface DimensionsAnalysis {
  status: string;
  confidence: number;
  aspect_ratio: number | null;
  expected_aspect_ratio: number | null;
  deviation_percent: number | null;
}

export interface CNNClassification {
  result: string;
  confidence: number;
  model: string;
  processing_time_ms: number;
}

export interface AnalysisResult {
  id: number;
  result: string;
  confidence: number;
  currency_denomination: string | null;
  denomination_confidence: number | null;
  analysis: {
    cnn_classification: CNNClassification;
    watermark: WatermarkAnalysis;
    security_thread: SecurityThreadAnalysis;
    color_analysis: ColorAnalysis;
    texture_analysis: TextureAnalysis;
    serial_number: SerialNumberAnalysis;
    dimensions: DimensionsAnalysis;
  };
  ensemble_score: number;
  annotated_image: string;
  processing_time_ms: number;
  timestamp: string;
}

export interface HistoryItem {
  id: number;
  result: string;
  confidence: number;
  denomination: string | null;
  thumbnail: string;
  analyzed_at: string;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  total_pages: number;
}

export interface HistoryResponse {
  data: HistoryItem[];
  pagination: PaginationInfo;
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { analyzeImage, getHistory, getAnalysisById, deleteAnalysis } from "@/services/api";

export function useAnalyzeImage() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      base64Image,
      source,
    }: {
      base64Image: string;
      source?: "upload" | "camera";
    }) => analyzeImage(base64Image, source),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      return data;
    },
  });
}

export function useHistory(
  params: { page?: number; limit?: number; filter?: "all" | "real" | "fake" } = {}
) {
  return useQuery({
    queryKey: ["history", params],
    queryFn: () => getHistory(params),
  });
}

export function useAnalysis(id: string) {
  return useQuery({
    queryKey: ["analysis", id],
    queryFn: () => getAnalysisById(Number(id)),
    enabled: !!id,
  });
}

export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
    },
  });
}

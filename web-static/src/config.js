// Mirrors engine/config.py - keep the two in sync.
export const MODEL_ID = "onnx-community/embeddinggemma-2-ONNX";
export const MODEL_DTYPE = "q4"; // ~285 MB text+vision; fp32 activations (fp16 risks NaN per model card)
export const EMBED_DIM = 256;
export const VISION_TOKENS = 280;

// Ingestion
export const DEDUP_THRESHOLD = 0.92;
export const FACE_MATCH_THRESHOLD = 0.8;
export const MIN_FACE_PX = 40;
export const MAX_SIDE = 1600;
export const ACCEPT_FACELESS_SINGLE = true;

// Recommendation
export const ALPHA = 0.7;
export const DECAY_LAMBDA = 0.15;
export const VENUE_REPEAT_DAYS = 45;
export const VENUE_RECENT_LOOKS = 5;
export const VENUE_REPEAT_PENALTY = 0.5;
export const VIBE_WEIGHT = 0.6;
export const VIBE_BOOST = 0.25;
export const SEASON_BOOST = 0.1;

export const CLUSTER_RADIUS_KM = 0.5;
export const TRAVEL_RADIUS_KM = 60;

export const TASK_PREFIXES = {
  classification: "task: classification | query: ",
  clustering: "task: clustering | query: ",
  search: "task: search result | query: ",
  similarity: "task: sentence similarity | query: ",
  document: "title: none | text: ",
  none: "",
};

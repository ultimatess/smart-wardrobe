// Port of engine/ingest.py PersonLocator: face detect -> identity vs selfie -> pose body box.
import { FaceDetector, FilesetResolver, ImageEmbedder, PoseLandmarker } from "@mediapipe/tasks-vision";
import { ACCEPT_FACELESS_SINGLE, FACE_MATCH_THRESHOLD, MIN_FACE_PX } from "./config.js";
import { dot } from "./vec.js";

const BASE = import.meta.env?.BASE_URL ?? "/";

export function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

export const box = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 });
export const area = (b) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);
export const clip = (b, w, h) => box(Math.max(0, Math.round(b.x0)), Math.max(0, Math.round(b.y0)),
  Math.min(w, Math.round(b.x1)), Math.min(h, Math.round(b.y1)));
const contains = (b, x, y) => b.x0 <= x && x <= b.x1 && b.y0 <= y && y <= b.y1;

export function boxesFromPose(p, face, w, h) {
  const vis = p.filter((l) => l[2] > 0.3);
  const xs = (vis.length >= 4 ? vis : p).map((l) => l[0]);
  const [s1, s2] = [p[11], p[12]];
  const hips = [p[23], p[24]];
  const legs = [p[25], p[26], p[27], p[28]].filter((l) => l[2] > 0.3);
  const shoulderW = Math.abs(s1[0] - s2[0]) || w * 0.2;
  const top = face ? face.y1 : Math.min(s1[1], s2[1]) - 0.25 * shoulderW;
  const hipY = (hips[0][1] + hips[1][1]) / 2;
  const bottom = legs.length ? Math.max(...legs.map((l) => l[1])) + 0.1 * shoulderW : hipY + 1.2 * (hipY - top);
  const x0 = Math.min(...xs) - 0.35 * shoulderW, x1 = Math.max(...xs) + 0.35 * shoulderW;
  return {
    body: clip(box(x0, top, x1, bottom), w, h),
    upper: clip(box(x0, top, x1, hipY + 0.05 * shoulderW), w, h),
    lower: hipY < h * 0.95 ? clip(box(x0, hipY - 0.05 * shoulderW, x1, bottom), w, h) : null,
  };
}

export function boxesFromFace(f, w, h) {
  const fw = f.x1 - f.x0, fh = f.y1 - f.y0, cx = (f.x0 + f.x1) / 2;
  const body = clip(box(cx - 1.8 * fw, f.y1, cx + 1.8 * fw, f.y1 + 7 * fh), w, h);
  const mid = f.y1 + 3 * fh;
  return {
    body,
    upper: box(body.x0, body.y0, body.x1, Math.min(mid, body.y1)),
    lower: mid < body.y1 - fh ? box(body.x0, mid, body.x1, body.y1) : null,
  };
}

export class PersonLocator {
  static async create() {
    const files = await FilesetResolver.forVisionTasks(`${BASE}mediapipe`);
    const m = (n) => ({ baseOptions: { modelAssetPath: `${BASE}models/${n}`, delegate: "CPU" } });
    const loc = new PersonLocator();
    loc.faceDet = await FaceDetector.createFromOptions(files, { ...m("blaze_face_short_range.tflite"), minDetectionConfidence: 0.5 });
    loc.faceEmb = await ImageEmbedder.createFromOptions(files, { ...m("image_embedder.tflite"), l2Normalize: true, quantize: false });
    loc.pose = await PoseLandmarker.createFromOptions(files, { ...m("pose_landmarker_lite.task"), numPoses: 6, minPoseDetectionConfidence: 0.5 });
    loc.refVec = null;
    return loc;
  }

  faces(canvas) {
    return this.faceDet.detect(canvas).detections.map((d) => d.boundingBox)
      .filter((b) => Math.min(b.width, b.height) >= MIN_FACE_PX)
      .map((b) => box(b.originX, b.originY, b.originX + b.width, b.originY + b.height));
  }

  faceVec(canvas, f) {
    const w = f.x1 - f.x0, h = f.y1 - f.y0;
    const b = clip(box(f.x0 - 0.2 * w, f.y0 - 0.2 * h, f.x1 + 0.2 * w, f.y1 + 0.2 * h), canvas.width, canvas.height);
    const c = makeCanvas(224, 224);
    c.getContext("2d").drawImage(canvas, b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0, 0, 0, 224, 224);
    return Float32Array.from(this.faceEmb.embed(c).embeddings[0].floatEmbedding);
  }

  /** Returns a face vector for the reference selfie, or throws if no face is found. */
  selfieVec(canvas) {
    const faces = this.faces(canvas);
    if (!faces.length) throw new Error("No face found in the selfie - try a clearer, front-facing photo.");
    return this.faceVec(canvas, faces.reduce((a, b) => (area(b) > area(a) ? b : a)));
  }

  poses(canvas) {
    const { width: w, height: h } = canvas;
    return this.pose.detect(canvas).landmarks.map((p) => p.map((l) => [l.x * w, l.y * h, l.visibility ?? 1]));
  }

  locate(canvas) {
    const { width: w, height: h } = canvas;
    const faces = this.faces(canvas);
    const poses = this.poses(canvas);
    let face = null, identity = null;
    if (faces.length) {
      if (this.refVec) {
        const scored = faces.map((f) => [dot(this.faceVec(canvas, f), this.refVec), f]).sort((a, b) => b[0] - a[0]);
        [identity, face] = scored[0];
        if (identity < FACE_MATCH_THRESHOLD) return { body: null, identity, reason: "no face matched reference" };
      } else {
        face = faces.reduce((a, b) => (area(b) > area(a) ? b : a)); // single-user mode
      }
    } else if (!(ACCEPT_FACELESS_SINGLE && poses.length === 1)) {
      return { body: null, identity, reason: "no face (and not a single-person photo)" };
    }
    let pose = null;
    if (face) {
      const fw = face.x1 - face.x0, fh = face.y1 - face.y0;
      const grown = box(face.x0 - 0.3 * fw, face.y0 - 0.3 * fh, face.x1 + 0.3 * fw, face.y1 + 0.3 * fh);
      pose = poses.find((p) => contains(grown, p[0][0], p[0][1])) || null;
    } else pose = poses[0];
    if (pose) {
      const reason = face ? "pose" : "pose; unanchored single person (no face visible)";
      return { ...boxesFromPose(pose, face, w, h), identity, reason };
    }
    return { ...boxesFromFace(face, w, h), identity, reason: "face-heuristic" };
  }
}

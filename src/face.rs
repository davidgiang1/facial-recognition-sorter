use anyhow::{Result, Context};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

// ArcFace canonical landmark positions for 112x112 aligned face
const ARCFACE_REF: [[f32; 2]; 5] = [
    [38.2946, 51.6963],  // left eye
    [73.5318, 51.5014],  // right eye
    [56.0252, 71.7366],  // nose
    [41.5493, 92.3655],  // left mouth corner
    [70.7299, 92.2041],  // right mouth corner
];

const SCRFD_STRIDES: [usize; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;
const NMS_THRESHOLD: f32 = 0.4;
const MIN_FACE_SIZE: f32 = 40.0;

#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: [f32; 4],
    pub landmarks: Vec<[f32; 2]>,
    pub score: f32,
}

// --- Raw output structs for split preprocess/infer/postprocess ---

/// Raw tensor data from SCRFD face detection inference
pub struct RawFaceOutput {
    /// 9 tensors: (flattened data, shape) for each
    pub tensors: Vec<(Vec<f32>, Vec<usize>)>,
}

/// Preprocessing result for face detection
pub struct FacePreprocessed {
    pub tensor: Array4<f32>,
    pub scale: f32,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub struct FaceDetector {
    session: Mutex<Session>,
    input_name: String,
    output_names: Vec<String>,
}

impl FaceDetector {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::ep::DirectML::default().build()])?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();
        let output_names: Vec<String> = session.outputs().iter()
            .map(|o| o.name().to_string())
            .collect();

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_names,
        })
    }

    /// Print model input/output metadata for debugging
    pub fn inspect(&self) {
        let session = self.session.lock().unwrap();
        println!("=== Detection Model ===");
        println!("Inputs:");
        for input in session.inputs() {
            println!("  name: {:?}, type: {:?}", input.name(), input.dtype());
        }
        println!("Outputs:");
        for output in session.outputs() {
            println!("  name: {:?}, type: {:?}", output.name(), output.dtype());
        }
    }

    /// Run detection and print raw output shapes (for model inspection)
    pub fn inspect_outputs(&self, img: &DynamicImage) -> Result<()> {
        let preprocessed = Self::preprocess_image(img);
        let input_tensor = Value::from_array(preprocessed.tensor)?;
        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;

        println!("\nDetection output tensors:");
        for i in 0..outputs.len() {
            if let Ok((shape, _data)) = outputs[i].try_extract_tensor::<f32>() {
                println!("  [{}] name={:?}, shape={:?}", i, self.output_names.get(i), &*shape);
            }
        }

        Ok(())
    }

    /// CPU-only preprocessing: resize + normalize to 640x640 tensor. No mutex needed.
    pub fn preprocess_image(img: &DynamicImage) -> FacePreprocessed {
        let target_size = 640usize;
        let (width, height) = img.dimensions();
        let scale = target_size as f32 / width.max(height) as f32;
        let new_w = (width as f32 * scale) as u32;
        let new_h = (height as f32 * scale) as u32;

        let resized = img.resize(new_w, new_h, image::imageops::FilterType::Triangle);
        let mut input = Array4::<f32>::zeros((1, 3, target_size, target_size));

        for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0; // B
            input[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0; // G
            input[[0, 2, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0; // R
        }

        FacePreprocessed { tensor: input, scale, orig_width: width, orig_height: height }
    }

    /// GPU-only inference: lock mutex, run session, clone outputs, unlock.
    pub fn infer(&self, preprocessed: &FacePreprocessed) -> Result<RawFaceOutput> {
        let input_tensor = Value::from_array(preprocessed.tensor.clone())?;
        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;

        let output_count = outputs.len();
        if output_count != 9 {
            anyhow::bail!(
                "Expected 9 output tensors (SCRFD), got {}. Run with --inspect to debug.",
                output_count
            );
        }

        // Clone tensor data out so we can drop the session lock
        let mut tensors = Vec::with_capacity(output_count);
        for i in 0..output_count {
            let (shape, data) = outputs[i].try_extract_tensor::<f32>()
                .context(format!("Failed to extract output tensor {}", i))?;
            let shape_vec: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
            tensors.push((data.to_vec(), shape_vec));
        }

        Ok(RawFaceOutput { tensors })
    }

    /// CPU-only postprocessing: decode raw tensors into face detections + NMS.
    pub fn postprocess(raw: &RawFaceOutput, scale: f32, orig_width: u32, orig_height: u32) -> Result<Vec<FaceDetection>> {
        // Categorize outputs by their last dimension (cols)
        struct TensorMeta {
            index: usize,
            rows: usize,
        }
        let mut score_tensors = Vec::new();
        let mut bbox_tensors = Vec::new();
        let mut landmark_tensors = Vec::new();

        for (i, (_, shape)) in raw.tensors.iter().enumerate() {
            let rows = shape[0];
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            match cols {
                1 => score_tensors.push(TensorMeta { index: i, rows }),
                4 => bbox_tensors.push(TensorMeta { index: i, rows }),
                10 => landmark_tensors.push(TensorMeta { index: i, rows }),
                _ => anyhow::bail!("Unexpected output cols {} at index {} (shape={:?})", cols, i, shape),
            }
        }

        if score_tensors.len() != 3 || bbox_tensors.len() != 3 || landmark_tensors.len() != 3 {
            anyhow::bail!(
                "Expected 3 score, 3 bbox, 3 landmark tensors. Got {}, {}, {}.",
                score_tensors.len(), bbox_tensors.len(), landmark_tensors.len()
            );
        }

        // Sort by row count descending (stride 8 = most anchors first)
        score_tensors.sort_by(|a, b| b.rows.cmp(&a.rows));
        bbox_tensors.sort_by(|a, b| b.rows.cmp(&a.rows));
        landmark_tensors.sort_by(|a, b| b.rows.cmp(&a.rows));

        let anchor_centers = generate_anchor_centers(640, &SCRFD_STRIDES, ANCHORS_PER_CELL);

        let width = orig_width as f32;
        let height = orig_height as f32;
        let mut all_detections = Vec::new();

        for (stride_idx, &stride) in SCRFD_STRIDES.iter().enumerate() {
            let s_data = &raw.tensors[score_tensors[stride_idx].index].0;
            let b_data = &raw.tensors[bbox_tensors[stride_idx].index].0;
            let l_data = &raw.tensors[landmark_tensors[stride_idx].index].0;

            let centers = &anchor_centers[stride_idx];
            let n_anchors = score_tensors[stride_idx].rows;

            for i in 0..n_anchors {
                let score = s_data[i];

                if score < 0.5 {
                    continue;
                }

                let (cx, cy) = centers[i];
                let stride_f = stride as f32;

                let bi = i * 4;
                let x1 = (cx - b_data[bi] * stride_f) / scale;
                let y1 = (cy - b_data[bi + 1] * stride_f) / scale;
                let x2 = (cx + b_data[bi + 2] * stride_f) / scale;
                let y2 = (cy + b_data[bi + 3] * stride_f) / scale;

                let x1 = x1.max(0.0).min(width);
                let y1 = y1.max(0.0).min(height);
                let x2 = x2.max(0.0).min(width);
                let y2 = y2.max(0.0).min(height);

                if (x2 - x1) < MIN_FACE_SIZE || (y2 - y1) < MIN_FACE_SIZE {
                    continue;
                }

                let face_cx = (x1 + x2) / 2.0;
                let face_cy = (y1 + y2) / 2.0;
                if face_cx < 0.0 || face_cx > width || face_cy < 0.0 || face_cy > height {
                    continue;
                }

                let li = i * 10;
                let mut lmks = Vec::with_capacity(5);
                for k in 0..5 {
                    let lx = (cx + l_data[li + k * 2] * stride_f) / scale;
                    let ly = (cy + l_data[li + k * 2 + 1] * stride_f) / scale;
                    lmks.push([lx, ly]);
                }

                all_detections.push(FaceDetection {
                    bbox: [x1, y1, x2, y2],
                    landmarks: lmks,
                    score,
                });
            }
        }

        let result = nms(&mut all_detections, NMS_THRESHOLD);
        Ok(result)
    }

    /// Convenience: combined preprocess + infer + postprocess (for target extraction and simple usage)
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<FaceDetection>> {
        let preprocessed = Self::preprocess_image(img);
        let raw = self.infer(&preprocessed)?;
        Self::postprocess(&raw, preprocessed.scale, preprocessed.orig_width, preprocessed.orig_height)
    }
}

pub struct FaceRecognizer {
    session: Mutex<Session>,
    input_name: String,
}

impl FaceRecognizer {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::ep::DirectML::default().build()])?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();

        Ok(Self {
            session: Mutex::new(session),
            input_name,
        })
    }

    pub fn inspect(&self) {
        let session = self.session.lock().unwrap();
        println!("=== Recognition Model ===");
        println!("Inputs:");
        for input in session.inputs() {
            println!("  name: {:?}, type: {:?}", input.name(), input.dtype());
        }
        println!("Outputs:");
        for output in session.outputs() {
            println!("  name: {:?}, type: {:?}", output.name(), output.dtype());
        }
    }

    pub fn recognize(&self, face_img: &DynamicImage) -> Result<Vec<f32>> {
        let batch = [face_img.clone()];
        let embeddings = self.recognize_batch(&batch)?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    pub fn recognize_batch(&self, face_imgs: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        if face_imgs.is_empty() { return Ok(Vec::new()); }

        let mut results = Vec::with_capacity(face_imgs.len());

        for img in face_imgs {
            let mut input = Array4::<f32>::zeros((1, 3, 112, 112));
            let resized = img.resize_exact(112, 112, image::imageops::FilterType::Triangle);
            
            for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
                input[[0, 0, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0; // B
                input[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0; // G
                input[[0, 2, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0; // R
            }

            let input_tensor = Value::from_array(input)?;
            let inputs = ort::inputs![self.input_name.as_str() => input_tensor];

            let mut session = self.session.lock().unwrap();
            let outputs = session.run(inputs)?;

            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let embedding_dim = shape[1] as usize;

            let raw = &data[0..embedding_dim];

            // L2 normalize
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let embedding: Vec<f32> = if norm > 0.0 {
                raw.iter().map(|x| x / norm).collect()
            } else {
                raw.to_vec()
            };
            results.push(embedding);
        }

        Ok(results)
    }
}

// --- Face alignment ---


/// Align a face using detected landmarks and the ArcFace reference template.
/// optimized for performance using direct buffer access where safe.
pub fn align_face(img: &DynamicImage, landmarks: &[[f32; 2]; 5]) -> DynamicImage {
    let transform = umeyama_similarity(landmarks, &ARCFACE_REF);
    let inv = invert_similarity(&transform);

    let rgb = img.to_rgb8();
    let (src_w, src_h) = (rgb.width(), rgb.height());
    let mut output = RgbImage::new(112, 112);

    // Flattened buffer access for speed
    let input_buf = rgb.as_raw();
    let output_buf = output.as_flat_samples_mut().samples;
    let stride = (src_w * 3) as usize;

    for out_y in 0..112u32 {
        for out_x in 0..112u32 {
            let src_x = inv[0] * out_x as f32 + inv[1] * out_y as f32 + inv[2];
            let src_y = inv[3] * out_x as f32 + inv[4] * out_y as f32 + inv[5];

            // Bilinear interpolation inline
            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            // Safe boundary checks
            let mut pixel = [0u8; 3];
            if x0 >= 0 && x1 < src_w as i32 && y0 >= 0 && y1 < src_h as i32 {
                 // Fast path: all pixels inside bounds
                let idx00 = (y0 as usize * stride) + (x0 as usize * 3);
                let idx10 = (y0 as usize * stride) + (x1 as usize * 3);
                let idx01 = (y1 as usize * stride) + (x0 as usize * 3);
                let idx11 = (y1 as usize * stride) + (x1 as usize * 3);

                for c in 0..3 {
                     let p00 = input_buf[idx00 + c] as f32;
                     let p10 = input_buf[idx10 + c] as f32;
                     let p01 = input_buf[idx01 + c] as f32;
                     let p11 = input_buf[idx11 + c] as f32;

                     let val = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
                     pixel[c] = val as u8;
                }
            } else {
                 // Slow path: boundary handling (clamp to edges)
                 let get_comp = |x: i32, y: i32, c: usize| -> f32 {
                     let cx = x.clamp(0, src_w as i32 - 1) as usize;
                     let cy = y.clamp(0, src_h as i32 - 1) as usize;
                     input_buf[cy * stride + cx * 3 + c] as f32
                 };

                 for c in 0..3 {
                     let p00 = get_comp(x0, y0, c);
                     let p10 = get_comp(x1, y0, c);
                     let p01 = get_comp(x0, y1, c);
                     let p11 = get_comp(x1, y1, c);

                     let val = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
                     pixel[c] = val as u8;
                 }
            }

            let out_idx = ((out_y * 112 + out_x) * 3) as usize;
            output_buf[out_idx] = pixel[0];
            output_buf[out_idx + 1] = pixel[1];
            output_buf[out_idx + 2] = pixel[2];
        }
    }

    DynamicImage::ImageRgb8(output)
}

fn umeyama_similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [f32; 6] {
    let n = 5.0f32;

    let (mut scx, mut scy) = (0.0f32, 0.0f32);
    let (mut dcx, mut dcy) = (0.0f32, 0.0f32);
    for i in 0..5 {
        scx += src[i][0]; scy += src[i][1];
        dcx += dst[i][0]; dcy += dst[i][1];
    }
    scx /= n; scy /= n;
    dcx /= n; dcy /= n;

    let (mut sxx, mut sxy, mut syx, mut syy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut src_var = 0.0f32;

    for i in 0..5 {
        let sx = src[i][0] - scx;
        let sy = src[i][1] - scy;
        let dx = dst[i][0] - dcx;
        let dy = dst[i][1] - dcy;

        sxx += sx * dx;
        sxy += sx * dy;
        syx += sy * dx;
        syy += sy * dy;
        src_var += sx * sx + sy * sy;
    }
    src_var /= n;

    let num_cos = sxx + syy;
    let num_sin = syx - sxy;
    let denom = (num_cos * num_cos + num_sin * num_sin).sqrt();

    if denom < 1e-10 || src_var < 1e-10 {
        return [1.0, 0.0, dcx - scx, 0.0, 1.0, dcy - scy];
    }

    let cos_t = num_cos / denom;
    let sin_t = num_sin / denom;
    let scale = denom / (src_var * n);

    let a = scale * cos_t;
    let b = scale * sin_t;
    let tx = dcx - a * scx + b * scy;
    let ty = dcy - b * scx - a * scy;

    [a, -b, tx, b, a, ty]
}

fn invert_similarity(t: &[f32; 6]) -> [f32; 6] {
    let a = t[0];
    let tx = t[2];
    let b = t[3];
    let ty = t[5];

    let det = a * a + b * b;
    if det < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    }

    let inv_a = a / det;
    let inv_b = b / det;
    let inv_tx = -(inv_a * tx + inv_b * ty);
    let inv_ty = inv_b * tx - inv_a * ty;

    [inv_a, inv_b, inv_tx, -inv_b, inv_a, inv_ty]
}

fn generate_anchor_centers(
    input_size: usize,
    strides: &[usize],
    anchors_per_cell: usize,
) -> Vec<Vec<(f32, f32)>> {
    strides
        .iter()
        .map(|&stride| {
            let feat_size = input_size / stride;
            let mut centers = Vec::with_capacity(feat_size * feat_size * anchors_per_cell);
            for row in 0..feat_size {
                for col in 0..feat_size {
                    let cx = col as f32 * stride as f32;
                    let cy = row as f32 * stride as f32;
                    for _ in 0..anchors_per_cell {
                        centers.push((cx, cy));
                    }
                }
            }
            centers
        })
        .collect()
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);
    let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter_area / (area_a + area_b - inter_area + 1e-6)
}

fn nms(detections: &mut Vec<FaceDetection>, iou_threshold: f32) -> Vec<FaceDetection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut suppressed = vec![false; detections.len()];
    let mut keep = Vec::new();

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&detections[i].bbox, &detections[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep.into_iter()
        .map(|i| detections[i].clone())
        .collect()
}

// --- Rotation correction for tilted face detection ---

const EYE_TILT_THRESHOLD: f32 = 0.2618; // ~15 degrees

/// Angle of the left-eye → right-eye line from horizontal (radians).
/// 0 = perfectly horizontal (upright face).
pub fn eye_line_angle(landmarks: &[[f32; 2]]) -> f32 {
    if landmarks.len() < 2 { return 0.0; }
    let dx = landmarks[1][0] - landmarks[0][0];
    let dy = landmarks[1][1] - landmarks[0][1];
    dy.atan2(dx)
}

/// Check whether 5-point landmarks follow the expected spatial layout for an
/// upright face: eyes at the top, nose in the middle, mouth at the bottom,
/// with reasonable proportions.
fn landmarks_plausible(landmarks: &[[f32; 2]]) -> bool {
    if landmarks.len() < 5 { return true; } // can't validate without all 5

    let eye_cy = (landmarks[0][1] + landmarks[1][1]) / 2.0;
    let nose_y = landmarks[2][1];
    let mouth_cy = (landmarks[3][1] + landmarks[4][1]) / 2.0;

    // 1. Eye-line tilt must be small
    if eye_line_angle(landmarks).abs() > EYE_TILT_THRESHOLD {
        return false;
    }

    // 2. Vertical ordering: eyes above nose above mouth
    if nose_y <= eye_cy || mouth_cy <= nose_y {
        return false;
    }

    // 3. Nose should be 20-80% of the way from eyes to mouth (typical ~45%)
    let total_v = mouth_cy - eye_cy;
    if total_v < 1.0 { return false; }
    let nose_ratio = (nose_y - eye_cy) / total_v;
    if nose_ratio < 0.2 || nose_ratio > 0.8 {
        return false;
    }

    true
}

/// Rotate a DynamicImage by `angle_rad` (positive = CCW) around its center.
pub fn rotate_image(img: &DynamicImage, angle_rad: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let new_w = (w as f32 * cos_a.abs() + h as f32 * sin_a.abs()).ceil() as u32;
    let new_h = (w as f32 * sin_a.abs() + h as f32 * cos_a.abs()).ceil() as u32;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ncx = new_w as f32 / 2.0;
    let ncy = new_h as f32 / 2.0;

    let input_buf = rgb.as_raw();
    let src_stride = (w * 3) as usize;
    let mut output = RgbImage::new(new_w, new_h);
    let output_buf = output.as_flat_samples_mut().samples;

    for out_y in 0..new_h {
        for out_x in 0..new_w {
            let dx = out_x as f32 - ncx;
            let dy = out_y as f32 - ncy;
            // Inverse rotation to map output → input
            let src_x = dx * cos_a + dy * sin_a + cx;
            let src_y = -dx * sin_a + dy * cos_a + cy;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            if x0 < 0 || x0 + 1 >= w as i32 || y0 < 0 || y0 + 1 >= h as i32 {
                continue; // black background
            }

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let idx00 = y0 as usize * src_stride + x0 as usize * 3;
            let idx10 = y0 as usize * src_stride + (x0 + 1) as usize * 3;
            let idx01 = (y0 + 1) as usize * src_stride + x0 as usize * 3;
            let idx11 = (y0 + 1) as usize * src_stride + (x0 + 1) as usize * 3;

            let out_idx = (out_y * new_w + out_x) as usize * 3;
            for c in 0..3 {
                let val = input_buf[idx00 + c] as f32 * w00
                    + input_buf[idx10 + c] as f32 * w10
                    + input_buf[idx01 + c] as f32 * w01
                    + input_buf[idx11 + c] as f32 * w11;
                output_buf[out_idx + c] = val as u8;
            }
        }
    }

    DynamicImage::ImageRgb8(output)
}

/// Score a set of detections for landmark quality. Higher = better.
/// Detections with at least one plausible face rank far above all-implausible ones;
/// within each tier, smaller eye-line tilt wins.
fn detection_quality(dets: &[FaceDetection]) -> f32 {
    let plausible_min_tilt = dets.iter()
        .filter(|d| landmarks_plausible(&d.landmarks))
        .map(|d| eye_line_angle(&d.landmarks).abs())
        .fold(f32::MAX, f32::min);

    if plausible_min_tilt < f32::MAX {
        // Has at least one plausible face — prefer smaller remaining tilt
        100.0 - plausible_min_tilt.to_degrees()
    } else {
        // All implausible — prefer the least-bad tilt as a tie-breaker
        let min_tilt = dets.iter()
            .filter(|d| d.landmarks.len() >= 2)
            .map(|d| eye_line_angle(&d.landmarks).abs())
            .fold(f32::MAX, f32::min);
        -min_tilt.to_degrees()
    }
}

/// Pre-rotate the image by the negative of the detected eye-line tilt, then
/// re-detect. Accepts the new detections only if landmark plausibility improves.
/// Returns (best detections, Option<rotated image>).
/// Use the rotated image (if Some) for alignment; otherwise use the original.
pub fn correct_rotation(
    detector: &FaceDetector,
    img: &DynamicImage,
    detections: Vec<FaceDetection>,
) -> (Vec<FaceDetection>, Option<DynamicImage>) {
    // 1. Quick exit if we have perfect results already
    if detections.is_empty() {
        return (detections, None);
    }
    let initial_quality = detection_quality(&detections);
    if initial_quality > 98.0 {
        return (detections, None);
    }

    // 2. Work on a "proxy" image to make rotation search cheap.
    // Rotating a 4K image 5 times is slow. Rotating an 800px image is fast.
    // SCRFD resizes to 640x640 anyway, so we lose no detection accuracy.
    let (w, h) = img.dimensions();
    let proxy_img = if w > 800 || h > 800 {
        img.resize(800, 800, image::imageops::FilterType::Triangle)
    } else {
        img.clone()
    };

    let mut best_angle = 0.0;
    let mut best_score = -999.0; // Start lower than any possible score
    let mut best_proxy_dets = Vec::new();
    
    // 3. Determine Search Candidates
    let mut candidates = Vec::new();
    candidates.push(0.0);

    let has_plausible = detections.iter().any(|d| landmarks_plausible(&d.landmarks));
    
    if has_plausible {
        // Smart Search: We have a face, just fix its tilt.
        // Check exact correction +/- 5 degrees to handle estimation noise.
        if let Some(primary) = detections.iter().max_by_key(|d| ((d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1])) as i32) {
             let tilt = eye_line_angle(&primary.landmarks);
             let correction = -tilt;
             candidates.push(correction);
             candidates.push(correction - 0.087); // -5 deg
             candidates.push(correction + 0.087); // +5 deg
        }
    } else {
        // Blind Search: Face is likely upside down or sideways. Check Cardinals.
        let pi = std::f32::consts::PI;
        candidates.push(pi / 2.0);
        candidates.push(-pi / 2.0);
        candidates.push(pi);
    }

    // 4. Run Search on Proxy
    for angle in candidates {
        let rotated_proxy = rotate_image(&proxy_img, angle);
        if let Ok(dets) = detector.detect(&rotated_proxy) {
            // Even if empty, we check (score will be low)
            let score = if dets.is_empty() { -999.0 } else { detection_quality(&dets) };
            if score > best_score {
                best_score = score;
                best_angle = angle;
                best_proxy_dets = dets;
            }
        }
    }

    // 5. Refinement (Optimization)
    // If we picked a cardinal direction (or even a rough tilt), we can do one final 
    // micro-adjustment based on the detections *at that angle* to get 0.00° tilt.
    if !best_proxy_dets.is_empty() && best_score > -100.0 {
        if let Some(primary) = best_proxy_dets.iter().max_by_key(|d| ((d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1])) as i32) {
             let residual_tilt = eye_line_angle(&primary.landmarks);
             // If significant residual tilt, correct it
             if residual_tilt.abs() > 0.017 { // > 1 degree
                 best_angle -= residual_tilt;
             }
        }
    }

    // 6. Apply Final Result to Original High-Res Image
    // Only return if we actually improved over the baseline
    if best_angle.abs() > 1e-4 {
        let final_rotated = rotate_image(img, best_angle);
        if let Ok(final_dets) = detector.detect(&final_rotated) {
            let final_q = if final_dets.is_empty() { -999.0 } else { detection_quality(&final_dets) };
            
            // Accept if better, or if we rescued an implausible face
            if final_q > initial_quality || (!has_plausible && final_q > -100.0) {
                 return (final_dets, Some(final_rotated));
            }
        }
    }

    (detections, None)
}

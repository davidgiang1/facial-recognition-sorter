mod face;
mod gui;
mod utils;

use anyhow::Result;
use face::{FaceDetector, FaceRecognizer, align_face, correct_rotation};
use gui::{FaceSearchApp, UiMessage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;
use crossbeam_channel::bounded;

#[derive(Serialize, Deserialize)]
struct TargetCache {
    hash: u64,
    entries: Vec<PersonInfo>,
}

const TARGET_CACHE_FILE: &str = "target_cache.bin";

fn compute_target_hash(target_images: &[(PathBuf, u64)]) -> u64 {
    let mut hasher = DefaultHasher::new();
    11u64.hash(&mut hasher);  // bumped: forces cache invalidation when PersonInfo schema changes
    for (path, size) in target_images {
        path.to_string_lossy().hash(&mut hasher);
        size.hash(&mut hasher);
    }
    hasher.finish()
}

fn load_target_cache() -> Option<TargetCache> {
    if Path::new(TARGET_CACHE_FILE).exists() {
        if let Ok(data) = fs::read(TARGET_CACHE_FILE) {
            if let Ok(cache) = bincode::deserialize::<TargetCache>(&data) {
                return Some(cache);
            }
        }
    }
    None
}

fn save_target_cache(cache: &TargetCache) {
    if let Ok(data) = bincode::serialize(cache) {
        let _ = fs::write(TARGET_CACHE_FILE, data);
    }
}

const DB_VERSION: u32 = 6;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersonInfo {
    pub face_bbox: [f32; 4],
    pub face_score: f32,
    pub face_embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Database {
    #[serde(default)]
    pub version: u32,
    pub images: HashMap<String, Vec<PersonInfo>>,
}

pub const DB_FILE: &str = "faces_db.bin";
pub const DB_FILE_JSON: &str = "faces_db.json";

fn load_database() -> Database {
    if Path::new(DB_FILE).exists() {
        if let Ok(data) = fs::read(DB_FILE) {
            if let Ok(db) = bincode::deserialize::<Database>(&data) {
                if db.version == DB_VERSION {
                    return db;
                }
            }
        }
    }
    if Path::new(DB_FILE_JSON).exists() {
        if let Ok(content) = fs::read_to_string(DB_FILE_JSON) {
            if let Ok(db) = serde_json::from_str::<Database>(&content) {
                if db.version == DB_VERSION {
                    return db;
                }
            }
        }
    }
    Database::default()
}

fn save_database(db: &mut Database) -> Result<()> {
    db.version = DB_VERSION;
    let data = bincode::serialize(db)?;
    fs::write(DB_FILE, data)?;
    Ok(())
}

fn ensure_video_thumbnail(
    video_path: &Path,
    detector: &FaceDetector,
) -> anyhow::Result<PathBuf> {
    let thumb_path = crate::utils::get_video_thumbnail_path(video_path);
    if thumb_path.exists() {
        return Ok(thumb_path);
    }

    if let Some(ffmpeg_cmd) = crate::utils::find_ffmpeg_path() {
        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("fr_thumb_{}_{}", std::process::id(), timestamp));
        std::fs::create_dir(&temp_dir)?;

        let output_pattern = temp_dir.join("frame_%04d.jpg");
        let status = std::process::Command::new(&ffmpeg_cmd)
            .arg("-i").arg(video_path)
            .arg("-vf").arg("fps=1")
            .arg(&output_pattern)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()?;

        if status.success() {
            let frames: Vec<PathBuf> = walkdir::WalkDir::new(&temp_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| crate::utils::is_image(e.path()))
                .map(|e| e.path().to_path_buf())
                .collect();

            let mut best_score = -1.0;
            let mut best_frame: Option<PathBuf> = None;

            for frame_path in &frames {
                if let Ok(img) = crate::utils::load_image_robustly(frame_path) {
                    if let Ok(dets) = detector.detect(&img) {
                        for fd in dets {
                            if fd.score > best_score {
                                best_score = fd.score;
                                best_frame = Some(frame_path.clone());
                            }
                        }
                    }
                }
            }

            if let Some(best) = best_frame {
                let _ = std::fs::create_dir_all(thumb_path.parent().unwrap());
                let _ = std::fs::copy(&best, &thumb_path);
            } else if let Some(first) = frames.first() {
                let _ = std::fs::create_dir_all(thumb_path.parent().unwrap());
                let _ = std::fs::copy(first, &thumb_path);
            } else {
                 let _ = std::fs::create_dir_all(thumb_path.parent().unwrap());
                 let _ = std::fs::File::create(&thumb_path);
            }
        }
        let _ = std::fs::remove_dir_all(&temp_dir);
        
        if thumb_path.exists() {
            return Ok(thumb_path);
        }
    }
    
    anyhow::bail!("Failed to generate thumbnail for video")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot_product.clamp(-1.0, 1.0)
}

fn calculate_similarity(target: &PersonInfo, candidate: &PersonInfo) -> f32 {
    cosine_similarity(&target.face_embedding, &candidate.face_embedding)
}

pub fn process_directory(
    input: PathBuf,
    target_dir: Option<PathBuf>,
    people_dir: Option<PathBuf>,
    match_threshold_min: f32,
    match_threshold_max: f32,
    filter_threshold: f32,
    tx: std::sync::mpsc::Sender<UiMessage>,
) -> Result<()> {
    macro_rules! log {
        ($($arg:tt)*) => {
            let msg = format!($($arg)*);
            println!("{}", msg);
            let _ = tx.send(UiMessage::Log(msg));
        };
    }

    log!("Loading ONNX models...");
    let models_dir = PathBuf::from("models");

    // NOTE: FP16 variants are preferred if available (run quantize_models.py to generate)
    let det_path = models_dir.join("det.onnx"); // Keep FP32 for biometric accuracy
    let rec_path = models_dir.join("rec.onnx"); // Keep FP32 for biometric accuracy

    if !det_path.exists() || !rec_path.exists() {
        anyhow::bail!("Model files not found in 'models' directory. Ensure correct filenames are present.");
    }

    let detector = Arc::new(FaceDetector::new(&det_path)?);
    let recognizer = Arc::new(FaceRecognizer::new(&rec_path)?);

    let mut target_images_with_size: Vec<(PathBuf, u64)> = Vec::new();
    let mut target_filesizes: std::collections::HashMap<u64, Vec<PathBuf>> = std::collections::HashMap::new();

    // Build target_images_with_size for face recognition reference (target dir only)
    if let Some(td) = &target_dir {
        if td.exists() {
            if let Ok(entries) = fs::read_dir(td) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_file() && (crate::utils::is_image(&path) || crate::utils::is_video(&path)) {
                        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        target_images_with_size.push((path.clone(), file_size));
                    }
                }
            }
        }
    }

    // Build target_filesizes "already seen" filter from people directory
    // This recursively scans all person subdirectories, excluding images already
    // sorted into any person's folder (including siblings) from appearing in results.
    if let Some(pd) = &people_dir {
        if pd.exists() {
            for entry in WalkDir::new(pd).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() && (crate::utils::is_image(path) || crate::utils::is_video(path)) {
                    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    target_filesizes.entry(file_size).or_default().push(path.to_path_buf());
                }
            }
        }
        log!("Using people directory for seen-filter: {}", pd.display());
    }

    target_images_with_size.sort_by(|a, b| a.0.cmp(&b.0));
    let mut query_people: Vec<PersonInfo> = Vec::new();

    if !target_images_with_size.is_empty() {
        let current_hash = compute_target_hash(&target_images_with_size);

        let mut cache_hit = false;
        if let Some(cache) = load_target_cache() {
            if cache.hash == current_hash {
                log!("Using cached target features ({} entries).", cache.entries.len());
                query_people = cache.entries;
                cache_hit = true;
            }
        }

        if !cache_hit {
            log!("Extracting features from {} target images (cross-image majority vote)...",
                 target_images_with_size.len());

            struct TargetCandidate {
                face_bbox: [f32; 4],
                face_score: f32,
                face_embedding: Vec<f32>,
                debug_img: image::DynamicImage,
            }

            // Pass 1: detect all faces, extract embeddings
            let per_image_candidates: Vec<Vec<TargetCandidate>> = target_images_with_size
                .par_iter()
                .map(|(query_path, _)| {
                    let path_to_load = if crate::utils::is_video(query_path) {
                        ensure_video_thumbnail(query_path, &detector).unwrap_or_else(|_| query_path.clone())
                    } else {
                        query_path.clone()
                    };

                    if let Ok(img) = crate::utils::load_image_robustly(&path_to_load) {
                        let initial_dets = detector.detect(&img).unwrap_or_default();
                        let (face_dets, rotated_img) = correct_rotation(&detector, &img, initial_dets);
                        let alignment_img = rotated_img.as_ref().unwrap_or(&img);

                        let mut candidates = Vec::new();
                        for fd in face_dets {
                            let face_crop = if fd.landmarks.len() == 5 {
                                let lmks: [_; 5] = [
                                    fd.landmarks[0], fd.landmarks[1], fd.landmarks[2],
                                    fd.landmarks[3], fd.landmarks[4],
                                ];
                                align_face(alignment_img, &lmks)
                            } else {
                                let fx = fd.bbox[0].max(0.0) as u32;
                                let fy = fd.bbox[1].max(0.0) as u32;
                                let fw = (fd.bbox[2] - fd.bbox[0]).max(1.0) as u32;
                                let fh = (fd.bbox[3] - fd.bbox[1]).max(1.0) as u32;
                                alignment_img.crop_imm(fx, fy, fw, fh)
                            };
                            if let Ok(emb) = recognizer.recognize(&face_crop) {
                                candidates.push(TargetCandidate {
                                    face_bbox: fd.bbox,
                                    face_score: fd.score,
                                    face_embedding: emb,
                                    debug_img: face_crop,
                                });
                            }
                        }
                        candidates
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Pass 2: Global Clustering to find the Dominant Identity
            // Instead of picking a "winner" per image, we pool ALL faces and find the largest cluster.
            // This is robust against images that contain ONLY bystanders (which would be false positives otherwise).
            
            // Flatten per_image_candidates into a flat list, but keep track of source image index
            struct FlatCandidate {
                image_idx: usize,
                face: TargetCandidate,
            }

            let mut all_candidates: Vec<FlatCandidate> = Vec::new();
            for (idx, candidates) in per_image_candidates.into_iter().enumerate() {
                for c in candidates {
                    all_candidates.push(FlatCandidate { image_idx: idx, face: c });
                }
            }

            let total_faces = all_candidates.len();

            if total_faces == 0 {
                anyhow::bail!("Could not detect any faces in the provided target images.");
            }

            // Calculate "neighbor count" for every face
            // A face has a neighbor if similarity > 0.6
            // The face with the most neighbors is the "Centroid".
            let mut neighbor_counts = vec![0; total_faces];
            const CLUSTER_SIM_THRESHOLD: f32 = 0.6;

            // Only parallelize the outer loop if we have many faces (n^2 complexity)
            // For < 1000 faces, single thread is instant.
            for i in 0..total_faces {
                for j in 0..total_faces {
                    if i == j { continue; }
                    let sim = cosine_similarity(&all_candidates[i].face.face_embedding, &all_candidates[j].face.face_embedding);
                    if sim > CLUSTER_SIM_THRESHOLD {
                        neighbor_counts[i] += 1;
                    }
                }
            }

            // Find the centroid (face with max neighbors)
            // If multiple faces have the same max count, picking the first one is fine as they are likely similar.
            let (best_idx, max_neighbors) = neighbor_counts.iter().enumerate()
                .max_by_key(|&(_, &count)| count)
                .unwrap();
            
            let centroid_embedding = all_candidates[best_idx].face.face_embedding.clone();
            log!("Identified dominant identity from {} total faces (centroid has {} neighbors).", 
                 total_faces, max_neighbors);

            // Pass 3: Filter - Keep ONLY faces that match the centroid
            // Higher threshold = stricter, rejects more bystanders but may lose target variations.
            log!("Using target rejection threshold: {:.2}", filter_threshold);
            
            let mut final_people = Vec::new();
            let mut rejected_count = 0;

            // Prepare debug output
            let debug_dir = PathBuf::from("output").join("debug_targets");
            if debug_dir.exists() {
                let _ = fs::remove_dir_all(&debug_dir);
            }
            let _ = fs::create_dir_all(&debug_dir);

            let debug_rejected_dir = PathBuf::from("output").join("debug_rejected");
            if debug_rejected_dir.exists() {
                let _ = fs::remove_dir_all(&debug_rejected_dir);
            }
            let _ = fs::create_dir_all(&debug_rejected_dir);

            for (idx, candidate) in all_candidates.into_iter().enumerate() {
                let sim = cosine_similarity(&candidate.face.face_embedding, &centroid_embedding);

                let (orig_path, _) = &target_images_with_size[candidate.image_idx];
                let stem = orig_path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("face_{}", idx));

                if sim > filter_threshold {
                    let debug_path = debug_dir.join(format!("{}_face_{}.png", stem, idx));
                    let _ = candidate.face.debug_img.save(&debug_path);

                    final_people.push(PersonInfo {
                        face_bbox: candidate.face.face_bbox,
                        face_score: candidate.face.face_score,
                        face_embedding: candidate.face.face_embedding,
                    });
                } else {
                    let debug_path = debug_rejected_dir.join(format!("{}_face_{}_sim{:.3}.png", stem, idx, sim));
                    let _ = candidate.face.debug_img.save(&debug_path);
                    rejected_count += 1;
                }
            }

            log!("Kept {} faces matching the dominant identity. Rejected {} bystanders/outliers.", 
                 final_people.len(), rejected_count);

            if final_people.is_empty() {
                // If we filtered everything (including the centroid itself?), something is wrong.
                // But centroid always matches itself with sim=1.0, so final_people cannot be empty.
                anyhow::bail!("Clustering logic error: Dominant identity lost during filtering.");
            }

            // Deduplicate
            let original_count = final_people.len();
            let mut deduped: Vec<PersonInfo> = Vec::new();
            for info in final_people {
                let is_dup = deduped.iter().any(|existing| {
                    cosine_similarity(&existing.face_embedding, &info.face_embedding) > 0.95
                });
                if !is_dup {
                    deduped.push(info);
                }
            }
            let pruned = original_count - deduped.len();
            if pruned > 0 {
                log!("Pruned {} near-duplicate faces from target gallery.", pruned);
            }
            query_people = deduped;

            save_target_cache(&TargetCache { hash: current_hash, entries: query_people.clone() });
        }
    }

    log!("Scanning directory for files...");
    let all_paths: Vec<PathBuf> = WalkDir::new(&input)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            crate::utils::is_image(e.path()) || crate::utils::is_video(e.path())
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if all_paths.is_empty() {
        anyhow::bail!("No images or videos found in input directory.");
    }

    let mut db = load_database();
    log!("Database loaded: {} entries cached", db.images.len());

    // Filter out already processed files, but KEEP videos that are missing thumbnails
    let mut images_to_process = Vec::new();
    let mut videos_to_process = Vec::new();
    let mut backfill_count = 0usize;

    for p in all_paths {
        let path_str = p.to_string_lossy().to_string();
        let is_video = crate::utils::is_video(&p);

        if !db.images.contains_key(&path_str) {
            if is_video {
                videos_to_process.push(p);
            } else {
                images_to_process.push(p);
            }
        } else if is_video {
            // Smart Backfill: If video is in DB but missing thumbnail, re-process it
            let thumb_path = crate::utils::get_video_thumbnail_path(&p);
            if !thumb_path.exists() {
                videos_to_process.push(p);
                backfill_count += 1;
            }
        }
    }

    // --- Process Images ---
    if !images_to_process.is_empty() {
        log!("Extracting faces from {} new images...", images_to_process.len());

        let db_arc = Arc::new(Mutex::new(db));
        let batch_size = 50;
        let batches: Vec<Vec<PathBuf>> = images_to_process.chunks(batch_size)
            .map(|c| c.to_vec())
            .collect();
        let total_batches = batches.len();

        // --- Pipelined batch processing ---
        // Producer thread decodes+preprocesses batch N+1 on CPU while
        // consumer thread runs GPU inference for batch N.

        type DecodedBatch = Vec<(
            String,                                              // path
            image::DynamicImage,                                 // decoded image
            crate::face::FacePreprocessed,                       // face detection tensor
        )>;

        let (send_decoded, recv_decoded) = bounded::<(usize, DecodedBatch)>(2);

        let det = Arc::clone(&detector);
        let rec = Arc::clone(&recognizer);
        let db_arc2 = Arc::clone(&db_arc);
        let tx2 = tx.clone();

        std::thread::scope(|s| {
            // --- Producer thread: parallel decode + preprocess on CPU ---
            let db_arc3 = Arc::clone(&db_arc);
            s.spawn(move || {
                for (batch_idx, batch) in batches.into_iter().enumerate() {
                    // Record files that fail to decode so they aren't retried every run
                    let failed_paths: Vec<String> = batch
                        .par_iter()
                        .filter_map(|path| {
                            match crate::utils::load_image_robustly(path) {
                                Ok(_) => None,
                                Err(_) => Some(path.to_string_lossy().to_string()),
                            }
                        })
                        .collect();

                    if !failed_paths.is_empty() {
                        let mut db_guard = db_arc3.lock().unwrap();
                        for path in failed_paths {
                            db_guard.images.entry(path).or_insert_with(Vec::new);
                        }
                    }

                    let decoded: DecodedBatch = batch
                        .par_iter()
                        .filter_map(|path| {
                            let img = crate::utils::load_image_robustly(path).ok()?;
                            let face_pre = crate::face::FaceDetector::preprocess_image(&img);
                            Some((path.to_string_lossy().to_string(), img, face_pre))
                        })
                        .collect();

                    if send_decoded.send((batch_idx, decoded)).is_err() {
                        break;
                    }
                }
                drop(send_decoded);
            });

            // --- Consumer: GPU inference + postprocess + save ---
            for (batch_idx, decoded) in recv_decoded {
                let _ = tx2.send(UiMessage::Log(format!("Processing batch {}/{}...", batch_idx + 1, total_batches)));

                let face_tensors: Vec<&crate::face::FacePreprocessed> = decoded.iter()
                    .map(|(_, _, fp)| fp)
                    .collect();

                // Run face detection (GPU)
                let face_raw_results: Vec<Option<crate::face::RawFaceOutput>> = face_tensors.iter()
                    .map(|ft| det.infer(ft).ok())
                    .collect();

                struct PendingFace {
                    path_idx: usize,
                    bbox: [f32; 4],
                    score: f32,
                    crop_idx: usize,
                }

                let mut all_face_crops = Vec::new();
                let mut pending_faces = Vec::new();

                // Postprocess detections and collect crops
                for (path_idx, ((_path_str, img, face_pre), face_raw))
                    in decoded.iter()
                        .zip(face_raw_results.iter())
                        .enumerate()
                {
                    let face_detections = match face_raw {
                        Some(raw) => crate::face::FaceDetector::postprocess(
                            raw, face_pre.scale, face_pre.orig_width, face_pre.orig_height
                        ).unwrap_or_default(),
                        None => Vec::new(),
                    };

                    // Rotation correction: re-detect on rotated image if eye-line tilt is excessive
                    let (face_detections, rotated_img) = correct_rotation(&det, img, face_detections);
                    let alignment_img = rotated_img.as_ref().unwrap_or(img);

                    for fd in face_detections {
                        let face_crop = if fd.landmarks.len() == 5 {
                            let lmks: [_; 5] = [
                                fd.landmarks[0], fd.landmarks[1],
                                fd.landmarks[2], fd.landmarks[3],
                                fd.landmarks[4],
                            ];
                            align_face(alignment_img, &lmks)
                        } else {
                            let fx = fd.bbox[0].max(0.0) as u32;
                            let fy = fd.bbox[1].max(0.0) as u32;
                            let fw = (fd.bbox[2] - fd.bbox[0]).max(1.0) as u32;
                            let fh = (fd.bbox[3] - fd.bbox[1]).max(1.0) as u32;
                            alignment_img.crop_imm(fx, fy, fw, fh)
                        };
                        let crop_idx = all_face_crops.len();
                        all_face_crops.push(face_crop);
                        pending_faces.push(PendingFace {
                            path_idx,
                            bbox: fd.bbox,
                            score: fd.score,
                            crop_idx,
                        });
                    }
                }

                // Batch recognition (GPU)
                let face_embeddings = if !all_face_crops.is_empty() {
                    let mut embs = Vec::with_capacity(all_face_crops.len());
                    for chunk in all_face_crops.chunks(128) {
                        if let Ok(e) = rec.recognize_batch(chunk) {
                            embs.extend(e);
                        } else {
                            embs.extend(vec![vec![0.0; 512]; chunk.len()]);
                        }
                    }
                    embs
                } else {
                    Vec::new()
                };

                // Reassemble results
                let mut batch_results: Vec<(String, Vec<PersonInfo>)> = decoded
                    .into_iter()
                    .map(|(p, _, _)| (p, Vec::new()))
                    .collect();

                for pf in pending_faces {
                    let emb = face_embeddings[pf.crop_idx].clone();
                    if emb.iter().all(|&x| x == 0.0) { continue; }

                    batch_results[pf.path_idx].1.push(PersonInfo {
                        face_bbox: pf.bbox,
                        face_score: pf.score,
                        face_embedding: emb,
                    });
                }

                // Save batch to database
                let mut db_guard = db_arc2.lock().unwrap();
                for (path, people) in batch_results {
                    db_guard.images.insert(path, people);
                }
                let _ = save_database(&mut *db_guard);
            }
        });

        drop(db_arc2);
        db = Arc::try_unwrap(db_arc).unwrap().into_inner().unwrap();
    } else {
        log!("No new images to process.");
    }

    // --- Process Videos ---
    if !videos_to_process.is_empty() {
        log!("Processing {} videos ({} new, {} backfill thumbnails)...",
            videos_to_process.len(),
            videos_to_process.len() - backfill_count,
            backfill_count);

        let db_arc = Arc::new(Mutex::new(db));
        let total_videos = videos_to_process.len();

        for (i, video_path) in videos_to_process.iter().enumerate() {
            let msg = format!("Processing video {}/{}: {}", i + 1, total_videos, video_path.file_name().unwrap_or_default().to_string_lossy());
            let _ = tx.send(UiMessage::Log(msg.clone()));
            println!("{}", msg);

            if let Ok(thumb_path) = ensure_video_thumbnail(video_path, &detector) {
                let mut db_guard = db_arc.lock().unwrap();
                let path_str = video_path.to_string_lossy().to_string();
                
                if !db_guard.images.contains_key(&path_str) {
                    let mut final_faces = Vec::new();
                    if let Ok(img) = crate::utils::load_image_robustly(&thumb_path) {
                         let dets = detector.detect(&img).unwrap_or_default();
                         let (dets, rotated) = correct_rotation(&detector, &img, dets);
                         let align_img = rotated.as_ref().unwrap_or(&img);

                         for fd in dets {
                             let crop = if fd.landmarks.len() == 5 {
                                 let lmks = [fd.landmarks[0], fd.landmarks[1], fd.landmarks[2], fd.landmarks[3], fd.landmarks[4]];
                                 align_face(align_img, &lmks)
                             } else {
                                 align_img.crop_imm(
                                     fd.bbox[0].max(0.0) as u32,
                                     fd.bbox[1].max(0.0) as u32,
                                     (fd.bbox[2]-fd.bbox[0]).max(1.0) as u32,
                                     (fd.bbox[3]-fd.bbox[1]).max(1.0) as u32
                                 )
                             };

                             if let Ok(emb) = recognizer.recognize(&crop) {
                                 final_faces.push(PersonInfo {
                                     face_bbox: fd.bbox,
                                     face_score: fd.score,
                                     face_embedding: emb,
                                 });
                             }
                         }
                    }
                    db_guard.images.insert(path_str, final_faces);
                    let _ = save_database(&mut *db_guard);
                }
            } else {
                let mut db_guard = db_arc.lock().unwrap();
                db_guard.images.entry(video_path.to_string_lossy().to_string()).or_insert_with(Vec::new);
                let _ = save_database(&mut *db_guard);
            }
        }
        
        db = Arc::try_unwrap(db_arc).unwrap().into_inner().unwrap();
    } else {
        log!("No new videos to process.");
    }

    let processed_count = AtomicUsize::new(0);

    if !query_people.is_empty() {
        log!("Gallery matching with {} target faces...", query_people.len());

        let input_prefix = input.to_string_lossy().to_string();
        let relevant_images: Vec<(&String, &Vec<PersonInfo>)> = db.images.iter()
            .filter(|(path_str, _)| path_str.starts_with(&input_prefix))
            .collect();

        log!("Searching for target across {} images...", relevant_images.len());

        // Parallel similarity matching
        let mut candidates: Vec<(PathBuf, f32)> = relevant_images
            .par_iter()
            .filter_map(|(path_str, people)| {
                let source_path = Path::new(path_str.as_str());
                if !source_path.exists() { return None; }

                let mut is_duplicate = false;
                if let Ok(metadata) = source_path.metadata() {
                    let size = metadata.len();
                    if let Some(target_paths) = target_filesizes.get(&size) {
                        if let Ok(source_bytes) = fs::read(source_path) {
                            for target_path in target_paths {
                                if let Ok(target_bytes) = fs::read(target_path) {
                                    if source_bytes == target_bytes {
                                        is_duplicate = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                if is_duplicate { return None; }

                processed_count.fetch_add(1, Ordering::Relaxed);

                if people.is_empty() { return None; }

                const TOP_K: usize = 10;
                let mut best_top_k_avg = -1.0f32;

                if !query_people.is_empty() {
                    for candidate_person in people.iter() {
                        let mut sims: Vec<f32> = query_people
                            .iter()
                            .map(|target_person| calculate_similarity(target_person, candidate_person))
                            .collect();

                        sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                        let k = TOP_K.min(sims.len());
                        let top_k_avg = sims[..k].iter().sum::<f32>() / k as f32;

                        if top_k_avg > best_top_k_avg {
                            best_top_k_avg = top_k_avg;
                        }
                    }
                }

                let dist = 1.0 - best_top_k_avg;

                if dist >= match_threshold_min && dist <= match_threshold_max {
                    Some((source_path.to_path_buf(), dist))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        log!("Finished searching! Found {} candidates ranked by similarity.", candidates.len());
        if candidates.is_empty() {
            log!("No candidates found. Threshold range: {:.3}–{:.3}. Searched {} images against {} targets.",
                 match_threshold_min, match_threshold_max,
                 processed_count.load(Ordering::Relaxed), query_people.len());
        } else {
            let best = candidates.first().map(|(_, d)| *d).unwrap_or(0.0);
            let worst = candidates.last().map(|(_, d)| *d).unwrap_or(0.0);
            log!("Distance range of matches: {:.3}–{:.3}.", best, worst);
        }

        let _ = tx.send(UiMessage::Done(processed_count.load(Ordering::Relaxed), candidates));
        return Ok(());
    } else {
         anyhow::bail!("No target faces found.");
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 400.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Facial Recognition Sorter",
        options,
        Box::new(|cc| Ok(Box::new(FaceSearchApp::new(cc)))),
    )
}

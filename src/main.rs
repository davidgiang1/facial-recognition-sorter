mod face;
mod gui;

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
    10u64.hash(&mut hasher);  // bumped: forces cache invalidation when PersonInfo schema changes
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
    dump_dir: Option<PathBuf>,
    match_threshold_min: f32,
    match_threshold_max: f32,
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
                    if path.is_file() {
                        let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                        if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp") {
                            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                            target_images_with_size.push((path.clone(), file_size));
                        }
                    }
                }
            }
        }
    }

    // Build target_filesizes "already seen" filter from dump_dir (comprehensive collection)
    if let Some(dd) = &dump_dir {
        if dd.exists() {
            for entry in WalkDir::new(dd).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                    if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp") {
                        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                        target_filesizes.entry(file_size).or_default().push(path.to_path_buf());
                    }
                }
            }
        }
        log!("Using dump directory for seen-filter: {}", dd.display());
    }

    // Also exclude photos from sibling reference directories (other people's reference photos)
    if let Some(td) = &target_dir {
        if let Some(parent) = td.parent() {
            let mut sibling_count = 0usize;
            if let Ok(siblings) = fs::read_dir(parent) {
                for sibling in siblings.filter_map(|e| e.ok()) {
                    let sibling_path = sibling.path();
                    if sibling_path.is_dir() && sibling_path != *td {
                        sibling_count += 1;
                        if let Ok(entries) = fs::read_dir(&sibling_path) {
                            for entry in entries.filter_map(|e| e.ok()) {
                                let path = entry.path();
                                if path.is_file() {
                                    let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                                    if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp") {
                                        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                                        target_filesizes.entry(file_size).or_default().push(path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if sibling_count > 0 {
                log!("Auto-excluding photos from {} sibling directories.", sibling_count);
            }
        }
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
            let mut per_image_candidates: Vec<Vec<TargetCandidate>> = Vec::new();

            for (query_path, _) in target_images_with_size.iter() {
                if let Ok(img) = image::open(query_path) {
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
                    per_image_candidates.push(candidates);
                } else {
                    per_image_candidates.push(Vec::new());
                }
            }

            // Pass 2+3: score by cross-image support, select winner per image
            const CROSS_SIM_THRESHOLD: f32 = 0.6;
            let n_images = per_image_candidates.len();

            struct ExtractedPerson {
                image_idx: usize,
                info: PersonInfo,
                debug_img: image::DynamicImage,
            }
            let mut all_people: Vec<ExtractedPerson> = Vec::new();

            for img_idx in 0..n_images {
                let candidates = &per_image_candidates[img_idx];
                if candidates.is_empty() { continue; }

                let winner_idx = if n_images == 1 {
                    // Only 1 image — no cross-comparison possible, fall back to largest face
                    candidates.iter().enumerate()
                        .max_by(|(_, a), (_, b)| {
                            let aa = (a.face_bbox[2] - a.face_bbox[0]) * (a.face_bbox[3] - a.face_bbox[1]);
                            let ab = (b.face_bbox[2] - b.face_bbox[0]) * (b.face_bbox[3] - b.face_bbox[1]);
                            aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                } else {
                    let scores: Vec<usize> = candidates.iter().map(|c| {
                        (0..n_images)
                            .filter(|&j| j != img_idx)
                            .filter(|&j| {
                                per_image_candidates[j].iter().any(|other| {
                                    let sim = cosine_similarity(&c.face_embedding, &other.face_embedding);
                                    sim > CROSS_SIM_THRESHOLD
                                })
                            })
                            .count()
                    }).collect();

                    let max_score = *scores.iter().max().unwrap_or(&0);
                    // Tiebreak: largest face box area
                    scores.iter().enumerate()
                        .filter(|&(_, s)| *s == max_score)
                        .max_by(|(ia, _), (ib, _)| {
                            let area = |i: usize| {
                                let b = candidates[i].face_bbox;
                                (b[2] - b[0]) * (b[3] - b[1])
                            };
                            area(*ia).partial_cmp(&area(*ib)).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                };

                let winner = &per_image_candidates[img_idx][winner_idx];

                all_people.push(ExtractedPerson {
                    image_idx: img_idx,
                    info: PersonInfo {
                        face_bbox: winner.face_bbox,
                        face_score: winner.face_score,
                        face_embedding: winner.face_embedding.clone(),
                    },
                    debug_img: winner.debug_img.clone(),
                });
            }

            if all_people.is_empty() {
                anyhow::bail!("Could not detect any faces in the provided target images.");
            }

            log!("Extracted {} faces from {} images. Saving debug crops...",
                 all_people.len(), target_images_with_size.len());

            let debug_dir = PathBuf::from("output").join("debug_targets");
            if debug_dir.exists() {
                let _ = fs::remove_dir_all(&debug_dir);
            }
            let _ = fs::create_dir_all(&debug_dir);

            for (idx, p) in all_people.iter().enumerate() {
                let (orig_path, _) = &target_images_with_size[p.image_idx];
                let stem = orig_path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("face_{}", idx));
                let debug_path = debug_dir.join(format!("{}_face.png", stem));
                let _ = p.debug_img.save(&debug_path);
            }

            let all_infos: Vec<PersonInfo> = all_people.into_iter().map(|p| p.info).collect();
            let original_count = all_infos.len();
            let mut deduped: Vec<PersonInfo> = Vec::new();
            for info in all_infos {
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

    log!("Scanning directory for images...");
    let image_paths: Vec<PathBuf> = WalkDir::new(&input)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| {
                    let ext = ext.to_string_lossy().to_lowercase();
                    matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp")
                })
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if image_paths.is_empty() {
        anyhow::bail!("No images found in input directory.");
    }

    let mut db = load_database();
    let to_process: Vec<PathBuf> = image_paths
        .into_iter()
        .filter(|p| !db.images.contains_key(&p.to_string_lossy().to_string()))
        .collect();

    if !to_process.is_empty() {
        log!("Extracting faces from {} new images...", to_process.len());

        let db_arc = Arc::new(Mutex::new(db));
        let batch_size = 50;
        let batches: Vec<Vec<PathBuf>> = to_process.chunks(batch_size)
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
            s.spawn(|| {
                for (batch_idx, batch) in batches.into_iter().enumerate() {
                    let decoded: DecodedBatch = batch
                        .par_iter()
                        .filter_map(|path| {
                            let img = image::open(path).ok()?;
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
        log!("No new images to process (database up to date).");
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

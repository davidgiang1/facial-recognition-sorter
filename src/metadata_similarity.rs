//! Ranks photos in the input directory by how likely they are to contain a
//! person whose face the recognition pipeline could not see (occluded, turned
//! away), using proximity to already-confirmed photos of that person on
//! several metadata axes: filename sequence number, capture time, pixel
//! dimensions, camera, and GPS location.
//!
//! See the "Similar Timing" tab in the GUI, which is the only caller of
//! `rank_by_metadata`.
//!
//! # Why the weights look the way they do
//!
//! The axes were measured against a hand-confirmed person folder (248 photos)
//! over a 53k-candidate library, scoring each axis by lift - how much more
//! often it fires on that person's photos than on the library at large:
//!
//! | signal                                          | lift  |
//! |-------------------------------------------------|-------|
//! | sequence gap <= 1, same kind, time-consistent    | 194x  |
//! | sequence gap <= 3, same kind, time-consistent    | 105x  |
//! | exact pixel dimensions of some anchor            | 6.1x  |
//! | capture time within 1 minute of an anchor        | 3.8x  |
//! | capture time within 1 minute, *no* sequence tie  | ~0x   |
//!
//! The filename sequence is far and away the strongest axis and timestamp
//! proximity on its own is close to worthless, so sequence carries the
//! largest weight and is allowed to keep a candidate that the time window
//! would otherwise discard. Dimensions stand in for "same device" in
//! libraries of screenshots and re-shared images, where EXIF `Model` is
//! stripped from every file and the camera axis can never fire.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Bump whenever `PhotoMeta`'s shape - or the logic that fills it - changes,
/// so a cache written by an older build is discarded instead of silently
/// feeding stale values into the ranking.
const META_CACHE_VERSION: u32 = 1;

/// Exponential time-decay constant, in seconds: score = exp(-delta / this).
/// Deliberately sharp - it only rewards "same moment".
const TIME_DECAY_SECS: f64 = 300.0;
/// A second, much slower time decay. `TIME_DECAY_SECS` is so sharp that
/// anything past roughly half an hour scores an indistinguishable zero, which
/// leaves whole blocks of candidates tied on the time axis; this term keeps
/// "same afternoon" ahead of "same decade".
const SLOW_TIME_DECAY_SECS: f64 = 6.0 * 3600.0;
/// Exponential distance-decay constant, in km, for the GPS signal.
const GPS_DECAY_KM: f64 = 0.5;
/// Exponential decay constant for filename-sequence-number gap.
const SEQUENCE_DECAY: f64 = 5.0;

const SEQUENCE_WEIGHT: f32 = 0.40;
const TIME_WEIGHT: f32 = 0.20;
const SLOW_TIME_WEIGHT: f32 = 0.07;
const DIMENSION_WEIGHT: f32 = 0.23;
const CAMERA_WEIGHT: f32 = 0.05;
const GPS_WEIGHT: f32 = 0.05;

/// Credit for a candidate whose pixel dimensions only share an anchor's
/// aspect ratio rather than matching it exactly (a resized export of the
/// same source).
const ASPECT_ONLY_CREDIT: f32 = 0.5;
/// How close two width/height ratios must be to count as the same aspect.
const ASPECT_TOLERANCE: f64 = 0.02;

/// A candidate this far (or less) from an anchor in filename sequence is kept
/// even when its timestamp falls outside the caller's time window.
const SEQUENCE_RESCUE_GAP: u64 = 25;
/// A sequence tie is only believed if the two timestamps are consistent with
/// it: within `max(this, gap * SEQUENCE_CONSISTENCY_PER_STEP_SECS)`. Camera
/// rolls reuse the same `IMG_####` numbers across devices and years, so
/// without this an `IMG_1711` from 2021 "matches" an `IMG_1714` from 2024.
const SEQUENCE_CONSISTENCY_FLOOR_SECS: i64 = 6 * 3600;
const SEQUENCE_CONSISTENCY_PER_STEP_SECS: i64 = 6 * 3600;

/// Confidence multiplier applied to a timestamp that only came from file
/// mtime (weakest source - see `TimeSource`).
const MTIME_BASE_CONFIDENCE: f32 = 0.6;
/// Confidence used instead when mtime looks like a bulk-transfer artifact
/// (see `MTIME_COLLISION_THRESHOLD`) rather than a real capture time.
const MTIME_COLLISION_CONFIDENCE: f32 = 0.05;
/// Bucket width, in seconds, used to detect "many files share almost the
/// same mtime" (a bulk copy/sync, not organic photography).
const MTIME_COLLISION_BUCKET_SECS: i64 = 600;
/// A bucket with more than this many mtime-sourced files is treated as a
/// bulk-transfer artifact.
const MTIME_COLLISION_THRESHOLD: u32 = 1000;

/// Offset between the Mac/QuickTime epoch (1904-01-01) and Unix epoch
/// (1970-01-01), in seconds - used to decode MP4/MOV `mvhd` creation time.
const MAC_EPOCH_OFFSET: i64 = 2_082_844_800;

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TimeSource {
    /// EXIF `DateTimeOriginal`/`DateTime` - a real camera-written capture time.
    Exif,
    /// A timestamp parsed out of the filename itself (e.g. an iOS
    /// screenshot's "Screenshot 2024-01-15 at 3.45.12 PM.png"). Trusted as
    /// much as EXIF: it's written on-device before any transfer can corrupt it.
    FilenameTimestamp,
    /// The `mvhd` box's `creation_time` field in an MP4/MOV container.
    VideoAtom,
    /// Filesystem modified-time - the only source with no origin guarantee.
    /// A bulk copy/sync can stamp thousands of unrelated files with the same
    /// mtime, so this is deliberately the least-trusted source.
    Mtime,
}

impl TimeSource {
    fn base_confidence(self) -> f32 {
        match self {
            TimeSource::Mtime => MTIME_BASE_CONFIDENCE,
            _ => 1.0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct PhotoMeta {
    secs: i64,
    time_source: TimeSource,
    camera: Option<String>,
    /// (latitude, longitude) in decimal degrees.
    gps: Option<(f64, f64)>,
    /// (sequence family, trailing number) parsed from the filename, e.g.
    /// "IMG_1234.jpg" -> ("img|jpg", 1234). See `parse_filename_sequence`.
    sequence: Option<(String, u64)>,
    /// Pixel dimensions, read from the header without decoding the image.
    dimensions: Option<(u32, u32)>,
}

struct Located {
    path: PathBuf,
    meta: PhotoMeta,
}

#[derive(Clone)]
pub struct MetaSimCandidate {
    pub path: PathBuf,
    /// Final blended score, roughly in [0, 1] (can exceed 1 slightly when
    /// every signal lines up perfectly).
    pub score: f32,
    /// The confirmed photo this candidate scored best against.
    pub anchor: PathBuf,
    pub delta_secs: i64,
    /// Combined trust in `delta_secs` (candidate confidence x anchor
    /// confidence) - low when either side's time only came from a
    /// bulk-transfer-tainted mtime. Surfaced so the UI can flag it.
    pub time_confidence: f32,
    pub same_camera: bool,
    pub gps_km: Option<f64>,
    pub sequence_gap: Option<u64>,
    /// How the candidate's pixel dimensions relate to the anchor's.
    pub dimension_match: DimensionMatch,
    /// True when only the sequence tie kept this candidate - its timestamp
    /// fell outside the caller's window. Surfaced so the UI can say so.
    pub rescued_by_sequence: bool,
}

/// How a candidate's pixel dimensions compare to its anchor's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimensionMatch {
    /// Same width and height - almost always the same capture device.
    Exact,
    /// Same shape, different size: a resized export of the same source.
    AspectOnly,
    /// Both known, and different.
    Different,
    /// At least one side's dimensions could not be read (videos, RAW, HEIC).
    Unknown,
}

impl DimensionMatch {
    fn score(self) -> f32 {
        match self {
            DimensionMatch::Exact => 1.0,
            DimensionMatch::AspectOnly => ASPECT_ONLY_CREDIT,
            DimensionMatch::Different | DimensionMatch::Unknown => 0.0,
        }
    }

    fn compare(a: Option<(u32, u32)>, b: Option<(u32, u32)>) -> Self {
        let (Some(a), Some(b)) = (a, b) else { return DimensionMatch::Unknown };
        if a == b {
            return DimensionMatch::Exact;
        }
        if a.1 == 0 || b.1 == 0 {
            return DimensionMatch::Different;
        }
        let ratio_a = a.0 as f64 / a.1 as f64;
        let ratio_b = b.0 as f64 / b.1 as f64;
        if (ratio_a - ratio_b).abs() < ASPECT_TOLERANCE {
            DimensionMatch::AspectOnly
        } else {
            DimensionMatch::Different
        }
    }
}

impl MetaSimCandidate {
    /// Human-readable breakdown for a tooltip.
    pub fn explain(&self) -> String {
        let anchor_name = self
            .anchor
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut parts = vec![format!("{} from {}", humanize_delta(self.delta_secs), anchor_name)];
        if let Some(gap) = self.sequence_gap {
            parts.push(match gap {
                0 => "same filename sequence number".to_string(),
                _ => format!("{} apart in filename sequence", gap),
            });
        }
        if self.rescued_by_sequence {
            parts.push("outside the time window, kept for its filename sequence".to_string());
        }
        if self.time_confidence < 0.5 {
            parts.push("timestamp looks like a bulk-transfer date, weighted down".to_string());
        }
        match self.dimension_match {
            DimensionMatch::Exact => parts.push("identical pixel dimensions".to_string()),
            DimensionMatch::AspectOnly => parts.push("same aspect ratio".to_string()),
            DimensionMatch::Different | DimensionMatch::Unknown => {}
        }
        if self.same_camera {
            parts.push("same camera".to_string());
        }
        if let Some(km) = self.gps_km {
            parts.push(format!("{:.2} km apart", km));
        }
        format!("{:.0}% match - {}", self.score.min(1.0) * 100.0, parts.join(", "))
    }
}

/// Rank `candidates` by metadata proximity to `anchors` (photos already
/// confirmed to contain the target person). Each candidate is scored against
/// every anchor and keeps whichever anchor scores best overall, not just the
/// nearest in time - a candidate can be a weak time match but a strong GPS or
/// filename-sequence match against a different anchor.
///
/// A candidate survives if it lands within `window_secs` of some anchor *or*
/// sits within `SEQUENCE_RESCUE_GAP` of one in a time-consistent filename
/// sequence. The window used to be the only way in, which silently discarded
/// the single strongest signal available: a screenshot named `IMG_1711.PNG`
/// taken ten hours after a confirmed `IMG_1714.PNG` was thrown away before it
/// could ever be scored, no matter how many pages the user paged through.
pub fn rank_by_metadata(
    anchors: &[PathBuf],
    candidates: Vec<PathBuf>,
    window_secs: i64,
) -> Vec<MetaSimCandidate> {
    rank_by_metadata_cached(anchors, candidates, window_secs, &meta_cache_file())
}

/// The body of `rank_by_metadata`, with the cache location injected so tests
/// can point it at a scratch file instead of clobbering the real one.
fn rank_by_metadata_cached(
    anchors: &[PathBuf],
    candidates: Vec<PathBuf>,
    window_secs: i64,
    cache_path: &Path,
) -> Vec<MetaSimCandidate> {
    let cache = load_meta_cache(cache_path);
    let anchor_entries = read_all_meta(anchors.to_vec(), &cache);
    if anchor_entries.is_empty() {
        return Vec::new();
    }
    let cand_entries = read_all_meta(candidates, &cache);

    let mut touched = anchor_entries.clone();
    touched.extend(cand_entries.iter().cloned());
    save_meta_cache(cache_path, &touched);
    drop(touched);
    drop(cache);

    let into_located = |entries: Vec<(PathBuf, CacheEntry)>| -> Vec<Located> {
        entries.into_iter().map(|(path, e)| Located { path, meta: e.meta }).collect()
    };
    let anchor_data = into_located(anchor_entries);
    let cand_data = into_located(cand_entries);

    // Detect mtime values that hundreds/thousands of files share within a
    // tight window - the signature of a bulk copy/sync, not real photography.
    let mut bucket_counts: HashMap<i64, u32> = HashMap::new();
    for meta in anchor_data.iter().chain(cand_data.iter()).map(|l| &l.meta) {
        if meta.time_source == TimeSource::Mtime {
            *bucket_counts.entry(meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS)).or_insert(0) += 1;
        }
    }
    let confidence_of = |meta: &PhotoMeta| -> f32 {
        if meta.time_source != TimeSource::Mtime {
            return 1.0;
        }
        let bucket = meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS);
        if bucket_counts.get(&bucket).copied().unwrap_or(0) > MTIME_COLLISION_THRESHOLD {
            MTIME_COLLISION_CONFIDENCE
        } else {
            meta.time_source.base_confidence()
        }
    };

    let mut results: Vec<MetaSimCandidate> = cand_data
        .par_iter()
        .filter_map(|cand| {
            let cand_conf = confidence_of(&cand.meta);
            let mut best: Option<MetaSimCandidate> = None;

            for anchor in &anchor_data {
                let delta_secs = (cand.meta.secs - anchor.meta.secs).abs();

                // A shared filename-sequence number is only evidence if the two
                // timestamps could plausibly belong to the same run of shots.
                let sequence_gap = match (&cand.meta.sequence, &anchor.meta.sequence) {
                    (Some((cp, cn)), Some((ap, an))) if cp.eq_ignore_ascii_case(ap) => {
                        let gap = cn.abs_diff(*an);
                        let allowance = SEQUENCE_CONSISTENCY_FLOOR_SECS
                            .max(gap.saturating_mul(SEQUENCE_CONSISTENCY_PER_STEP_SECS as u64) as i64);
                        (delta_secs <= allowance).then_some(gap)
                    }
                    _ => None,
                };

                let within_window = delta_secs <= window_secs;
                let rescued_by_sequence = !within_window
                    && sequence_gap.map(|g| g <= SEQUENCE_RESCUE_GAP).unwrap_or(false);
                if !within_window && !rescued_by_sequence {
                    continue;
                }

                let combined_conf = cand_conf * confidence_of(&anchor.meta);
                let time_score = (-(delta_secs as f64) / TIME_DECAY_SECS).exp() as f32 * combined_conf;
                let slow_time_score =
                    (-(delta_secs as f64) / SLOW_TIME_DECAY_SECS).exp() as f32 * combined_conf;

                let same_camera = matches!(
                    (&cand.meta.camera, &anchor.meta.camera),
                    (Some(a), Some(b)) if a == b
                );

                let gps_km = match (cand.meta.gps, anchor.meta.gps) {
                    (Some(a), Some(b)) => Some(haversine_km(a, b)),
                    _ => None,
                };
                let gps_score = gps_km.map(|km| (-km / GPS_DECAY_KM).exp() as f32).unwrap_or(0.0);

                let sequence_score = sequence_gap.map(|g| (-(g as f64) / SEQUENCE_DECAY).exp() as f32).unwrap_or(0.0);

                let dimension_match =
                    DimensionMatch::compare(cand.meta.dimensions, anchor.meta.dimensions);

                let score = sequence_score * SEQUENCE_WEIGHT
                    + time_score * TIME_WEIGHT
                    + slow_time_score * SLOW_TIME_WEIGHT
                    + dimension_match.score() * DIMENSION_WEIGHT
                    + if same_camera { CAMERA_WEIGHT } else { 0.0 }
                    + gps_score * GPS_WEIGHT;

                if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
                    best = Some(MetaSimCandidate {
                        path: cand.path.clone(),
                        score,
                        anchor: anchor.path.clone(),
                        delta_secs,
                        time_confidence: combined_conf,
                        same_camera,
                        gps_km,
                        sequence_gap,
                        dimension_match,
                        rescued_by_sequence,
                    });
                }
            }
            best
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// A `PhotoMeta` alongside the stat fields that decide whether it is still
/// valid: if the file's size and mtime are unchanged, its metadata is too.
#[derive(Clone, Serialize, Deserialize)]
struct CacheEntry {
    size: u64,
    mtime_secs: i64,
    meta: PhotoMeta,
}

#[derive(Serialize, Deserialize)]
struct MetaCache {
    version: u32,
    entries: HashMap<PathBuf, CacheEntry>,
}

fn meta_cache_file() -> PathBuf {
    crate::get_app_data_dir().join("metasim_cache.bin")
}

/// Previously-read metadata, or an empty map if there is no cache, it can't
/// be read, or it was written by a build with different extraction logic.
fn load_meta_cache(cache_path: &Path) -> HashMap<PathBuf, CacheEntry> {
    let Ok(bytes) = std::fs::read(cache_path) else { return HashMap::new() };
    match bincode::deserialize::<MetaCache>(&bytes) {
        Ok(cache) if cache.version == META_CACHE_VERSION => cache.entries,
        _ => HashMap::new(),
    }
}

/// Write back only what this scan actually touched, so the cache tracks the
/// current library instead of growing forever as files come and go.
fn save_meta_cache(cache_path: &Path, entries: &[(PathBuf, CacheEntry)]) {
    let cache = MetaCache {
        version: META_CACHE_VERSION,
        entries: entries
            .iter()
            // serde renders a path as a str, which a non-UTF-8 name would
            // fail; dropping those keeps one odd filename from costing the
            // whole cache.
            .filter(|(path, _)| path.to_str().is_some())
            .map(|(path, entry)| (path.clone(), entry.clone()))
            .collect(),
    };
    if let Ok(bytes) = bincode::serialize(&cache) {
        let _ = std::fs::write(cache_path, bytes);
    }
}

/// Read every path's metadata, reusing cached values for files that haven't
/// changed. Parsing EXIF and image headers dominates a scan, and across
/// repeat scans almost nothing has changed.
fn read_all_meta(
    paths: Vec<PathBuf>,
    cache: &HashMap<PathBuf, CacheEntry>,
) -> Vec<(PathBuf, CacheEntry)> {
    paths
        .into_par_iter()
        .filter_map(|path| {
            let stat = std::fs::metadata(&path).ok()?;
            let size = stat.len();
            let mtime_secs = stat
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            if let Some(hit) = cache.get(&path) {
                if hit.size == size && hit.mtime_secs == mtime_secs {
                    return Some((path, hit.clone()));
                }
            }
            let meta = read_photo_meta(&path)?;
            Some((path, CacheEntry { size, mtime_secs, meta }))
        })
        .collect()
}

fn read_photo_meta(path: &Path) -> Option<PhotoMeta> {
    let mut secs = None;
    let mut source = TimeSource::Mtime;
    let mut camera = None;
    let mut gps = None;

    if crate::utils::is_video(path) {
        if is_isobmff_video(path) {
            secs = read_mp4_creation_time(path);
            if secs.is_some() {
                source = TimeSource::VideoAtom;
            }
        }
    } else if let Some(exif) = read_exif(path) {
        secs = exif_capture_secs(&exif);
        if secs.is_some() {
            source = TimeSource::Exif;
        }
        camera = exif_camera(&exif);
        gps = read_gps(&exif);
    }

    if secs.is_none() {
        secs = parse_filename_timestamp(path);
        if secs.is_some() {
            source = TimeSource::FilenameTimestamp;
        }
    }
    if secs.is_none() {
        secs = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);
        source = TimeSource::Mtime;
    }

    let sequence = parse_filename_sequence(path);
    let dimensions = read_dimensions(path);
    secs.map(|secs| PhotoMeta { secs, time_source: source, camera, gps, sequence, dimensions })
}

/// Pixel dimensions straight from the file header - no full decode, so this
/// stays cheap enough to run over a whole input directory. Returns `None` for
/// anything the `image` crate can't parse a header for (videos, RAW, HEIC),
/// which the dimension axis then simply sits out.
fn read_dimensions(path: &Path) -> Option<(u32, u32)> {
    if crate::utils::is_video(path) {
        return None;
    }
    // Sniff the real format rather than trusting the extension: this library
    // is full of files saved with the "wrong" suffix.
    image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

fn is_isobmff_video(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("mp4" | "mov" | "m4v")
    )
}

fn read_exif(path: &Path) -> Option<exif::Exif> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    exif::Reader::new().read_from_container(&mut reader).ok()
}

fn exif_capture_secs(exif: &exif::Exif) -> Option<i64> {
    exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)
        .or_else(|| exif.get_field(exif::Tag::DateTime, exif::In::PRIMARY))
        .and_then(|f| ascii_field(&f.value))
        .and_then(|s| exif::DateTime::from_ascii(s.as_bytes()).ok())
        .map(|dt| exif_datetime_to_secs(&dt))
}

fn exif_camera(exif: &exif::Exif) -> Option<String> {
    exif.get_field(exif::Tag::Model, exif::In::PRIMARY)
        .and_then(|f| ascii_field(&f.value))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_gps(exif: &exif::Exif) -> Option<(f64, f64)> {
    let lat = gps_coord(exif, exif::Tag::GPSLatitude, exif::Tag::GPSLatitudeRef, 'S')?;
    let lon = gps_coord(exif, exif::Tag::GPSLongitude, exif::Tag::GPSLongitudeRef, 'W')?;
    Some((lat, lon))
}

fn gps_coord(exif: &exif::Exif, value_tag: exif::Tag, ref_tag: exif::Tag, negative_ref: char) -> Option<f64> {
    let field = exif.get_field(value_tag, exif::In::PRIMARY)?;
    let exif::Value::Rational(dms) = &field.value else { return None };
    if dms.len() < 3 {
        return None;
    }
    let degrees = dms[0].to_f64() + dms[1].to_f64() / 60.0 + dms[2].to_f64() / 3600.0;

    let negative = exif
        .get_field(ref_tag, exif::In::PRIMARY)
        .and_then(|f| ascii_field(&f.value))
        .map(|s| s.trim().eq_ignore_ascii_case(&negative_ref.to_string()))
        .unwrap_or(false);

    Some(if negative { -degrees } else { degrees })
}

fn haversine_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    let (lat1, lon1) = (a.0.to_radians(), a.1.to_radians());
    let (lat2, lon2) = (b.0.to_radians(), b.1.to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * h.sqrt().asin()
}

fn ascii_field(value: &exif::Value) -> Option<String> {
    match value {
        exif::Value::Ascii(v) if !v.is_empty() => Some(String::from_utf8_lossy(&v[0]).to_string()),
        _ => None,
    }
}

fn exif_datetime_to_secs(dt: &exif::DateTime) -> i64 {
    days_from_civil(dt.year as i64, dt.month as u32, dt.day as u32) * 86400
        + dt.hour as i64 * 3600
        + dt.minute as i64 * 60
        + dt.second as i64
}

/// Days since 1970-01-01 for a civil (year, month, day), correct across the
/// whole proleptic Gregorian calendar including leap years. Howard Hinnant's
/// well-known `days_from_civil` algorithm - see
/// http://howardhinnant.github.io/date_algorithms.html
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Try to recover a real capture timestamp embedded in the filename itself -
/// written on-device, so it survives a transfer that clobbers mtime.
fn parse_filename_timestamp(path: &Path) -> Option<i64> {
    let stem = path.file_stem()?.to_str()?;
    parse_ios_screenshot_name(stem).or_else(|| parse_yyyymmdd_hhmmss(stem))
}

/// "Screenshot 2024-01-15 at 3.45.12 PM" (iOS's screenshot filename format).
fn parse_ios_screenshot_name(stem: &str) -> Option<i64> {
    let lower = stem.to_ascii_lowercase();
    let after_prefix = lower.strip_prefix("screenshot ")?;
    let (date_part, time_part) = after_prefix.split_once(" at ")?;

    let mut date_fields = date_part.split('-');
    let year: i64 = date_fields.next()?.parse().ok()?;
    let month: u32 = date_fields.next()?.parse().ok()?;
    let day: u32 = date_fields.next()?.parse().ok()?;

    let mut time_fields = time_part.split_whitespace();
    let hms = time_fields.next()?;
    let ampm = time_fields.next()?;

    let mut hms_fields = hms.split('.');
    let mut hour: u32 = hms_fields.next()?.parse().ok()?;
    let minute: u32 = hms_fields.next()?.parse().ok()?;
    let second: u32 = hms_fields.next()?.parse().ok()?;

    if ampm.starts_with('p') && hour != 12 {
        hour += 12;
    } else if ampm.starts_with('a') && hour == 12 {
        hour = 0;
    }

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}

/// A `YYYYMMDD` digit run, optionally immediately followed by `HHMMSS`
/// (contiguous, or separated by one non-digit character) - covers
/// "Screenshot_20240115-154512.png", "IMG_20240115_154512.jpg",
/// "PXL_20240115_154512332.jpg", and similar Android/messaging-app exports.
fn parse_yyyymmdd_hhmmss(stem: &str) -> Option<i64> {
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        let run_len = j - start;

        if run_len >= 8 {
            let date_str: String = chars[start..start + 8].iter().collect();
            if let Some((y, mo, d)) = parse_yyyymmdd(&date_str) {
                let time_digits: Option<String> = if run_len >= 14 {
                    Some(chars[start + 8..start + 14].iter().collect())
                } else {
                    let mut k = start + run_len;
                    if k < n && !chars[k].is_ascii_digit() {
                        k += 1;
                    }
                    (k + 6 <= n && chars[k..k + 6].iter().all(|c| c.is_ascii_digit()))
                        .then(|| chars[k..k + 6].iter().collect())
                };
                if let Some((h, mi, s)) = time_digits.and_then(|t| parse_hhmmss(&t)) {
                    return Some(days_from_civil(y, mo, d) * 86400 + h as i64 * 3600 + mi as i64 * 60 + s as i64);
                }
            }
        }
        i = j.max(i + 1);
    }
    None
}

fn parse_yyyymmdd(s: &str) -> Option<(i64, u32, u32)> {
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(4..6)?.parse().ok()?;
    let day: u32 = s.get(6..8)?.parse().ok()?;
    ((1990..=2035).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day))
        .then_some((year, month, day))
}

fn parse_hhmmss(s: &str) -> Option<(u32, u32, u32)> {
    let hour: u32 = s.get(0..2)?.parse().ok()?;
    let minute: u32 = s.get(2..4)?.parse().ok()?;
    let second: u32 = s.get(4..6)?.parse().ok()?;
    (hour <= 23 && minute <= 59 && second <= 59).then_some((hour, minute, second))
}

/// Trailing digit run in the filename stem plus whatever comes before it,
/// e.g. "IMG_1234.jpg" -> ("img|jpg", 1234). The strongest signal this module
/// has: consecutive numbers within one roll are shots from one sitting.
///
/// Two wrinkles, both of which cost real recall before they were handled:
///
/// * A `(n)` copy suffix is stripped first. "IMG_1711(1).JPG" would otherwise
///   parse its trailing run as the single digit `1`, fail the two-digit
///   minimum, and lose the sequence signal entirely - and a library that has
///   been merged a few times is full of these (28% of the files in the one
///   this was measured on).
/// * The family is keyed by *kind* as well as prefix. `IMG_####` numbers are
///   reused by every phone and every export, so a bare "img" prefix matches
///   across unrelated rolls spanning years. Splitting screenshots from camera
///   JPEGs took the out-of-order rate within one folder from 13% to 2.6%.
fn parse_filename_sequence(path: &Path) -> Option<(String, u64)> {
    let stem = path.file_stem()?.to_str()?;
    let stem = strip_copy_suffix(stem);
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();

    let mut end = n;
    while end > 0 && !chars[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 && chars[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if end - start < 2 {
        return None; // require >=2 digits to cut down on noise
    }

    let number: u64 = chars[start..end].iter().collect::<String>().parse().ok()?;
    let prefix: String = chars[..start]
        .iter()
        .collect::<String>()
        .trim_end_matches(['_', '-', ' '])
        .to_ascii_lowercase();
    Some((format!("{}|{}", prefix, sequence_kind(path)), number))
}

/// Drop a trailing "(1)", "(2)", ... that a copy/merge added to the stem.
fn strip_copy_suffix(stem: &str) -> &str {
    let trimmed = stem.trim_end();
    let Some(open) = trimmed.strip_suffix(')').and_then(|s| s.rfind('(')) else {
        return stem;
    };
    let inner = &trimmed[open + 1..trimmed.len() - 1];
    if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit()) {
        trimmed[..open].trim_end()
    } else {
        stem
    }
}

/// Coarse "which roll is this" bucket, used to keep separate sequences from
/// colliding on the same `IMG_####` numbers. Screenshots (PNG) and camera or
/// shared photos (JPEG) interleave in the same folder with independently
/// numbered sequences.
fn sequence_kind(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("png") => "png",
        Some("jpg" | "jpeg") => "jpg",
        Some("mp4" | "mov" | "m4v" | "avi" | "mkv" | "webm") => "vid",
        _ => "other",
    }
}

/// Read the `creation_time` field out of an MP4/MOV `moov/mvhd` box. Returns
/// `None` (never panics) on any malformed/unexpected structure.
fn read_mp4_creation_time(path: &Path) -> Option<i64> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let moov = find_box(&mut file, 0, file_len, b"moov")?;
    let mvhd = find_box(&mut file, moov.0, moov.0 + moov.1, b"mvhd")?;

    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(mvhd.0)).ok()?;
    let mut version = [0u8; 1];
    file.read_exact(&mut version).ok()?;

    let mac_secs = if version[0] == 1 {
        let mut buf = [0u8; 3 + 8];
        file.read_exact(&mut buf).ok()?;
        i64::from_be_bytes(buf[3..11].try_into().ok()?)
    } else {
        let mut buf = [0u8; 3 + 4];
        file.read_exact(&mut buf).ok()?;
        u32::from_be_bytes(buf[3..7].try_into().ok()?) as i64
    };
    if mac_secs == 0 {
        return None; // unset, common in some encoders
    }
    Some(mac_secs - MAC_EPOCH_OFFSET)
}

/// Search sibling ISO-BMFF boxes in byte range `[start, end)` for `fourcc`,
/// returning that box's (content_start, content_len). Bounds the number of
/// boxes walked so a malformed file can't spin forever.
fn find_box(file: &mut std::fs::File, start: u64, end: u64, fourcc: &[u8]) -> Option<(u64, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut pos = start;
    let mut guard = 0u32;
    while pos + 8 <= end {
        guard += 1;
        if guard > 10_000 {
            return None;
        }
        file.seek(SeekFrom::Start(pos)).ok()?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header).ok()?;
        let mut size = u32::from_be_bytes(header[0..4].try_into().ok()?) as u64;
        let box_type = &header[4..8];
        let mut header_len = 8u64;

        if size == 1 {
            let mut ext = [0u8; 8];
            file.read_exact(&mut ext).ok()?;
            size = u64::from_be_bytes(ext);
            header_len = 16;
        } else if size == 0 {
            return None; // "extends to end of file" - not worth chasing here
        }
        if size < header_len {
            return None;
        }

        if box_type == fourcc {
            return Some((pos + header_len, size - header_len));
        }
        pos += size;
    }
    None
}

pub fn humanize_delta(secs: i64) -> String {
    let secs = secs.max(0);
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_from_civil_matches_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 2, 29), 11016); // leap day
        assert_eq!(days_from_civil(2024, 3, 1), 19783);
    }

    #[test]
    fn humanize_delta_formats_ranges() {
        assert_eq!(humanize_delta(45), "45s");
        assert_eq!(humanize_delta(134), "2m14s");
        assert_eq!(humanize_delta(3900), "1h05m");
    }

    #[test]
    fn ios_screenshot_name_parses_to_the_right_moment() {
        let secs = parse_ios_screenshot_name("screenshot 2024-01-15 at 3.45.12 pm").unwrap();
        assert_eq!(secs, days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12);
    }

    #[test]
    fn ios_screenshot_name_handles_midnight_and_noon() {
        let midnight = parse_ios_screenshot_name("screenshot 2024-01-15 at 12.00.00 am").unwrap();
        assert_eq!(midnight, days_from_civil(2024, 1, 15) * 86400);
        let noon = parse_ios_screenshot_name("screenshot 2024-01-15 at 12.00.00 pm").unwrap();
        assert_eq!(noon, days_from_civil(2024, 1, 15) * 86400 + 12 * 3600);
    }

    #[test]
    fn yyyymmdd_hhmmss_parses_common_android_and_messaging_names() {
        let expected = days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12;
        assert_eq!(parse_yyyymmdd_hhmmss("IMG_20240115_154512"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("Screenshot_20240115-154512"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("PXL_20240115_154512332"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("20240115154512"), Some(expected));
    }

    #[test]
    fn yyyymmdd_hhmmss_rejects_implausible_dates() {
        assert_eq!(parse_yyyymmdd_hhmmss("IMG_99999999_999999"), None);
    }

    #[test]
    fn filename_sequence_extracts_trailing_number_and_prefix() {
        assert_eq!(
            parse_filename_sequence(Path::new("IMG_1234.jpg")),
            Some(("img|jpg".to_string(), 1234))
        );
        assert_eq!(
            parse_filename_sequence(Path::new("DSC05678.NEF")),
            Some(("dsc|other".to_string(), 5678))
        );
        assert_eq!(parse_filename_sequence(Path::new("photo.jpg")), None);
    }

    #[test]
    fn filename_sequence_sees_through_a_copy_suffix() {
        // Without the suffix strip this parses the "1" of "(1)" as the
        // sequence, fails the two-digit minimum, and returns None.
        assert_eq!(
            parse_filename_sequence(Path::new("IMG_1711(1).JPG")),
            Some(("img|jpg".to_string(), 1711))
        );
        assert_eq!(
            parse_filename_sequence(Path::new("IMG_1711(12).JPG")),
            Some(("img|jpg".to_string(), 1711))
        );
    }

    #[test]
    fn copy_suffix_strip_only_touches_numeric_parentheticals() {
        assert_eq!(strip_copy_suffix("IMG_1711(1)"), "IMG_1711");
        assert_eq!(strip_copy_suffix("IMG_1711(12)"), "IMG_1711");
        assert_eq!(strip_copy_suffix("IMG_1711 (3)"), "IMG_1711");
        assert_eq!(strip_copy_suffix("IMG_1711(final)"), "IMG_1711(final)");
        assert_eq!(strip_copy_suffix("IMG_1711()"), "IMG_1711()");
        assert_eq!(strip_copy_suffix("IMG_1711"), "IMG_1711");
        assert_eq!(strip_copy_suffix("(2)"), "");
    }

    #[test]
    fn filename_sequence_separates_interleaved_rolls_by_kind() {
        let png = parse_filename_sequence(Path::new("IMG_1711.PNG")).unwrap();
        let jpg = parse_filename_sequence(Path::new("IMG_1711.JPG")).unwrap();
        assert_eq!(png.1, jpg.1);
        assert_ne!(png.0, jpg.0, "a screenshot roll must not match a camera roll");
    }

    #[test]
    fn dimension_match_grades_exact_aspect_and_unknown() {
        assert_eq!(
            DimensionMatch::compare(Some((1179, 2556)), Some((1179, 2556))),
            DimensionMatch::Exact
        );
        // Same 3:2 shape at two sizes - a resized export of one source.
        assert_eq!(
            DimensionMatch::compare(Some((3000, 2000)), Some((1500, 1000))),
            DimensionMatch::AspectOnly
        );
        assert_eq!(
            DimensionMatch::compare(Some((1179, 2556)), Some((1024, 1024))),
            DimensionMatch::Different
        );
        assert_eq!(DimensionMatch::compare(Some((100, 100)), None), DimensionMatch::Unknown);
        assert_eq!(DimensionMatch::compare(None, None), DimensionMatch::Unknown);
    }

    #[test]
    fn dimension_match_does_not_divide_by_a_zero_height() {
        assert_eq!(
            DimensionMatch::compare(Some((100, 0)), Some((50, 25))),
            DimensionMatch::Different
        );
    }

    #[test]
    fn haversine_of_same_point_is_zero_and_scales_with_distance() {
        let sf = (37.7749, -122.4194);
        assert!(haversine_km(sf, sf) < 1e-9);
        let nyc = (40.7128, -74.0060);
        let km = haversine_km(sf, nyc);
        assert!(km > 4000.0 && km < 4200.0); // SF-NYC is ~4130 km
    }

    #[test]
    fn mtime_confidence_drops_when_many_files_share_a_bucket() {
        let mut buckets: HashMap<i64, u32> = HashMap::new();
        buckets.insert(0, 25);
        buckets.insert(1, 1500);

        let sparse = PhotoMeta { secs: 100, time_source: TimeSource::Mtime, camera: None, gps: None, sequence: None, dimensions: None };
        let bulk = PhotoMeta { secs: MTIME_COLLISION_BUCKET_SECS + 5, time_source: TimeSource::Mtime, camera: None, gps: None, sequence: None, dimensions: None };
        let confidence_of = |meta: &PhotoMeta| -> f32 {
            if meta.time_source != TimeSource::Mtime {
                return 1.0;
            }
            let bucket = meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS);
            if buckets.get(&bucket).copied().unwrap_or(0) > MTIME_COLLISION_THRESHOLD {
                MTIME_COLLISION_CONFIDENCE
            } else {
                meta.time_source.base_confidence()
            }
        };

        assert_eq!(confidence_of(&sparse), MTIME_BASE_CONFIDENCE);
        assert_eq!(confidence_of(&bulk), MTIME_COLLISION_CONFIDENCE);
    }

    /// Write a real PNG (so `read_dimensions` has something to parse) and
    /// stamp its mtime, which is what `read_photo_meta` falls back to for a
    /// PNG carrying no EXIF.
    fn write_png_at(dir: &Path, name: &str, w: u32, h: u32, secs: i64) -> PathBuf {
        let path = dir.join(name);
        image::RgbImage::new(w, h).save(&path).unwrap();
        let when = UNIX_EPOCH + std::time::Duration::from_secs(secs as u64);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(when).unwrap();
        path
    }

    /// The regression this module was reworked for: a screenshot ten hours
    /// away from the nearest confirmed photo, but three apart from it in the
    /// filename sequence, used to be discarded by the time window before it
    /// could be scored at all.
    #[test]
    fn a_tight_sequence_tie_survives_a_window_that_would_drop_it_on_time() {
        let dir = std::env::temp_dir().join(format!("metasim_rescue_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let noon = 1_735_200_000_i64;
        let ten_hours = 10 * 3600;
        let anchor = write_png_at(&dir, "IMG_1714.png", 40, 80, noon);
        // Three apart in sequence, same shape, but far outside a 1h window.
        let near_in_sequence = write_png_at(&dir, "IMG_1711.png", 40, 80, noon - ten_hours);
        // Equally far in time with no sequence tie - should stay dropped.
        let unrelated = write_png_at(&dir, "IMG_9999.png", 40, 80, noon - ten_hours);
        // Two apart in sequence but 100 days off: the number is a coincidence
        // from another roll, and the consistency check must reject it.
        let stale = write_png_at(&dir, "IMG_1712.png", 40, 80, noon - 100 * 86400);

        let ranked = rank_by_metadata_cached(
            &[anchor.clone()],
            vec![near_in_sequence.clone(), unrelated.clone(), stale.clone()],
            3600,
            &dir.join("cache.bin"),
        );
        let found: Vec<&PathBuf> = ranked.iter().map(|c| &c.path).collect();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(found, vec![&near_in_sequence], "only the sequence neighbour should survive");
        let hit = &ranked[0];
        assert_eq!(hit.sequence_gap, Some(3));
        assert!(hit.rescued_by_sequence, "it is outside the window, so it was rescued");
        assert_eq!(hit.dimension_match, DimensionMatch::Exact);
        assert_eq!(hit.delta_secs, ten_hours);
    }

    #[test]
    fn a_candidate_inside_the_window_is_not_reported_as_rescued() {
        let dir = std::env::temp_dir().join(format!("metasim_window_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let noon = 1_735_200_000_i64;
        let anchor = write_png_at(&dir, "IMG_1714.png", 40, 80, noon);
        let close = write_png_at(&dir, "IMG_1713.png", 40, 80, noon - 120);

        let ranked = rank_by_metadata_cached(&[anchor], vec![close.clone()], 3600, &dir.join("cache.bin"));
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(ranked.len(), 1);
        assert!(!ranked[0].rescued_by_sequence);
        assert_eq!(ranked[0].sequence_gap, Some(1));
    }

    #[test]
    fn the_metadata_cache_round_trips_and_reruns_identically() {
        let dir = std::env::temp_dir().join(format!("metasim_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.join("cache.bin");

        let noon = 1_735_200_000_i64;
        let anchor = write_png_at(&dir, "IMG_1714.png", 40, 80, noon);
        let close = write_png_at(&dir, "IMG_1713.png", 40, 80, noon - 120);

        let cold = rank_by_metadata_cached(&[anchor.clone()], vec![close.clone()], 3600, &cache);
        assert!(cache.exists(), "the scan should have written a cache");
        // Second run reads every value back out of the cache instead of the files.
        let warm = rank_by_metadata_cached(&[anchor.clone()], vec![close.clone()], 3600, &cache);

        assert_eq!(cold.len(), warm.len());
        assert_eq!(cold[0].score, warm[0].score);
        assert_eq!(cold[0].delta_secs, warm[0].delta_secs);
        assert_eq!(cold[0].sequence_gap, warm[0].sequence_gap);
        assert_eq!(cold[0].dimension_match, warm[0].dimension_match);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_metadata_cache_is_invalidated_when_a_file_changes() {
        let dir = std::env::temp_dir().join(format!("metasim_stale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.join("cache.bin");

        let noon = 1_735_200_000_i64;
        let anchor = write_png_at(&dir, "IMG_1714.png", 40, 80, noon);
        let candidate = write_png_at(&dir, "IMG_1713.png", 40, 80, noon - 120);

        let before = rank_by_metadata_cached(&[anchor.clone()], vec![candidate.clone()], 3600, &cache);
        assert_eq!(before[0].dimension_match, DimensionMatch::Exact);

        // Rewrite the candidate at a different size and a different mtime. If
        // the cache ignored either, the stale dimensions would still say Exact.
        let candidate = write_png_at(&dir, "IMG_1713.png", 64, 64, noon - 60);
        let after = rank_by_metadata_cached(&[anchor], vec![candidate], 3600, &cache);

        assert_eq!(after[0].dimension_match, DimensionMatch::Different);
        assert_eq!(after[0].delta_secs, 60);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cache_from_an_older_version_is_discarded_rather_than_trusted() {
        let dir = std::env::temp_dir().join(format!("metasim_ver_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.join("cache.bin");

        let stale = MetaCache { version: META_CACHE_VERSION + 1, entries: HashMap::new() };
        std::fs::write(&cache, bincode::serialize(&stale).unwrap()).unwrap();
        assert!(load_meta_cache(&cache).is_empty());

        std::fs::write(&cache, b"not a cache at all").unwrap();
        assert!(load_meta_cache(&cache).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mp4_creation_time_reads_a_minimal_synthetic_moov_mvhd_box() {
        // ftyp (8 bytes header + 4 bytes "isom") + moov > mvhd (version 0).
        let mut data = Vec::new();
        data.extend_from_slice(&12u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"isom");

        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // size placeholder
        mvhd.extend_from_slice(b"mvhd");
        mvhd.push(0); // version 0
        mvhd.extend_from_slice(&[0, 0, 0]); // flags
        // creation_time: 2024-01-15 15:45:12 UTC in Unix time, converted to Mac epoch.
        let unix_secs = days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12;
        let mac_secs = (unix_secs + MAC_EPOCH_OFFSET) as u32;
        mvhd.extend_from_slice(&mac_secs.to_be_bytes());
        mvhd.extend_from_slice(&[0u8; 4]); // modification_time (unused)
        let mvhd_len = mvhd.len() as u32;
        mvhd[0..4].copy_from_slice(&mvhd_len.to_be_bytes());

        let mut moov = Vec::new();
        moov.extend_from_slice(&0u32.to_be_bytes());
        moov.extend_from_slice(b"moov");
        moov.extend_from_slice(&mvhd);
        let moov_len = moov.len() as u32;
        moov[0..4].copy_from_slice(&moov_len.to_be_bytes());

        data.extend_from_slice(&moov);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("metasim_test_{}.mp4", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let result = read_mp4_creation_time(&path);
        let _ = std::fs::remove_file(&path);

        assert_eq!(result, Some(unix_secs));
    }
}

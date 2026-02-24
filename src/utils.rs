use std::path::{Path, PathBuf};
use std::process::Command;

pub fn find_ffmpeg_path() -> Option<PathBuf> {
    // 1. Check PATH
    if Command::new("ffmpeg").arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from("ffmpeg"));
    }

    // 2. Check beside executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let ffmpeg_path = dir.join("ffmpeg.exe");
            if ffmpeg_path.exists() {
                return Some(ffmpeg_path);
            }
            let ffmpeg_path_linux = dir.join("ffmpeg");
            if ffmpeg_path_linux.exists() {
                return Some(ffmpeg_path_linux);
            }
        }
    }

    // 3. Check current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let ffmpeg_path = cwd.join("ffmpeg.exe");
        if ffmpeg_path.exists() {
            return Some(ffmpeg_path);
        }
        let ffmpeg_path_linux = cwd.join("ffmpeg");
        if ffmpeg_path_linux.exists() {
            return Some(ffmpeg_path_linux);
        }
    }

    None
}

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .map(|ext| {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), "mp4" | "mkv" | "mov" | "webm" | "avi" | "m4v")
        })
        .unwrap_or(false)
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .map(|ext| {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp" | "avif" | "gif" | "heic" | "ithmb")
        })
        .unwrap_or(false)
}

pub fn load_image_robustly(path: &Path) -> anyhow::Result<image::DynamicImage> {
    // Try standard image crate first
    if let Ok(img) = image::open(path) {
        return Ok(img);
    }

    // Fallback to ffmpeg for HEIC or if standard open fails
    let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
    if ext == "heic" || ext == "avif" {
        if let Some(ffmpeg_cmd) = find_ffmpeg_path() {
            let output = Command::new(&ffmpeg_cmd)
                .arg("-i").arg(path)
                .arg("-vframes").arg("1")
                .arg("-f").arg("image2pipe")
                .arg("-vcodec").arg("png")
                .arg("-")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()?;

            if output.status.success() {
                return Ok(image::load_from_memory(&output.stdout)?);
            }
        }
    }

    anyhow::bail!("Could not decode image: {}", path.display())
}

pub fn get_video_thumbnail_path(video_path: &Path) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Use absolute path for more stable hashing if possible, 
    // but relative is fine if the workspace is moved.
    // For now, let's use the string representation as it is.
    video_path.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();
    
    crate::get_app_data_dir().join("thumbnails").join(format!("{:x}.jpg", hash))
}

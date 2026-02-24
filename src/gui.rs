use std::path::PathBuf;
use eframe::egui;
use rfd::FileDialog;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashSet;

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize, Deserialize, Default)]
struct AppSettings {
    input_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
}

impl AppSettings {
    fn load() -> Self {
        let config_dir = crate::get_app_data_dir();
        let config_path = config_dir.join("settings.json");
        if let Ok(data) = fs::read_to_string(config_path) {
            if let Ok(settings) = serde_json::from_str(&data) {
                return settings;
            }
        }
        Self::default()
    }

    fn save(&self) {
        let config_dir = crate::get_app_data_dir();
        if fs::create_dir_all(&config_dir).is_ok() {
            let config_path = config_dir.join("settings.json");
            if let Ok(data) = serde_json::to_string_pretty(self) {
                let _ = fs::write(config_path, data);
            }
        }
    }
}

pub struct FaceSearchApp {
    input_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    match_threshold_min: f32,
    match_threshold: f32,
    filter_threshold: f32,
    page_size: usize,
    thumbnail_size: f32,
    current_page: usize,
    is_processing: bool,
    status_msg: String,

    // Stats and Matches state
    processed_count: usize,
    target_image_count: usize,
    target_video_count: usize,
    all_ranked_matches: Vec<(PathBuf, f32)>,
    matched_images_cache: Vec<(PathBuf, f32, bool, Option<Result<egui::TextureHandle, String>>)>,
    last_selected_index: Option<usize>,
    show_copy_confirm: bool,
    show_rebuild_confirm: bool,
    show_new_person_modal: bool,
    new_person_name: String,
    new_person_image_path: Option<PathBuf>,

    // Log history for display during processing
    log_messages: Vec<String>,

    // Scroll control
    scroll_to_top: bool,

    // Communication with background thread
    tx: Sender<UiMessage>,
    rx: Receiver<UiMessage>,
}

pub enum UiMessage {
    Log(String),
    Done(usize, Vec<(PathBuf, f32)>), // (processed_count, sorted matches with euclidean distance)
    Error(String),
}

impl Default for FaceSearchApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let settings = AppSettings::load();
        Self {
            input_dir: settings.input_dir,
            target_dir: settings.target_dir,
            match_threshold_min: 0.0,
            match_threshold: 0.65,
            filter_threshold: 0.2,
            page_size: 100,
            thumbnail_size: 300.0,
            current_page: 0,
            is_processing: false,
            status_msg: "Ready".to_string(),
            processed_count: 0,
            target_image_count: 0,
            target_video_count: 0,
            all_ranked_matches: Vec::new(),
            matched_images_cache: Vec::new(),
            last_selected_index: None,
            show_copy_confirm: false,
            show_rebuild_confirm: false,
            show_new_person_modal: false,
            new_person_name: String::new(),
            new_person_image_path: None,
            log_messages: Vec::new(),
            scroll_to_top: false,
            tx,
            rx,
        }
    }
}

fn get_unique_path(dir: &std::path::Path, file_name: &std::ffi::OsStr) -> PathBuf {
    let mut path = dir.join(file_name);
    let mut counter = 1;
    let stem = std::path::Path::new(file_name).file_stem().unwrap_or_default().to_string_lossy();
    let ext = std::path::Path::new(file_name).extension().unwrap_or_default().to_string_lossy();

    while path.exists() {
        let new_name = if ext.is_empty() {
            format!("{}_{}", stem, counter)
        } else {
            format!("{}_{}.{}", stem, counter, ext)
        };
        path = dir.join(new_name);
        counter += 1;
    }
    path
}

impl FaceSearchApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut app = Self::default();
        app.update_target_count();
        app
    }

    fn save_settings(&self) {
        let settings = AppSettings {
            input_dir: self.input_dir.clone(),
            target_dir: self.target_dir.clone(),
        };
        settings.save();
    }

    fn update_target_count(&mut self) {
        self.target_image_count = 0;
        self.target_video_count = 0;
        if let Some(dir) = &self.target_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_file() {
                        if crate::utils::is_image(&path) {
                            self.target_image_count += 1;
                        } else if crate::utils::is_video(&path) {
                            self.target_video_count += 1;
                        }
                    }
                }
            }
        }
    }

    fn total_pages(&self) -> usize {
        if self.page_size == 0 { return 1; }
        let total = self.all_ranked_matches.len();
        if total == 0 { return 1; }
        (total + self.page_size - 1) / self.page_size
    }

    fn load_page(&mut self, page: usize) {
        self.matched_images_cache.clear();
        let start = page * self.page_size;
        let end = (start + self.page_size).min(self.all_ranked_matches.len());
        for (path, dist) in &self.all_ranked_matches[start..end] {
            self.matched_images_cache.push((path.clone(), *dist, false, None));
        }
        self.current_page = page;
        self.last_selected_index = None;
        self.scroll_to_top = true;
    }
}

impl eframe::App for FaceSearchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle incoming messages from background thread
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                UiMessage::Log(txt) => {
                    self.status_msg = txt.clone();
                    self.log_messages.push(txt);
                    if self.log_messages.len() > 200 {
                        self.log_messages.remove(0);
                    }
                }
                UiMessage::Done(processed, ranked) => {
                    self.is_processing = false;
                    self.processed_count = processed;
                    let total = ranked.len();
                    self.all_ranked_matches = ranked;
                    self.load_page(0);
                    let pages = self.total_pages();
                    self.log_messages.clear();
                    if total > 0 {
                        self.status_msg = format!(
                            "Found {} candidates from {} images scanned. Showing page 1 of {}.",
                            total, processed, pages
                        );
                    } else {
                        self.status_msg = format!(
                            "Finished processing {} images. No candidates found within threshold.",
                            processed
                        );
                    }
                }
                UiMessage::Error(err) => {
                    self.is_processing = false;
                    self.log_messages.clear();
                    self.status_msg = format!("Error: {}", err);
                }
            }
        }

        // Handle drag and drop for target directory (we'll just take the first folder or parent of first file)
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    if path.is_dir() {
                        self.target_dir = Some(path.clone());
                        self.update_target_count();
                        break;
                    } else if let Some(parent) = path.parent() {
                        self.target_dir = Some(parent.to_path_buf());
                        self.update_target_count();
                        break;
                    }
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Facial Recognition Sorter v{}", APP_VERSION));
            ui.separator();

            // --- Directories Input ---
            ui.horizontal(|ui| {
                if ui.button("Select Input Directory").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.input_dir = Some(path);
                        self.save_settings();
                    }
                }
                if let Some(p) = &self.input_dir {
                    if ui.button("📂 Open").clicked() {
                        let _ = std::process::Command::new("explorer").arg(p).spawn();
                    }
                }
                ui.label(match &self.input_dir {
                    Some(p) => p.display().to_string(),
                    None => "No directory selected".to_string(),
                });
            });

            ui.separator();

            // --- Target Directory Input ---
            ui.horizontal(|ui| {
                if ui.button("Select Target Person Directory").clicked() {
                    let start_dir = self.target_dir.as_ref()
                        .and_then(|td| td.parent().map(|p| p.to_path_buf()));
                    let mut dialog = FileDialog::new();
                    if let Some(sd) = &start_dir {
                        dialog = dialog.set_directory(sd);
                    }
                    if let Some(path) = dialog.pick_folder() {
                        self.target_dir = Some(path);
                        self.update_target_count();
                        self.save_settings();
                    }
                }
                if let Some(p) = &self.target_dir {
                    if ui.button("📂 Open").clicked() {
                        let _ = std::process::Command::new("explorer").arg(p).spawn();
                    }
                    if ui.button("⟳ Refresh").clicked() {
                        self.update_target_count();
                    }
                }
                ui.label(match &self.target_dir {
                    Some(p) => format!("{} ({} images, {} videos)", p.display(), self.target_image_count, self.target_video_count),
                    None => "No specific person directory selected (will cluster everyone)".to_string(),
                });
            });

            if self.target_dir.is_some() {
                ui.horizontal(|ui| {
                    ui.label("Distance range:");
                    ui.add(
                        egui::DragValue::new(&mut self.match_threshold_min)
                            .range(0.0..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                            .prefix("Min: "),
                    );
                    ui.label("–");
                    ui.add(
                        egui::DragValue::new(&mut self.match_threshold)
                            .range(0.0..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                            .prefix("Max: "),
                    );
                    ui.add_space(16.0);
                    ui.label("Per page:");
                    ui.add(
                        egui::DragValue::new(&mut self.page_size)
                            .range(10..=200)
                            .speed(1.0),
                    );
                    ui.add_space(16.0);
                    ui.label("Thumbnail:");
                    ui.add(
                        egui::DragValue::new(&mut self.thumbnail_size)
                            .range(60.0..=300.0)
                            .speed(1.0)
                            .suffix(" px"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Target rejection:");
                    ui.add(
                        egui::DragValue::new(&mut self.filter_threshold)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                    );
                    ui.label("(Higher = stricter. Faces below this similarity to the dominant identity are rejected as bystanders.)");
                });
                ui.label("(Lower distance = stricter. Results within the range are shown, ranked best-match-first.)");
            }

            ui.separator();

            // --- Start Button ---
            ui.add_enabled_ui(!self.is_processing && self.input_dir.is_some(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Start Processing").clicked() {
                        self.update_target_count();
                        self.is_processing = true;
                        self.status_msg = "Starting...".to_string();
                        self.all_ranked_matches.clear();
                        self.matched_images_cache.clear();
                        self.current_page = 0;

                        let tx_clone = self.tx.clone();
                        let input = self.input_dir.clone().unwrap();
                        let target_dir = self.target_dir.clone();
                        let people_dir = target_dir.as_ref().and_then(|td| td.parent().map(|p| p.to_path_buf()));
                        let threshold_min = self.match_threshold_min.min(self.match_threshold);
                        let threshold_max = self.match_threshold;
                        let filter_threshold = self.filter_threshold;

                        thread::spawn(move || {
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                crate::process_directory(input, target_dir, people_dir, threshold_min, threshold_max, filter_threshold, tx_clone.clone())
                            }));
                            match result {
                                Ok(Err(e)) => {
                                    let _ = tx_clone.send(UiMessage::Error(e.to_string()));
                                }
                                Err(panic_info) => {
                                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                                        format!("Internal error (panic): {}", s)
                                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                                        format!("Internal error (panic): {}", s)
                                    } else {
                                        "Internal error (panic): unknown".to_string()
                                    };
                                    let _ = tx_clone.send(UiMessage::Error(msg));
                                }
                                Ok(Ok(())) => {}
                            }
                        });
                    }
                    if ui.button("Rebuild Database").clicked() {
                        self.show_rebuild_confirm = true;
                    }
                });
            });

            // --- Status ---
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.status_msg).strong());

                if !self.is_processing && !self.all_ranked_matches.is_empty() && self.target_dir.is_some() {
                    let selected_count = self.matched_images_cache.iter().filter(|(_, _, s, _)| *s).count();
                    if selected_count > 0 {
                        if ui.button(format!("Copy {} Selected to Target Directory", selected_count)).clicked() {
                            self.show_copy_confirm = true;
                        }
                    }
                    if ui.button("Export All to Output Folder").clicked() {
                        let out_dir = crate::get_app_data_dir().join("output").join("target_matches");
                        let _ = std::fs::create_dir_all(&out_dir);
                        for (src_path, _) in &self.all_ranked_matches {
                            if let Some(name) = src_path.file_name() {
                                let dest = out_dir.join(name);
                                if !dest.exists() {
                                    let _ = std::fs::copy(src_path, &dest);
                                }
                            }
                        }
                        self.status_msg = format!("Exported {} candidates to output folder.", self.all_ranked_matches.len());
                    }
                    if ui.button("Open Output Folder").clicked() {
                        let output = crate::get_app_data_dir().join("output").join("target_matches");
                        if output.exists() {
                            let _ = std::process::Command::new("explorer").arg(&output).spawn();
                        }
                    }
                    if ui.button("Open Debug Targets").clicked() {
                        let debug_dir = crate::get_app_data_dir().join("output").join("debug_targets");
                        if debug_dir.exists() {
                            let _ = std::process::Command::new("explorer").arg(&debug_dir).spawn();
                        }
                    }
                    if ui.button("Open Debug Rejected").clicked() {
                        let rejected_dir = crate::get_app_data_dir().join("output").join("debug_rejected");
                        if rejected_dir.exists() {
                            let _ = std::process::Command::new("explorer").arg(&rejected_dir).spawn();
                        }
                    }
                }
            });

            // --- Copy Confirmation Modal ---
            if self.show_copy_confirm {
                let selected_indices: Vec<usize> = self.matched_images_cache.iter()
                    .enumerate()
                    .filter(|(_, (_, _, s, _))| *s)
                    .map(|(i, _)| i)
                    .collect();

                let mut do_copy = false;
                let mut do_cancel = false;

                egui::Window::new("Confirm Copy")
                    .collapsible(false)
                    .resizable(true)
                    .default_size([600.0, 450.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading(format!("Copy {} selected photo(s) to target directory?", selected_indices.len()));
                        ui.separator();

                        let cell_size = 100.0_f32;
                        let available_w = ui.available_width();
                        let cols = ((available_w / cell_size).floor() as usize).max(1);

                        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                            egui::Grid::new("confirm_preview_grid")
                                .num_columns(cols)
                                .spacing([4.0, 4.0])
                                .show(ui, |ui| {
                                    for (col_i, &idx) in selected_indices.iter().enumerate() {
                                        let (img_path, _, _, texture_res_opt) = &self.matched_images_cache[idx];
                                        match texture_res_opt {
                                            Some(Ok(texture)) => {
                                                let image = egui::Image::new(&*texture)
                                                    .fit_to_exact_size(egui::vec2(88.0, 88.0))
                                                    .maintain_aspect_ratio(true);
                                                ui.add(image);
                                            }
                                            Some(Err(_)) => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new("⚠ Error"));
                                            }
                                            None => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new(img_path.file_name()
                                                    .map(|n| n.to_string_lossy().to_string())
                                                    .unwrap_or_default()));
                                            }
                                        }
                                        if (col_i + 1) % cols == 0 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("✔ Confirm Copy").color(egui::Color32::from_rgb(80, 200, 100))).clicked() {
                                do_copy = true;
                            }
                            ui.add_space(12.0);
                            if ui.button(egui::RichText::new("✖ Cancel").color(egui::Color32::from_rgb(220, 80, 80))).clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_copy {
                    let target_dest = self.target_dir.clone().unwrap();
                    let mut copy_count = 0;
                    for (path, _, selected, _) in &self.matched_images_cache {
                        if *selected {
                            if let Some(file_name) = path.file_name() {
                                let destination = get_unique_path(&target_dest, file_name);
                                if std::fs::copy(path, &destination).is_ok() {
                                    copy_count += 1;
                                }
                            }
                        }
                    }

                    // Remove copied items from master ranked list
                    let removed: HashSet<PathBuf> = self.matched_images_cache.iter()
                        .filter(|(_, _, s, _)| *s)
                        .map(|(p, _, _, _)| p.clone())
                        .collect();
                    self.all_ranked_matches.retain(|(p, _)| !removed.contains(p));
                    self.matched_images_cache.retain(|(_, _, s, _)| !*s);

                    // Reload: if current page is now past the end, go back one page
                    let max_page = if self.all_ranked_matches.is_empty() { 0 } else { self.total_pages() - 1 };
                    let reload_page = self.current_page.min(max_page);
                    self.load_page(reload_page);

                    self.status_msg = format!("Successfully copied {} images to Target Directory!", copy_count);
                    self.update_target_count();
                    self.show_copy_confirm = false;
                }
                if do_cancel {
                    self.show_copy_confirm = false;
                }
            }

            // --- Rebuild Database Confirmation Modal ---
            if self.show_rebuild_confirm {
                let mut do_rebuild = false;
                let mut do_cancel = false;

                egui::Window::new("Confirm Rebuild Database")
                    .collapsible(false)
                    .resizable(false)
                    .default_size([450.0, 150.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading("Are you sure?");
                        ui.separator();
                        ui.label("This will delete the face database (faces_db.bin and faces_db.json).");
                        ui.label("All images will need to be re-processed from scratch on the next run.");
                        ui.label(
                            egui::RichText::new("This cannot be undone.")
                                .color(egui::Color32::from_rgb(220, 80, 80))
                                .strong()
                        );
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(
                                egui::RichText::new("Confirm Rebuild")
                                    .color(egui::Color32::from_rgb(220, 80, 80))
                            ).clicked() {
                                do_rebuild = true;
                            }
                            ui.add_space(12.0);
                            if ui.button("Cancel").clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_rebuild {
                    let db_path = crate::get_db_file();
                    if db_path.exists() {
                        let _ = std::fs::remove_file(db_path);
                    }
                    let db_json_path = crate::get_db_file_json();
                    if db_json_path.exists() {
                        let _ = std::fs::remove_file(db_json_path);
                    }
                    self.status_msg = "Database deleted. Click 'Start Processing' to rebuild from scratch.".to_string();
                    self.show_rebuild_confirm = false;
                }
                if do_cancel {
                    self.show_rebuild_confirm = false;
                }
            }

            // --- New Person Modal ---
            if self.show_new_person_modal {
                let mut do_create = false;
                let mut do_cancel = false;

                let people_dir = self.target_dir.as_ref()
                    .and_then(|td| td.parent().map(|p| p.to_path_buf()));

                egui::Window::new("Create New Person")
                    .collapsible(false)
                    .resizable(false)
                    .default_size([400.0, 200.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading("Create a new person directory");
                        ui.separator();

                        if let Some(pd) = &people_dir {
                            ui.label(format!("Parent: {}", pd.display()));
                        }

                        ui.horizontal(|ui| {
                            ui.label("Person name:");
                            ui.text_edit_singleline(&mut self.new_person_name);
                        });

                        let name_trimmed = self.new_person_name.trim();
                        let is_valid = !name_trimmed.is_empty()
                            && !name_trimmed.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']);
                        let already_exists = people_dir.as_ref()
                            .map(|pd| pd.join(name_trimmed).exists())
                            .unwrap_or(false);

                        if !name_trimmed.is_empty() && !is_valid {
                            ui.colored_label(egui::Color32::from_rgb(220, 80, 80),
                                "Name contains invalid characters.");
                        }
                        if already_exists {
                            ui.colored_label(egui::Color32::from_rgb(220, 180, 0),
                                "Directory already exists. Image will be added to it.");
                        }

                        if let Some(img_path) = &self.new_person_image_path {
                            ui.label(format!("Image: {}", img_path.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default()));
                        }

                        ui.separator();
                        ui.horizontal(|ui| {
                            let can_create = is_valid && people_dir.is_some();
                            if ui.add_enabled(can_create,
                                egui::Button::new(
                                    egui::RichText::new("Create + Copy")
                                        .color(egui::Color32::from_rgb(80, 200, 100))
                                )).clicked() {
                                do_create = true;
                            }
                            ui.add_space(12.0);
                            if ui.button(
                                egui::RichText::new("Cancel")
                                    .color(egui::Color32::from_rgb(220, 80, 80))
                            ).clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_create {
                    if let (Some(people_dir), Some(img_path)) = (people_dir, &self.new_person_image_path.clone()) {
                        let person_dir = people_dir.join(self.new_person_name.trim());
                        let _ = std::fs::create_dir_all(&person_dir);
                        if let Some(file_name) = img_path.file_name() {
                            let dest = get_unique_path(&person_dir, file_name);
                            if std::fs::copy(img_path, &dest).is_ok() {
                                self.status_msg = format!(
                                    "Created '{}' and copied image.",
                                    self.new_person_name.trim()
                                );
                            } else {
                                self.status_msg = "Failed to copy image.".to_string();
                            }
                        }
                    }
                    self.show_new_person_modal = false;
                    self.new_person_image_path = None;
                }
                if do_cancel {
                    self.show_new_person_modal = false;
                    self.new_person_image_path = None;
                }
            }

            if self.is_processing {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new("Processing...").italics());
                });
                if !self.log_messages.is_empty() {
                    egui::ScrollArea::vertical()
                        .id_source("log_scroll")
                        .max_height(150.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui: &mut egui::Ui| {
                            for line in &self.log_messages {
                                ui.label(egui::RichText::new(line).monospace().size(11.0));
                            }
                        });
                }
                ctx.request_repaint();
            }

            // --- Image Grid ---
            if !self.matched_images_cache.is_empty() && !self.is_processing {
                ui.separator();

                let selected_count = self.matched_images_cache.iter().filter(|(_, _, s, _)| *s).count();
                let total_pages = self.total_pages();
                let total_matches = self.all_ranked_matches.len();
                let page_start = self.current_page * self.page_size + 1;
                let page_end = ((self.current_page + 1) * self.page_size).min(total_matches);

                ui.label(format!(
                    "Page {}/{} — {} total candidates, showing {}-{} ({} selected on this page)",
                    self.current_page + 1, total_pages, total_matches, page_start, page_end, selected_count
                ));

                // Pagination controls
                ui.horizontal(|ui| {
                    let on_first = self.current_page == 0;
                    let on_last = self.current_page + 1 >= total_pages;
                    if ui.add_enabled(!on_first, egui::Button::new("< Prev")).clicked() {
                        let prev = self.current_page - 1;
                        self.load_page(prev);
                    }
                    if ui.add_enabled(!on_last, egui::Button::new("Next >")).clicked() {
                        let next = self.current_page + 1;
                        self.load_page(next);
                    }
                });

                // Pre-load any textures that haven't been loaded yet
                for (_img_path, _dist, _selected, texture_res_opt) in self.matched_images_cache.iter_mut() {
                    if texture_res_opt.is_none() {
                        let path = &*_img_path;
                        let load_path = if crate::utils::is_video(path) {
                            crate::utils::get_video_thumbnail_path(path)
                        } else {
                            path.to_path_buf()
                        };

                        match crate::utils::load_image_robustly(&load_path) {
                            Ok(img) => {
                                let size = [img.width() as usize, img.height() as usize];
                                let image_buffer = img.to_rgba8();
                                let pixels = image_buffer.as_flat_samples();
                                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
                                *texture_res_opt = Some(Ok(ctx.load_texture(
                                    _img_path.display().to_string(),
                                    color_image,
                                    egui::TextureOptions::LINEAR,
                                )));
                            }
                            Err(e) => {
                                *texture_res_opt = Some(Err(e.to_string()));
                            }
                        }
                    }
                }

                let cell_size = self.thumbnail_size + 8.0; // 8 = inner_margin 4px × 2
                let available_width = ui.available_width();
                let columns = ((available_width / cell_size).floor() as usize).max(1);
                let mut clicked_idx: Option<usize> = None;
                let mut new_person_trigger: Option<PathBuf> = None;

                let mut scroll_area = egui::ScrollArea::vertical();
                if self.scroll_to_top {
                    scroll_area = scroll_area.vertical_scroll_offset(0.0);
                    self.scroll_to_top = false;
                }
                scroll_area.show(ui, |ui| {
                    egui::Grid::new("matched_photos_grid")
                        .num_columns(columns)
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for (idx, (img_path, dist, selected, texture_res_opt)) in self.matched_images_cache.iter_mut().enumerate() {
                                let mut frame = egui::Frame::default().inner_margin(4.0);
                                if *selected {
                                    frame = frame.fill(egui::Color32::from_rgba_unmultiplied(0, 150, 255, 60))
                                                 .rounding(4.0);
                                } else {
                                    frame = frame.fill(egui::Color32::TRANSPARENT);
                                }

                                let resp = frame.show(ui, |ui| {
                                    let img_resp = match texture_res_opt {
                                        Some(Ok(texture)) => {
                                            let image = egui::Image::new(&*texture)
                                                .fit_to_exact_size(egui::vec2(self.thumbnail_size, self.thumbnail_size))
                                                .maintain_aspect_ratio(true)
                                                .sense(egui::Sense::click());
                                            let r = ui.add(image);
                                            if r.clicked() {
                                                clicked_idx = Some(idx);
                                            }
                                            r
                                        }
                                        Some(Err(_)) => {
                                            let label = egui::Label::new("⚠ Error")
                                                .sense(egui::Sense::click());
                                            let r = ui.add_sized([self.thumbnail_size, self.thumbnail_size], label);
                                            if r.clicked() {
                                                clicked_idx = Some(idx);
                                            }
                                            r
                                        }
                                        None => {
                                            ui.spinner()
                                        }
                                    };
                                    // Paint video icon if applicable
                                    if crate::utils::is_video(img_path) {
                                        let icon_pos = egui::pos2(
                                            img_resp.rect.right() - 4.0,
                                            img_resp.rect.top() + 4.0,
                                        );
                                        // Draw a semi-transparent background for the icon
                                        ui.painter().rect_filled(
                                            egui::Rect::from_min_size(
                                                egui::pos2(img_resp.rect.right() - 24.0, img_resp.rect.top() + 4.0),
                                                egui::vec2(20.0, 20.0)
                                            ),
                                            4.0,
                                            egui::Color32::from_black_alpha(128),
                                        );
                                        ui.painter().text(
                                            icon_pos,
                                            egui::Align2::RIGHT_TOP,
                                            "🎬",
                                            egui::FontId::proportional(14.0),
                                            egui::Color32::WHITE,
                                        );
                                    }

                                    // Paint distance as an overlay on the image so it doesn't
                                    // add any layout height/width to the cell.
                                    let text = format!("d={:.3}", dist);
                                    let font = egui::FontId::proportional(10.0);
                                    let text_pos = egui::pos2(
                                        img_resp.rect.left() + 3.0,
                                        img_resp.rect.bottom() - 14.0,
                                    );
                                    ui.painter().text(
                                        text_pos,
                                        egui::Align2::LEFT_TOP,
                                        &text,
                                        font,
                                        egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
                                    );
                                    img_resp
                                });

                                // Right-click context menu
                                resp.inner.context_menu(|ui| {
                                    if ui.button("📂 Open in Explorer").clicked() {
                                        #[cfg(target_os = "windows")]
                                        {
                                            let _ = std::process::Command::new("explorer")
                                                .args(["/select,", &img_path.to_string_lossy()])
                                                .spawn();
                                        }
                                        ui.close_menu();
                                    }
                                    ui.separator();
                                    let has_people_dir = self.target_dir.as_ref()
                                        .and_then(|td| td.parent())
                                        .is_some();
                                    if has_people_dir {
                                        if ui.button("➕ Create New Person + Add Image").clicked() {
                                            new_person_trigger = Some(img_path.clone());
                                            ui.close_menu();
                                        }
                                    } else {
                                        ui.add_enabled(false, egui::Button::new("➕ Create New Person (set target dir first)"));
                                    }
                                });

                                if (idx + 1) % columns == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                });

                // Process shift-click logic outside the grid
                if let Some(idx) = clicked_idx {
                    if ctx.input(|i| i.modifiers.shift) {
                        if let Some(last_idx) = self.last_selected_index {
                            let min_idx = std::cmp::min(last_idx, idx);
                            let max_idx = std::cmp::max(last_idx, idx);
                            let current_selection_state = self.matched_images_cache[idx].2;

                            for i in min_idx..=max_idx {
                                self.matched_images_cache[i].2 = !current_selection_state;
                            }
                        }
                    } else {
                        self.matched_images_cache[idx].2 = !self.matched_images_cache[idx].2;
                    }
                    self.last_selected_index = Some(idx);
                }

                // Handle "Create New Person" trigger from context menu
                if let Some(path) = new_person_trigger {
                    self.new_person_image_path = Some(path);
                    self.new_person_name.clear();
                    self.show_new_person_modal = true;
                }
            }
        });
    }
}

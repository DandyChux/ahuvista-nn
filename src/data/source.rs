use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use regex::Regex;
use tracing::{info, warn};

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceManifest {
    #[serde(default = "manifest_version")]
    pub version: u32,
    pub groups: Vec<SourceGroup>,
}
fn manifest_version() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceGroup {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub files: Vec<ManifestSource>,
    #[serde(default)]
    pub image_roots: Vec<ImageRoot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestSource {
    pub path: String,
    #[serde(default)]
    pub format: Option<InputFormat>,
    #[serde(default)]
    pub kind: Option<InputKind>,
    #[serde(default)]
    pub patient_id: Option<String>,
    #[serde(default)]
    pub mapping: Option<ColumnMapping>,
}

/// Optional semantic mapping for a tabular source. When omitted, CSV imports
/// continue to use the built-in column heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    #[serde(default)]
    pub patient_id_column: Option<String>,
    #[serde(default)]
    pub timestamp_column: Option<String>,
    #[serde(default)]
    pub disable_timestamp_detection: bool,
    #[serde(default)]
    pub note_column: Option<String>,
    #[serde(default)]
    pub note_type_column: Option<String>,
    #[serde(default)]
    pub outcome_column: Option<String>,
    #[serde(default)]
    pub ignored_columns: Vec<String>,
    #[serde(default)]
    pub feature_aliases: HashMap<String, String>,
    /// Column containing the event/code name for long-form event rows.
    #[serde(default)]
    pub event_feature_column: Option<String>,
    /// Optional numeric or categorical value column for long-form events.
    #[serde(default)]
    pub event_value_column: Option<String>,
    /// Optional source table/type column. When present, it prefixes the feature name.
    #[serde(default)]
    pub event_table_column: Option<String>,
    /// Optional visit/encounter column. When present, it is appended to the timestamp.
    #[serde(default)]
    pub event_visit_column: Option<String>,
    /// Value assigned when an event has no parseable numeric value.
    #[serde(default = "default_event_presence_value")]
    pub event_presence_value: f32,
}

fn default_event_presence_value() -> f32 {
    1.0
}

impl Default for ColumnMapping {
    fn default() -> Self {
        Self {
            patient_id_column: None,
            timestamp_column: None,
            disable_timestamp_detection: false,
            note_column: None,
            note_type_column: None,
            outcome_column: None,
            ignored_columns: Vec::new(),
            feature_aliases: HashMap::new(),
            event_feature_column: None,
            event_value_column: None,
            event_table_column: None,
            event_visit_column: None,
            event_presence_value: default_event_presence_value(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRoot {
    pub path: String,
    #[serde(default)]
    pub patient_id_pattern: Option<String>,
    #[serde(default)]
    pub patient_id_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InputFormat {
    Csv,
    Parquet,
    Json,
    Ndjson,
    FhirJson,
    FhirNdjson,
    Image,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    Patient,
    Observation,
    ClinicalNote,
    Image,
    ImageMetadata,
    Label,
    FhirResource,
    FhirBundle,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ClassifiedInput {
    pub path: PathBuf,
    pub format: InputFormat,
    pub kind: InputKind,
    pub patient_id: Option<String>,
    pub mapping: Option<ColumnMapping>,
    pub source: String,
    pub group: String,
}

pub fn load_manifest(path: &Path) -> AppResult<SourceManifest> {
    let content = fs::read_to_string(path).map_err(AppError::Io)?;
    let mut manifest: SourceManifest = serde_json::from_str(&content).map_err(AppError::JsonError)?;

    if manifest.version != 1 {
        return Err(AppError::InvalidData(format!(
            "unsupported source manifest version: {}",
            manifest.version
        )));
    }
    if manifest.groups.is_empty() {
        return Err(AppError::InvalidData(
            "manifest must define at least one source group".to_string(),
        ));
    }

    // Store paths canonically relative to the manifest directory. This lets a manifest
    // be passed from any working directory while keeping the file portable.
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    for group in &mut manifest.groups {
        group.path = resolve_manifest_path(base, &group.path);
        for source in &mut group.files {
            source.path = resolve_group_entry_path(Path::new(&group.path), &source.path);
        }
        for image_root in &mut group.image_roots {
            image_root.path = resolve_group_entry_path(Path::new(&group.path), &image_root.path);
        }
    }

    Ok(manifest)
}

fn resolve_manifest_path(base: &Path, value: &str) -> String {
    let path = Path::new(value);

    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else if value == "." {
    	base.to_path_buf()
    } else {
        base.join(path)
    };

    normalize_path(resolved).to_string_lossy().into_owned()
}

fn resolve_group_entry_path(group_path: &Path, value: &str) -> String {
    let path = Path::new(value);

    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else if value == "." {
        group_path.to_path_buf()
    } else if path.starts_with(group_path) {
        path.to_path_buf()
    } else {
        group_path.join(path)
    };

    normalize_path(resolved).to_string_lossy().into_owned()
}

pub fn manifest_image_roots(
    manifest: &SourceManifest,
    _dataset_root: &Path,
) -> Vec<(String, PathBuf, ImageRoot)> {
    manifest
        .groups
        .iter()
        .flat_map(|group| {
            group.image_roots.iter().map(|root| {
                (group.name.clone(), PathBuf::from(&root.path), root.clone())
            })
        })
        .collect()
}

pub fn classify_inputs(
    inputs: &[PathBuf],
    manifest: Option<&SourceManifest>,
) -> AppResult<Vec<ClassifiedInput>> {
    if let Some(manifest) = manifest {
        return classify_manifest(inputs, manifest);
    }

    let mut paths = Vec::new();

    for input in inputs {
        if input.is_file() {
            paths.push((input.clone(), None, source_group_name(input, inputs)));
        } else if input.is_dir() {
            collect_files(input, &mut paths, input)?;
        }
    }

    classify_paths(paths)
}

fn classify_manifest(
    inputs: &[PathBuf],
    manifest: &SourceManifest,
) -> AppResult<Vec<ClassifiedInput>> {
    let mut paths = Vec::new();

    for group in &manifest.groups {
	    let group_root = PathBuf::from(&group.path);

        if !group_root.exists() {
            return Err(AppError::InvalidData(format!(
                "source group does not exist: {}",
                group.path
            )));
        }

        if group.files.is_empty() {
            collect_files(&group_root, &mut paths, &group_root)?;
        }

        for source in &group.files {
            let path = PathBuf::from(&source.path);

            if path.is_dir() {
                collect_declared_files(&path, source, &group.name, &mut paths)?;
            } else if path.is_file() {
                paths.push((path, Some(source.clone()), group.name.clone()));
            } else {
                return Err(AppError::InvalidData(format!(
                    "manifest source does not exist: {}",
                    path.display()
                )));
            }
        }
    }

    classify_paths(paths)
}

fn classify_paths(
    paths: Vec<(PathBuf, Option<ManifestSource>, String)>,
) -> AppResult<Vec<ClassifiedInput>> {
    paths
        .into_iter()
        .map(|(path, declared, group)| classify_one(&path, declared.as_ref(), group))
        .filter_map(|result| match result {
            Ok(input)
                if input.kind != InputKind::Unknown && input.format != InputFormat::Unknown =>
            {
                info!(
                    "Classified {} as {:?}/{:?} in group {}",
                    input.path.display(),
                    input.kind,
                    input.format,
                    input.group
                );
                Some(Ok(input))
            }
            Ok(input) => {
                warn!("Skipping unclassified input: {}", input.path.display());
                None
            }
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn collect_declared_files(
    path: &Path,
    declaration: &ManifestSource,
    group: &str,
    output: &mut Vec<(PathBuf, Option<ManifestSource>, String)>,
) -> AppResult<()> {
    for entry in fs::read_dir(path).map_err(AppError::Io)? {
        let child = entry.map_err(AppError::Io)?.path();
        if child.is_dir() {
            collect_declared_files(&child, declaration, group, output)?;
        } else {
            output.push((child, Some(declaration.clone()), group.to_string()));
        }
    }
    Ok(())
}

fn collect_files(
    path: &Path,
    output: &mut Vec<(PathBuf, Option<ManifestSource>, String)>,
    group_root: &Path,
) -> AppResult<()> {
    let group = group_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("root")
        .to_string();

    for entry in fs::read_dir(path).map_err(AppError::Io)? {
        let path = entry.map_err(AppError::Io)?.path();
        if path.is_dir() {
            collect_files(&path, output, group_root)?;
        } else {
            output.push((path, None, group.clone()));
        }
    }

    Ok(())
}

fn source_group_name(path: &Path, inputs: &[PathBuf]) -> String {
    for input in inputs {
        if input.is_dir() && path.starts_with(input) {
            return path
                .strip_prefix(input)
                .ok()
                .and_then(|relative| relative.components().next())
                .and_then(|component| component.as_os_str().to_str())
                .unwrap_or("root")
                .to_string();
        }
    }

    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("root")
        .to_string()
}

fn classify_one(
    path: &Path,
    declared: Option<&ManifestSource>,
    group: String,
) -> AppResult<ClassifiedInput> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut format = declared
        .and_then(|source| source.format.clone())
        .unwrap_or_else(|| match extension.as_str() {
            "csv" => InputFormat::Csv,
            "parquet" => InputFormat::Parquet,
            "json" => InputFormat::Json,
            "ndjson" | "jsonl" => InputFormat::Ndjson,
            "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "gif" | "dcm" => InputFormat::Image,
            _ => InputFormat::Unknown,
        });
    let mut kind = declared
        .and_then(|source| source.kind.clone())
        .unwrap_or(InputKind::Unknown);

    if matches!(format, InputFormat::Json | InputFormat::Ndjson) {
        let content = fs::read_to_string(path).map_err(AppError::Io)?;

        if format == InputFormat::Json {
            let value: Value = serde_json::from_str(&content).map_err(AppError::JsonError)?;

            if value.get("resourceType").and_then(Value::as_str) == Some("Bundle") {
                format = InputFormat::FhirJson;
                kind = InputKind::FhirBundle;
            } else if value.get("resourceType").is_some() {
                format = InputFormat::FhirJson;
                kind = InputKind::FhirResource;
            }
        } else if content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .any(|line| {
                serde_json::from_str::<Value>(line)
                    .ok()
                    .and_then(|value| value.get("resourceType").cloned())
                    .is_some()
            })
        {
            format = InputFormat::FhirNdjson;
            kind = InputKind::FhirResource;
        }
    }

    if kind == InputKind::Unknown {
        kind = match format {
            InputFormat::Image => InputKind::Image,
            InputFormat::Csv | InputFormat::Parquet => classify_table_headers(path, &format)?,
            _ => InputKind::Unknown,
        };
    }

    Ok(ClassifiedInput {
        path: path.to_path_buf(),
        format,
        kind,
        patient_id: declared.and_then(|source| source.patient_id.clone()),
        mapping: declared.and_then(|source| source.mapping.clone()),
        source: path.display().to_string(),
        group,
    })
}

fn classify_table_headers(path: &Path, format: &InputFormat) -> AppResult<InputKind> {
    if *format != InputFormat::Csv {
        return Ok(InputKind::Unknown);
    }

    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.iter().map(normalize).collect::<Vec<_>>();

    let has_patient = headers.iter().any(|header| {
        matches!(
            header.as_str(),
            "patientid" | "patient_id" | "subject" | "subjectid"
        )
    });

    let has_observation = headers.iter().any(|header| {
        matches!(
            header.as_str(),
            "observation" | "measurement" | "measurementtype" | "value" | "loinc"
        )
    });

    let has_note = headers.iter().any(|header| {
        matches!(
            header.as_str(),
            "note" | "clinicalnote" | "document" | "text"
        )
    });

    let has_image = headers.iter().any(|header| {
        matches!(
            header.as_str(),
            "image" | "imagepath" | "dicom" | "sopinstanceuid"
        )
    });

    if has_image && !has_patient {
        Ok(InputKind::ImageMetadata)
    } else if has_note {
        Ok(InputKind::ClinicalNote)
    } else if has_observation && has_patient {
        Ok(InputKind::Observation)
    } else if has_patient {
        Ok(InputKind::Patient)
    } else {
        Ok(InputKind::Unknown)
    }
}

pub fn discover_associated_images(
    root: &Path,
    image_root: &ImageRoot,
) -> AppResult<Vec<(String, PathBuf)>> {
    let mut paths = Vec::new();

    collect_image_files(root, &mut paths)?;

    let mut records = Vec::new();

    for path in paths {
	    let patient_id = infer_patient_id(root, &path, image_root)?
	        .ok_or_else(|| {
	            AppError::InvalidData(format!(
	                "cannot associate image {}; configure a matching patient_id_pattern or patient-folder rule",
	                path.display()
	            ))
	        })?;

        records.push((patient_id, path));
    }

    records.sort_by(|left, right| left.1.cmp(&right.1));

    Ok(records)
}

fn collect_image_files(root: &Path, output: &mut Vec<PathBuf>) -> AppResult<()> {
    for entry in fs::read_dir(root).map_err(AppError::Io)? {
        let path = entry.map_err(AppError::Io)?.path();

        if path.is_dir() {
            collect_image_files(&path, output)?;
        } else if is_associated_image_file(&path) {
            output.push(path);
        }
    }

    Ok(())
}

fn is_associated_image_file(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    matches!(
        extension.as_str(),
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "gif" | "dcm" | "dat" | "edf"
    )
}

fn infer_patient_id(
    root: &Path,
    image: &Path,
    config: &ImageRoot,
) -> AppResult<Option<String>> {
    let relative = image
        .strip_prefix(root)
        .map_err(|_| {
            AppError::InvalidData(format!(
                "image {} is outside image root {}",
                image.display(),
                root.display()
            ))
        })?;

    if let Some(pattern) = &config.patient_id_pattern {
        let regex = Regex::new(pattern).map_err(|error| {
            AppError::InvalidData(format!(
                "invalid patient_id_pattern {:?}: {}",
                pattern, error
            ))
        })?;

        let file_name = relative
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                AppError::InvalidData(format!(
                    "image filename is not valid UTF-8: {}",
                    image.display()
                ))
            })?;

        return Ok(regex
            .captures(file_name)
            .and_then(|captures| captures.get(1))
            .map(|capture| capture.as_str().to_string())
            .filter(|value| !value.is_empty()));
    }

    let components = relative
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();

    if let Some(depth) = config.patient_id_depth {
        let index = match components.len().checked_sub(depth) {
            Some(index) => index,
            None => return Ok(None),
        };

        return Ok(components
            .get(index)
            .map(|value| (*value).to_string()));
    }

    Ok(components
        .last()
        .map(|value| (*value).to_string())
        .filter(|value| !value.is_empty()))
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component),
        }
    }

    normalized
}

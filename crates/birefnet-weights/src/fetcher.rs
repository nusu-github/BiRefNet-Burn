//! Remote and test weight-fetching abstractions.

#[cfg(any(test, feature = "test-util"))]
use std::collections::BTreeMap;
use std::{fmt, path::PathBuf};

use crate::error::WeightError;

/// Resolves remote repository artifacts to local filesystem paths.
pub trait Fetcher: fmt::Debug + Send + Sync {
    /// Ensures `filename` from `repo_id` is available locally and returns its path.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if downloading or resolving the file fails.
    fn fetch(&self, repo_id: &str, filename: &str) -> Result<PathBuf, WeightError>;

    /// Returns `true` if `filename` from `repo_id` is already available in local cache.
    fn is_cached(&self, repo_id: &str, filename: &str) -> bool;
}

/// Synchronous `HuggingFace` Hub checkpoint fetcher backed by `hf-hub`.
#[cfg(feature = "hub")]
#[derive(Clone)]
pub struct HfHubFetcher {
    client: hf_hub::HFClientSync,
}

#[cfg(feature = "hub")]
impl fmt::Debug for HfHubFetcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfHubFetcher").finish_non_exhaustive()
    }
}

#[cfg(feature = "hub")]
impl HfHubFetcher {
    /// Initializes a blocking `HuggingFace` Hub client.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if the underlying Hub client cannot be created.
    pub fn new() -> Result<Self, WeightError> {
        let client = hf_hub::HFClientSync::new()?;
        Ok(Self { client })
    }
}

#[cfg(feature = "hub")]
impl Fetcher for HfHubFetcher {
    fn fetch(&self, repo_id: &str, filename: &str) -> Result<PathBuf, WeightError> {
        let (owner, name) = hf_hub::split_id(repo_id);
        let path = self
            .client
            .model(owner, name)
            .download_file()
            .filename(filename)
            .send()?;
        Ok(path)
    }

    fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        let (owner, name) = hf_hub::split_id(repo_id);
        self.client
            .model(owner, name)
            .download_file()
            .filename(filename)
            .local_files_only(true)
            .send()
            .is_ok()
    }
}

/// Offline in-memory lookup table mapping `(repo_id, filename)` pairs to local test files.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug, Default, Clone)]
pub struct MockFetcher {
    files: BTreeMap<(String, String), PathBuf>,
}

#[cfg(any(test, feature = "test-util"))]
impl MockFetcher {
    /// Creates an empty [`MockFetcher`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a local checkpoint `path` for `(repo_id, filename)`.
    #[must_use]
    pub fn with_file(
        mut self,
        repo_id: impl Into<String>,
        filename: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.files
            .insert((repo_id.into(), filename.into()), path.into());
        self
    }
}

#[cfg(any(test, feature = "test-util"))]
impl Fetcher for MockFetcher {
    fn fetch(&self, repo_id: &str, filename: &str) -> Result<PathBuf, WeightError> {
        self.files
            .get(&(repo_id.to_owned(), filename.to_owned()))
            .cloned()
            .ok_or_else(|| WeightError::not_found(&PathBuf::from(format!("{repo_id}/{filename}"))))
    }

    fn is_cached(&self, repo_id: &str, filename: &str) -> bool {
        self.files
            .get(&(repo_id.to_owned(), filename.to_owned()))
            .is_some_and(|path| path.exists())
    }
}

// Copyright (C) 2021 Quickwit, Inc.
//
// Quickwit is offered under the AGPL v3.0 and as commercial software.
// For commercial licensing, contact us at hello@quickwit.io.
//
// AGPL:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#[cfg(feature = "postgres")]
pub mod postgresql_metastore;
pub mod single_file_metastore;

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use quickwit_config::{DocMapping, IndexingSettings, SearchSettings, SourceConfig};
use quickwit_index_config::IndexConfig as LegacyIndexConfig;
use serde::{Deserialize, Serialize};

use crate::checkpoint::{Checkpoint, CheckpointDelta};
use crate::{MetastoreResult, SplitMetadataAndFooterOffsets, SplitState};

/// An index metadata carries all meta data about an index.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(into = "VersionedIndexMetadata")]
pub struct IndexMetadata {
    pub index_id: String,
    pub index_uri: String,
    pub checkpoint: Checkpoint,
    pub doc_mapping: DocMapping,
    pub indexing_settings: IndexingSettings,
    pub search_settings: SearchSettings,
    pub sources: Vec<SourceConfig>,
    pub create_timestamp: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct IndexMetadataV0 {
    pub index_id: String,
    pub index_uri: String,
    pub index_config: Arc<dyn LegacyIndexConfig>,
    pub checkpoint: Checkpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct IndexMetadataV1 {
    pub index_id: String,
    pub index_uri: String,
    pub checkpoint: Checkpoint,
    pub doc_mapping: DocMapping,
    pub indexing_settings: IndexingSettings,
    pub search_settings: SearchSettings,
    pub sources: Vec<SourceConfig>,
    pub create_timestamp: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
pub(crate) enum VersionedIndexMetadata {
    #[serde(rename = "1")]
    V1(IndexMetadataV1),
    #[serde(rename = "0")]
    V0(IndexMetadataV0),
}

impl From<IndexMetadata> for VersionedIndexMetadata {
    fn from(index_metadata: IndexMetadata) -> Self {
        VersionedIndexMetadata::V1(index_metadata.into())
    }
}

impl From<IndexMetadata> for IndexMetadataV1 {
    fn from(index_metadata: IndexMetadata) -> Self {
        Self {
            index_id: index_metadata.index_id,
            index_uri: index_metadata.index_uri,
            checkpoint: index_metadata.checkpoint,
            doc_mapping: index_metadata.doc_mapping,
            indexing_settings: index_metadata.indexing_settings,
            search_settings: index_metadata.search_settings,
            sources: index_metadata.sources,
            create_timestamp: index_metadata.create_timestamp,
        }
    }
}

impl From<VersionedIndexMetadata> for IndexMetadata {
    fn from(index_metadata: VersionedIndexMetadata) -> Self {
        match index_metadata {
            VersionedIndexMetadata::V0(v0_index_metadata) => Self {
                index_id: v0_index_metadata.index_id,
                index_uri: v0_index_metadata.index_uri,
                checkpoint: v0_index_metadata.checkpoint,
                doc_mapping: v0_index_metadata.index_config.into(),
                indexing_settings: IndexingSettings::default(), // FIXME
                search_settings: SearchSettings::default(),     // FIXME
                sources: Vec::new(),
                create_timestamp: Utc::now().timestamp(),
            },
            VersionedIndexMetadata::V1(v1_index_metadata) => Self {
                index_id: v1_index_metadata.index_id,
                index_uri: v1_index_metadata.index_uri,
                checkpoint: v1_index_metadata.checkpoint,
                doc_mapping: v1_index_metadata.doc_mapping,
                indexing_settings: v1_index_metadata.indexing_settings,
                search_settings: v1_index_metadata.search_settings,
                sources: v1_index_metadata.sources,
                create_timestamp: v1_index_metadata.create_timestamp,
            },
        }
    }
}

// TODO: Implement custom deserializer for missing version field.

/// Metastore meant to manage Quickwit's indexes and their splits.
///
/// Quickwit needs a way to ensure that we can cleanup unused files,
/// and this process needs to be resilient to any fail-stop failures.
/// We rely on atomically transitioning the status of splits.
///
/// The split state goes through the following life cycle:
/// 1. `New`
///   - Create new split and start indexing.
/// 2. `Staged`
///   - Start uploading the split files.
/// 3. `Published`
///   - Uploading the split files is complete and the split is searchable.
/// 4. `ScheduledForDeletion`
///   - Mark the split for deletion.
///
/// If a split has a file in the storage, it MUST be registered in the metastore,
/// and its state can be as follows:
/// - `Staged`: The split is almost ready. Some of its files may have been uploaded in the storage.
/// - `Published`: The split is ready and published.
/// - `ScheduledForDeletion`: The split is scheduled for deletion.
///
/// Before creating any file, we need to stage the split. If there is a failure, upon recovery, we
/// schedule for deletion all the staged splits. A client may not necessarily remove files from
/// storage right after marking it for deletion. A CLI client may delete files right away, but a
/// more serious deployment should probably only delete those files after a grace period so that the
/// running search queries can complete.
#[cfg_attr(any(test, feature = "testsuite"), mockall::automock)]
#[async_trait]
pub trait Metastore: Send + Sync + 'static {
    /// Checks if the metastore is available.
    async fn check_connectivity(&self) -> anyhow::Result<()>;

    /// Checks if the given index is in this metastore.
    async fn check_index_available(&self, index_id: &str) -> anyhow::Result<()> {
        self.index_metadata(index_id).await?;
        Ok(())
    }

    /// Creates an index.
    /// This API creates index metadata set in the metastore.
    /// An error will occur if an index that already exists in the storage is specified.
    async fn create_index(&self, index_metadata: IndexMetadata) -> MetastoreResult<()>;

    /// Returns the index_metadata for a given index.
    ///
    /// TODO consider merging with list_splits to remove one round-trip
    async fn index_metadata(&self, index_id: &str) -> MetastoreResult<IndexMetadata>;

    /// Deletes an index.
    /// This API removes the specified index metadata set from the metastore,
    /// but does not remove the index from the storage.
    /// An error will occur if an index that does not exist in the storage is specified.
    async fn delete_index(&self, index_id: &str) -> MetastoreResult<()>;

    /// Stages a split.
    /// A split needs to be staged before uploading any of its files to the storage.
    /// An error will occur if an index that does not exist in the storage is specified.
    /// Also, an error will occur if you specify a split that already exists.
    async fn stage_split(
        &self,
        index_id: &str,
        split_metadata: SplitMetadataAndFooterOffsets,
    ) -> MetastoreResult<()>;

    /// Publishes a list splits.
    /// This API only updates the state of the split from `Staged` to `Published`.
    /// At this point, the split files are assumed to have already been uploaded.
    /// If the split is already published, this API call returns a success.
    /// An error will occur if you specify an index or split that does not exist in the storage.
    async fn publish_splits<'a>(
        &self,
        index_id: &str,
        split_ids: &[&'a str],
        checkpoint_delta: CheckpointDelta,
    ) -> MetastoreResult<()>;

    /// Replaces a list of splits with another list.
    /// This API is useful during merge and demux operations.
    /// The new splits should be staged, and the replaced splits should exist.
    async fn replace_splits<'a>(
        &self,
        index_id: &str,
        new_split_ids: &[&'a str],
        replaced_split_ids: &[&'a str],
    ) -> MetastoreResult<()>;

    /// Lists the splits.
    /// Returns a list of splits that intersects the given `time_range`, `split_state` and `tag`.
    /// Regardless of the time range filter, if a split has no timestamp it is always returned.
    /// An error will occur if an index that does not exist in the storage is specified.
    async fn list_splits(
        &self,
        index_id: &str,
        split_state: SplitState,
        time_range: Option<Range<i64>>,
        tags: &[String],
    ) -> MetastoreResult<Vec<SplitMetadataAndFooterOffsets>>;

    /// Lists the splits without filtering.
    /// Returns a list of all splits currently known to the metastore regardless of their state.
    async fn list_all_splits(
        &self,
        index_id: &str,
    ) -> MetastoreResult<Vec<SplitMetadataAndFooterOffsets>>;

    /// Marks a list of splits for deletion.
    /// This API will change the state to `ScheduledForDeletion` so that it is not referenced by the
    /// client. It actually does not remove the split from storage.
    /// An error will occur if you specify an index or split that does not exist in the storage.
    async fn mark_splits_for_deletion<'a>(
        &self,
        index_id: &str,
        split_ids: &[&'a str],
    ) -> MetastoreResult<()>;

    /// Deletes a list of splits.
    /// This API only takes a split that is in `Staged` or `ScheduledForDeletion` state.
    /// This removes the split metadata from the metastore, but does not remove the split from
    /// storage. An error will occur if you specify an index or split that does not exist in the
    /// storage.
    async fn delete_splits<'a>(&self, index_id: &str, split_ids: &[&'a str])
        -> MetastoreResult<()>;

    /// Returns the Metastore uri.
    fn uri(&self) -> String;
}

// Returns true if filter_tags is empty (unspecified),
// or if filter_tags is specified and split_tags contains at least one of the tags in filter_tags.
pub fn match_tags_filter(split_tags: &[String], filter_tags: &[String]) -> bool {
    if filter_tags.is_empty() {
        return true;
    }
    for filter_tag in filter_tags {
        if split_tags.contains(filter_tag) {
            return true;
        }
    }
    false
}

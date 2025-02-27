// Copyright (C) 2022 Quickwit, Inc.
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

use std::collections::HashMap;

use quickwit_config::{IndexConfig, SourceConfig};
use serde::{self, Deserialize, Serialize};

use crate::checkpoint::IndexCheckpoint;
use crate::split_metadata::utc_now_timestamp;
use crate::IndexMetadata;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
pub(crate) enum VersionedIndexMetadata {
    #[serde(rename = "3")]
    V3(IndexMetadataV3),
}

impl From<IndexMetadata> for VersionedIndexMetadata {
    fn from(index_metadata: IndexMetadata) -> Self {
        VersionedIndexMetadata::V3(index_metadata.into())
    }
}

impl TryFrom<VersionedIndexMetadata> for IndexMetadata {
    type Error = anyhow::Error;

    fn try_from(index_metadata: VersionedIndexMetadata) -> anyhow::Result<Self> {
        match index_metadata {
            // When we have more than one version, you should chain version conversion.
            // ie. Implement conversion from V_k -> V_{k+1}
            VersionedIndexMetadata::V3(v3) => v3.try_into(),
        }
    }
}

impl From<IndexMetadata> for IndexMetadataV3 {
    fn from(index_metadata: IndexMetadata) -> Self {
        let sources: Vec<SourceConfig> = index_metadata.sources.values().cloned().collect();
        Self {
            index_config: index_metadata.index_config,
            checkpoint: index_metadata.checkpoint,
            create_timestamp: index_metadata.create_timestamp,
            update_timestamp: index_metadata.update_timestamp,
            sources,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct IndexMetadataV3 {
    pub index_config: IndexConfig,
    pub checkpoint: IndexCheckpoint,
    #[serde(default = "utc_now_timestamp")]
    pub create_timestamp: i64,
    #[serde(default = "utc_now_timestamp")]
    pub update_timestamp: i64,
    pub sources: Vec<SourceConfig>,
}

impl TryFrom<IndexMetadataV3> for IndexMetadata {
    type Error = anyhow::Error;

    fn try_from(v3: IndexMetadataV3) -> anyhow::Result<Self> {
        let mut sources: HashMap<String, SourceConfig> = Default::default();
        for source in v3.sources {
            if sources.contains_key(&source.source_id) {
                anyhow::bail!("Source `{}` is defined more than once", source.source_id);
            }
            sources.insert(source.source_id.clone(), source);
        }
        Ok(Self {
            index_config: v3.index_config,
            checkpoint: v3.checkpoint,
            create_timestamp: v3.create_timestamp,
            update_timestamp: v3.update_timestamp,
            sources,
        })
    }
}

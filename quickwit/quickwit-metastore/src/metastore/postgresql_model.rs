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

use std::convert::TryInto;
use std::str::FromStr;

use quickwit_proto::metastore_api::{DeleteQuery, DeleteTask as QuickwitDeleteTask};
use tracing::error;

use crate::{
    IndexMetadata, MetastoreError, MetastoreResult, Split as QuickwitSplit, SplitMetadata,
    SplitState,
};

#[derive(sqlx::FromRow, Debug)]
pub struct IndexIdSplitIdRow {
    pub index_id: String,
    pub split_id: Option<String>, // TODO can we get rid of option
}

/// A model structure for handling index metadata in a database.
#[derive(sqlx::FromRow)]
pub struct Index {
    /// Index ID. The index ID identifies the index when querying the metastore.
    pub index_id: String,
    // A JSON string containing all of the IndexMetadata.
    pub index_metadata_json: String,
    /// Timestamp for tracking when the split was created.
    pub create_timestamp: sqlx::types::time::PrimitiveDateTime,
    /// Timestamp for tracking when the split was last updated.
    pub update_timestamp: sqlx::types::time::PrimitiveDateTime,
}

impl Index {
    /// Deserializes index metadata from JSON string stored in column and sets appropriate
    /// timestamps.
    pub fn index_metadata(&self) -> MetastoreResult<IndexMetadata> {
        let mut index_metadata = serde_json::from_str::<IndexMetadata>(&self.index_metadata_json)
            .map_err(|error| {
            error!(index_id=%self.index_id, error=?error, "Failed to deserialize index metadata.");

            MetastoreError::JsonDeserializeError {
                struct_name: "IndexMetadata".to_string(),
                message: error.to_string(),
            }
        })?;
        // `create_timestamp` and `update_timestamp` are stored in dedicated columns but are also
        // duplicated in [`IndexMetadata`]. We must override the duplicates with the authentic
        // values upon deserialization.
        index_metadata.create_timestamp = self.create_timestamp.assume_utc().unix_timestamp();
        index_metadata.update_timestamp = self.update_timestamp.assume_utc().unix_timestamp();
        Ok(index_metadata)
    }
}

/// A model structure for handling split metadata in a database.
#[derive(sqlx::FromRow)]
pub struct Split {
    /// Split ID.
    pub split_id: String,
    /// The state of the split. With `update_timestamp`, this is the only mutable attribute of the
    /// split.
    pub split_state: String,
    /// If a timestamp field is available, the min timestamp of the split.
    pub time_range_start: Option<i64>,
    /// If a timestamp field is available, the max timestamp of the split.
    pub time_range_end: Option<i64>,
    /// Timestamp for tracking when the split was created.
    pub create_timestamp: sqlx::types::time::PrimitiveDateTime,
    /// Timestamp for tracking when the split was last updated.
    pub update_timestamp: sqlx::types::time::PrimitiveDateTime,
    /// Timestamp for tracking when the split was published.
    pub publish_timestamp: Option<sqlx::types::time::PrimitiveDateTime>,
    /// A list of tags for categorizing and searching group of splits.
    pub tags: Vec<String>,
    // The split's metadata serialized as a JSON string.
    pub split_metadata_json: String,
    /// Index ID. It is used as a foreign key in the database.
    pub index_id: String,
    /// Delete opstamp.
    pub delete_opstamp: i64,
}

impl Split {
    /// Deserializes and returns the split's metadata.
    fn split_metadata(&self) -> MetastoreResult<SplitMetadata> {
        serde_json::from_str::<SplitMetadata>(&self.split_metadata_json).map_err(|error| {
            error!(index_id=%self.index_id, split_id=%self.split_id, error=?error, "Failed to deserialize split metadata.");

            MetastoreError::JsonDeserializeError {
                struct_name: "SplitMetadata".to_string(),
                message: error.to_string(),
            }
        })
    }

    /// Deserializes and returns the split's state.
    fn split_state(&self) -> MetastoreResult<SplitState> {
        SplitState::from_str(&self.split_state).map_err(|error| {
            error!(index_id=%self.index_id, split_id=%self.split_id, split_state=?self.split_state, error=?error, "Failed to deserialize split state.");

            MetastoreError::JsonDeserializeError {
                struct_name: "SplitState".to_string(),
                message: error,
            }
        })
    }
}

impl TryInto<QuickwitSplit> for Split {
    type Error = MetastoreError;

    fn try_into(self) -> Result<QuickwitSplit, Self::Error> {
        let mut split_metadata = self.split_metadata()?;
        // `create_timestamp` and `delete_opstamp` are duplicated in `SplitMetadata` and needs to be
        // overridden with the "true" value stored in a column.
        split_metadata.create_timestamp = self.create_timestamp.assume_utc().unix_timestamp();
        let split_state = self.split_state()?;
        let update_timestamp = self.update_timestamp.assume_utc().unix_timestamp();
        let publish_timestamp = self
            .publish_timestamp
            .map(|publish_timestamp| publish_timestamp.assume_utc().unix_timestamp());
        split_metadata.index_id = self.index_id;
        split_metadata.delete_opstamp = self.delete_opstamp as u64;
        Ok(QuickwitSplit {
            split_metadata,
            split_state,
            update_timestamp,
            publish_timestamp,
        })
    }
}

/// A model structure for handling split metadata in a database.
#[derive(sqlx::FromRow)]
pub struct DeleteTask {
    /// Create timestamp.
    pub create_timestamp: sqlx::types::time::PrimitiveDateTime,
    /// Monotonic increasing unique opstamp.
    pub opstamp: i64,
    /// Index id.
    pub index_id: String,
    /// Query serialized as a JSON string.
    pub delete_query_json: String,
}

impl DeleteTask {
    /// Deserializes and returns the split's metadata.
    fn delete_query(&self) -> MetastoreResult<DeleteQuery> {
        serde_json::from_str::<DeleteQuery>(&self.delete_query_json).map_err(|error| {
            error!(index_id=%self.index_id, opstamp=%self.opstamp, error=?error, "Failed to deserialize delete query.");

            MetastoreError::JsonDeserializeError {
                struct_name: "DeleteQuery".to_string(),
                message: error.to_string(),
            }
        })
    }
}

impl TryInto<QuickwitDeleteTask> for DeleteTask {
    type Error = MetastoreError;

    fn try_into(self) -> Result<QuickwitDeleteTask, Self::Error> {
        let delete_query = self.delete_query()?;
        Ok(QuickwitDeleteTask {
            create_timestamp: self.create_timestamp.assume_utc().unix_timestamp(),
            opstamp: self.opstamp as u64,
            delete_query: Some(delete_query),
        })
    }
}

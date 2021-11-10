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

use async_trait::async_trait;
use quickwit_actors::{ActorExitStatus, Mailbox, HEARTBEAT};
use serde::{Deserialize, Serialize};

use crate::models::IndexerMessage;
use crate::source::{Source, SourceContext, TypedSourceFactory};

pub struct EmptySource;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FilePosition {
    pub num_bytes: u64,
}

#[async_trait]
impl Source for EmptySource {
    async fn emit_batches(
        &mut self,
        _: &Mailbox<IndexerMessage>,
        _: &SourceContext,
    ) -> Result<(), ActorExitStatus> {
        tokio::time::sleep(HEARTBEAT / 2).await;
        Ok(())
    }

    fn name(&self) -> String {
        "EmptySource".to_string()
    }

    fn observable_state(&self) -> serde_json::Value {
        serde_json::to_value(0).unwrap()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EmptySourceParams {}

pub struct EmptySourceFactory;

#[async_trait]
impl TypedSourceFactory for EmptySourceFactory {
    type Source = EmptySource;

    type Params = EmptySourceParams;

    async fn typed_create_source(
        _: EmptySourceParams,
        _: quickwit_metastore::checkpoint::Checkpoint,
    ) -> anyhow::Result<EmptySource> {
        Ok(EmptySource)
    }
}

#[cfg(test)]
mod tests {
    use quickwit_actors::{create_test_mailbox, Health, Supervisable, Universe};
    use quickwit_metastore::checkpoint::Checkpoint;
    use serde_json::json;

    use super::*;
    use crate::source::{quickwit_supported_sources, SourceActor, SourceConfig};

    #[tokio::test]
    async fn test_empty_source_loading() -> anyhow::Result<()> {
        let source_config = SourceConfig {
            source_id: "empty-test-source".to_string(),
            source_type: "empty".to_string(),
            params: json!({}),
        };
        let source_loader = quickwit_supported_sources();
        let _ = source_loader
            .load_source(source_config.clone(), Checkpoint::default())
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_source_running() -> anyhow::Result<()> {
        quickwit_common::setup_logging_for_tests();
        let universe = Universe::new();
        let (mailbox, _) = create_test_mailbox();
        let empty_source =
            EmptySourceFactory::typed_create_source(EmptySourceParams {}, Checkpoint::default())
                .await?;
        let empty_source_actor = SourceActor {
            source: Box::new(empty_source),
            batch_sink: mailbox,
        };
        let (_, empty_source_handle) = universe.spawn_actor(empty_source_actor).spawn_async();
        matches!(empty_source_handle.health(), Health::Healthy);
        let (actor_termination, observed_state) = empty_source_handle.quit().await;
        assert_eq!(observed_state, serde_json::to_value(0).unwrap());
        matches!(actor_termination, ActorExitStatus::Quit);
        Ok(())
    }
}

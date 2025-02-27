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

use std::collections::{HashSet, VecDeque};
use std::fmt::Display;
use std::io::{stdout, Stdout, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};
use std::{env, fmt, io};

use anyhow::{bail, Context};
use clap::{arg, ArgMatches, Command};
use colored::{ColoredString, Colorize};
use humantime::format_duration;
use itertools::Itertools;
use quickwit_actors::{ActorHandle, ObservationType, Universe};
use quickwit_common::uri::Uri;
use quickwit_common::GREEN_COLOR;
use quickwit_config::service::QuickwitService;
use quickwit_config::{
    ConfigFormat, IndexConfig, IndexerConfig, SourceConfig, SourceParams, CLI_INGEST_SOURCE_ID,
    INGEST_API_SOURCE_ID,
};
use quickwit_core::{
    clear_cache_directory, remove_indexing_directory, validate_storage_uri, IndexService,
};
use quickwit_indexing::actors::{IndexingPipeline, IndexingService};
use quickwit_indexing::models::{
    DetachPipeline, IndexingPipelineId, IndexingStatistics, SpawnMergePipeline, SpawnPipeline,
};
use quickwit_metastore::{
    quickwit_metastore_uri_resolver, IndexMetadata, ListSplitsQuery, Split, SplitState,
};
use quickwit_proto::{SearchRequest, SearchResponse};
use quickwit_search::{single_node_search, SearchResponseRest};
use quickwit_storage::{load_file, quickwit_storage_uri_resolver};
use quickwit_telemetry::payload::TelemetryEvent;
use tabled::object::{Columns, Segment};
use tabled::{Alignment, Concat, Format, Modify, Panel, Rotate, Style, Table, Tabled};
use thousands::Separable;
use tracing::{debug, warn, Level};

use crate::stats::{mean, percentile, std_deviation};
use crate::{
    load_quickwit_config, make_table, parse_duration_with_unit, prompt_confirmation,
    run_index_checklist, start_actor_runtimes, THROUGHPUT_WINDOW_SIZE,
};

pub fn build_index_command<'a>() -> Command<'a> {
    Command::new("index")
        .about("Create your index, ingest data, search, describe... every command you need to manage indexes.")
        .subcommand(
            Command::new("list")
                .about("List indexes.")
                .alias("ls")
                .args(&[
                    arg!(--"metastore-uri" <METASTORE_URI> "Metastore URI. Override the `metastore_uri` parameter defined in the config file. Defaults to file-backed, but could be Amazon S3 or PostgreSQL.")
                        .required(false)
                ])
            )
        .subcommand(
            Command::new("create")
                .about("Creates an index from an index config file.")
                .args(&[
                    arg!(--"index-config" <INDEX_CONFIG> "Location of the index config file."),
                    arg!(--overwrite "Overwrites pre-existing index. This will delete all existing data stored at `index-uri` before creating a new index.")
                        .required(false),
                    arg!(-y --"yes" "Assume \"yes\" as an answer to all prompts and run non-interactively.")
                        .required(false),
                ])
            )
        .subcommand(
            Command::new("ingest")
                .about("Indexes JSON documents read from a file or streamed from stdin.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                    arg!(--"input-path" <INPUT_PATH> "Location of the input file.")
                        .required(false),
                    arg!(--overwrite "Overwrites pre-existing index.")
                        .required(false),
                    arg!(--"keep-cache" "Does not clear local cache directory upon completion.")
                        .required(false),
                ])
            )
        .subcommand(
            Command::new("ingest-api")
                .about("Enables/disables the ingest API of an index.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                    arg!(--enable "Enables the ingest API.")
                        .required(true)
                        .conflicts_with("disable")
                        .takes_value(false),
                    arg!(--disable "Disables the ingest API.")
                        .takes_value(false)
                        .required(false),
                ])
            )
        .subcommand(
            Command::new("describe")
                .about("Displays descriptive statistics of an index: number of published splits, number of documents, splits min/max timestamps, size of splits.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                ])
            )
        .subcommand(
            Command::new("search")
                .about("Searches an index.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                    arg!(--query <QUERY> "Query expressed in natural query language ((barack AND obama) OR \"president of united states\"). Learn more on https://quickwit.io/docs/reference/search-language."),
                    arg!(--aggregation <AGG> "JSON serialized aggregation request in tantivy/elasticsearch format.")
                        .required(false),
                    arg!(--"max-hits" <MAX_HITS> "Maximum number of hits returned.")
                        .default_value("20")
                        .required(false),
                    arg!(--"start-offset" <OFFSET> "Offset in the global result set of the first hit returned.")
                        .default_value("0")
                        .required(false),
                    arg!(--"search-fields" <FIELD_NAME> "List of fields that Quickwit will search into if the user query does not explicitly target a field in the query. It overrides the default search fields defined in the index config. Space-separated list, e.g. \"field1 field2\". ")
                        .multiple_values(true)
                        .required(false),
                    arg!(--"snippet-fields" <FIELD_NAME> "List of fields that Quickwit will return snippet highlight on. Space-separated list, e.g. \"field1 field2\". ")
                        .multiple_values(true)
                        .required(false),
                    arg!(--"start-timestamp" <TIMESTAMP> "Filters out documents before that timestamp (time-series indexes only).")
                        .required(false),
                    arg!(--"end-timestamp" <TIMESTAMP> "Filters out documents after that timestamp (time-series indexes only).")
                        .required(false),
                    arg!(--"sort-by-score" "Setting this flag calculates and sorts documents by their BM25 score.")
                        .required(false),
                ])
            )
        .subcommand(
            Command::new("merge")
                .about("Merges all the splits of the index pipeline defined by the tuple (index ID, source ID, pipeline ordinal). The pipeline ordinal is 0 by default. If you have a source with `num_pipelines > 0`, you may want to merge splits on ordinals > 0.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index.")
                        .display_order(1),
                    arg!(--source <SOURCE_ID> "ID of the target source."),
                ])
            )
        .subcommand(
            Command::new("gc")
                .about("Garbage collects stale staged splits and splits marked for deletion.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                    arg!(--"grace-period" <GRACE_PERIOD> "Threshold period after which stale staged splits are garbage collected.")
                        .default_value("1h")
                        .required(false),
                    arg!(--"dry-run" "Executes the command in dry run mode and only displays the list of splits candidates for garbage collection.")
                        .required(false),
                ])
            )
        .subcommand(
            Command::new("clear")
                .alias("clr")
                .about("Clears and index. Deletes all its splits and resets its checkpoint. This operation is destructive and cannot be undone, proceed with caution.")
                .args(&[
                    arg!(--index <INDEX> "Index ID")
                        .display_order(1),
                    arg!(--yes),
                ])
            )
        .subcommand(
            Command::new("delete")
            .alias("del")
                .about("Deletes an index. This operation is destructive and cannot be undone, proceed with caution.")
                .args(&[
                    arg!(--index <INDEX> "ID of the target index")
                        .display_order(1),
                    arg!(--"dry-run" "Executes the command in dry run mode and only displays the list of splits candidates for deletion.")
                        .required(false),
                ])
            )
        .arg_required_else_help(true)
}

#[derive(Debug, Eq, PartialEq)]
pub struct ClearIndexArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub yes: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct CreateIndexArgs {
    pub config_uri: Uri,
    pub index_config_uri: Uri,
    pub overwrite: bool,
    pub assume_yes: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct DescribeIndexArgs {
    pub config_uri: Uri,
    pub index_id: String,
}

#[derive(Debug, Eq, PartialEq)]
pub struct IngestDocsArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub input_path_opt: Option<PathBuf>,
    pub overwrite: bool,
    pub clear_cache: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ToggleIngestApiArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub enable: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct SearchIndexArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub query: String,
    pub aggregation: Option<String>,
    pub max_hits: usize,
    pub start_offset: usize,
    pub search_fields: Option<Vec<String>>,
    pub snippet_fields: Option<Vec<String>>,
    pub start_timestamp: Option<i64>,
    pub end_timestamp: Option<i64>,
    pub sort_by_score: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct DeleteIndexArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub dry_run: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct GarbageCollectIndexArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub grace_period: Duration,
    pub dry_run: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct MergeArgs {
    pub config_uri: Uri,
    pub index_id: String,
    pub source_id: String,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ListIndexesArgs {
    pub config_uri: Uri,
}

#[derive(Debug, Eq, PartialEq)]
pub enum IndexCliCommand {
    Clear(ClearIndexArgs),
    Create(CreateIndexArgs),
    Delete(DeleteIndexArgs),
    Describe(DescribeIndexArgs),
    GarbageCollect(GarbageCollectIndexArgs),
    Ingest(IngestDocsArgs),
    ToggleIngestApi(ToggleIngestApiArgs),
    List(ListIndexesArgs),
    Merge(MergeArgs),
    Search(SearchIndexArgs),
}

impl IndexCliCommand {
    pub fn default_log_level(&self) -> Level {
        match self {
            Self::Search(_) => Level::ERROR,
            _ => Level::INFO,
        }
    }

    pub fn parse_cli_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let (subcommand, submatches) = matches
            .subcommand()
            .ok_or_else(|| anyhow::anyhow!("Failed to parse sub-matches."))?;
        match subcommand {
            "clear" => Self::parse_clear_args(submatches),
            "create" => Self::parse_create_args(submatches),
            "delete" => Self::parse_delete_args(submatches),
            "describe" => Self::parse_describe_args(submatches),
            "gc" => Self::parse_garbage_collect_args(submatches),
            "ingest" => Self::parse_ingest_args(submatches),
            "ingest-api" => Self::parse_toggle_ingest_api_args(submatches),
            "list" => Self::parse_list_args(submatches),
            "merge" => Self::parse_merge_args(submatches),
            "search" => Self::parse_search_args(submatches),
            _ => bail!("Index subcommand `{}` is not implemented.", subcommand),
        }
    }

    fn parse_clear_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let yes = matches.is_present("yes");
        Ok(Self::Clear(ClearIndexArgs {
            config_uri,
            index_id,
            yes,
        }))
    }

    fn parse_create_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_config_uri = matches
            .value_of("index-config")
            .map(Uri::from_str)
            .expect("`index-config` is a required arg.")?;
        let overwrite = matches.is_present("overwrite");
        let assume_yes = matches.is_present("yes");

        Ok(Self::Create(CreateIndexArgs {
            config_uri,
            index_config_uri,
            overwrite,
            assume_yes,
        }))
    }

    fn parse_describe_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        Ok(Self::Describe(DescribeIndexArgs {
            config_uri,
            index_id,
        }))
    }

    fn parse_list_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        Ok(Self::List(ListIndexesArgs { config_uri }))
    }

    fn parse_ingest_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let input_path_opt = if let Some(input_path) = matches.value_of("input-path") {
            Uri::from_str(input_path)?
                .filepath()
                .map(|path| path.to_path_buf())
        } else {
            None
        };
        let overwrite = matches.is_present("overwrite");
        let clear_cache = !matches.is_present("keep-cache");

        Ok(Self::Ingest(IngestDocsArgs {
            index_id,
            input_path_opt,
            overwrite,
            config_uri,
            clear_cache,
        }))
    }

    fn parse_toggle_ingest_api_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let enable = matches.is_present("enable");
        Ok(Self::ToggleIngestApi(ToggleIngestApiArgs {
            config_uri,
            index_id,
            enable,
        }))
    }
    fn parse_search_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let query = matches
            .value_of("query")
            .context("`query` is a required arg.")?
            .to_string();
        let aggregation = matches.value_of("aggregation").map(|el| el.to_string());

        let max_hits = matches.value_of_t::<usize>("max-hits")?;
        let start_offset = matches.value_of_t::<usize>("start-offset")?;
        let search_fields = matches
            .values_of("search-fields")
            .map(|values| values.map(|value| value.to_string()).collect());
        let snippet_fields = matches
            .values_of("snippet-fields")
            .map(|values| values.map(|value| value.to_string()).collect());
        let sort_by_score = matches.is_present("sort-by-score");
        let start_timestamp = if matches.is_present("start-timestamp") {
            Some(matches.value_of_t::<i64>("start-timestamp")?)
        } else {
            None
        };
        let end_timestamp = if matches.is_present("end-timestamp") {
            Some(matches.value_of_t::<i64>("end-timestamp")?)
        } else {
            None
        };
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        Ok(Self::Search(SearchIndexArgs {
            index_id,
            query,
            aggregation,
            max_hits,
            start_offset,
            search_fields,
            snippet_fields,
            start_timestamp,
            end_timestamp,
            config_uri,
            sort_by_score,
        }))
    }

    fn parse_merge_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .context("'index-id' is a required arg.")?
            .to_string();
        let source_id = matches
            .value_of("source")
            .context("'source-id' is a required arg.")?
            .to_string();
        Ok(Self::Merge(MergeArgs {
            index_id,
            source_id,
            config_uri,
        }))
    }

    fn parse_garbage_collect_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let grace_period = matches
            .value_of("grace-period")
            .map(parse_duration_with_unit)
            .expect("`grace-period` should have a default value.")?;
        let dry_run = matches.is_present("dry-run");
        Ok(Self::GarbageCollect(GarbageCollectIndexArgs {
            index_id,
            grace_period,
            dry_run,
            config_uri,
        }))
    }

    fn parse_delete_args(matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_uri = matches
            .value_of("config")
            .map(Uri::from_str)
            .expect("`config` is a required arg.")?;
        let index_id = matches
            .value_of("index")
            .expect("`index` is a required arg.")
            .to_string();
        let dry_run = matches.is_present("dry-run");
        Ok(Self::Delete(DeleteIndexArgs {
            index_id,
            dry_run,
            config_uri,
        }))
    }

    pub async fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Clear(args) => clear_index_cli(args).await,
            Self::Create(args) => create_index_cli(args).await,
            Self::Delete(args) => delete_index_cli(args).await,
            Self::Describe(args) => describe_index_cli(args).await,
            Self::GarbageCollect(args) => garbage_collect_index_cli(args).await,
            Self::Ingest(args) => ingest_docs_cli(args).await,
            Self::ToggleIngestApi(args) => toggle_ingest_api_index_cli(args).await,
            Self::List(args) => list_index_cli(args).await,
            Self::Merge(args) => merge_cli(args).await,
            Self::Search(args) => search_index_cli(args).await,
        }
    }
}

pub async fn clear_index_cli(args: ClearIndexArgs) -> anyhow::Result<()> {
    debug!(args=?args, "clear-index");
    if !args.yes {
        let prompt = format!(
            "This operation will delete all the splits of the index `{}` and reset its \
             checkpoint. Do you want to proceed?",
            args.index_id
        );
        if !prompt_confirmation(&prompt, false) {
            return Ok(());
        }
    }
    let config = load_quickwit_config(&args.config_uri).await?;
    let index_service = IndexService::from_config(config).await?;
    index_service.clear_index(&args.index_id).await?;
    println!("Index `{}` successfully cleared.", args.index_id);
    Ok(())
}

pub async fn create_index_cli(args: CreateIndexArgs) -> anyhow::Result<()> {
    debug!(args=?args, "create-index");
    quickwit_telemetry::send_telemetry_event(TelemetryEvent::Create).await;

    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let file_content = load_file(&args.index_config_uri).await?;
    let index_config_format = ConfigFormat::sniff_from_uri(&args.index_config_uri)?;
    let index_config = quickwit_config::load_index_config_from_user_config(
        index_config_format,
        file_content.as_slice(),
        &quickwit_config.default_index_root_uri,
    )
    .with_context(|| format!("Failed to parse index_config `{}`", &args.index_config_uri))?;
    let index_id = index_config.index_id.clone();
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let metastore = metastore_uri_resolver
        .resolve(&quickwit_config.metastore_uri)
        .await?;

    validate_storage_uri(
        quickwit_storage_uri_resolver(),
        &quickwit_config,
        &index_config,
    )
    .await?;

    // On overwrite and index present and `assume_yes` if false, ask the user to confirm the
    // destructive operation.
    let index_exists = metastore.index_exists(&index_id).await?;
    if args.overwrite && index_exists && !args.assume_yes {
        // Stop if user answers no.
        let prompt = format!(
            "This operation will overwrite the index `{}` and delete all its data. Do you want to \
             proceed?",
            index_id
        );
        if !prompt_confirmation(&prompt, false) {
            return Ok(());
        }
    }

    let index_service = IndexService::new(metastore, quickwit_storage_uri_resolver().clone());
    index_service
        .create_index(index_config, args.overwrite)
        .await?;
    println!("Index `{}` successfully created.", index_id);
    Ok(())
}

pub async fn list_index_cli(args: ListIndexesArgs) -> anyhow::Result<()> {
    debug!(args=?args, "list");
    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let metastore = metastore_uri_resolver
        .resolve(&quickwit_config.metastore_uri)
        .await?;
    let indexes = metastore.list_indexes_metadatas().await?;
    let index_table =
        make_list_indexes_table(indexes.into_iter().map(IndexMetadata::into_index_config));

    println!("\n{}\n", index_table);
    Ok(())
}

fn make_list_indexes_table<I>(indexes: I) -> Table
where I: IntoIterator<Item = IndexConfig> {
    let rows = indexes
        .into_iter()
        .map(|index| IndexRow {
            index_id: index.index_id,
            index_uri: index.index_uri,
        })
        .sorted_by(|left, right| left.index_id.cmp(&right.index_id));
    make_table("Indexes", rows, false)
}

#[derive(Tabled)]
struct IndexRow {
    #[tabled(rename = "Index ID")]
    index_id: String,
    #[tabled(rename = "Index URI")]
    index_uri: Uri,
}

pub async fn describe_index_cli(args: DescribeIndexArgs) -> anyhow::Result<()> {
    debug!(args=?args, "describe");
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let metastore = metastore_uri_resolver
        .resolve(&quickwit_config.metastore_uri)
        .await?;

    let query = ListSplitsQuery::for_index(&args.index_id).with_split_state(SplitState::Published);
    let splits = metastore.list_splits(query).await?;
    let index_metadata = metastore.index_metadata(&args.index_id).await?;
    let index_stats = IndexStats::from_metadata(index_metadata, splits)?;
    println!("{}", index_stats.display_as_table());
    Ok(())
}

pub struct IndexStats {
    pub index_id: String,
    pub index_uri: Uri,
    pub num_published_splits: usize,
    pub num_published_docs: usize,
    pub size_published_docs: usize,
    pub timestamp_field_name: Option<String>,
    pub timestamp_range: Option<(i64, i64)>,
    pub num_docs_descriptive: Option<DescriptiveStats>,
    pub num_bytes_descriptive: Option<DescriptiveStats>,
}

impl Tabled for IndexStats {
    const LENGTH: usize = 7;

    fn fields(&self) -> Vec<String> {
        vec![
            self.index_id.clone(),
            self.index_uri.to_string(),
            self.num_published_splits.to_string(),
            self.num_published_docs.to_string(),
            format!("{} MB", self.size_published_docs),
            display_option_in_table(&self.timestamp_field_name),
            display_timestamp_range(&self.timestamp_range),
        ]
    }

    fn headers() -> Vec<String> {
        vec![
            "Index ID: ".to_string(),
            "Index URI: ".to_string(),
            "Number of published splits: ".to_string(),
            "Number of published documents: ".to_string(),
            "Size of published splits: ".to_string(),
            "Timestamp field: ".to_string(),
            "Timestamp range: ".to_string(),
        ]
    }
}

fn display_option_in_table(opt: &Option<impl Display>) -> String {
    match opt {
        Some(opt_val) => format!("{}", opt_val),
        None => "Field does not exist for the index.".to_string(),
    }
}

fn display_timestamp_range(range: &Option<(i64, i64)>) -> String {
    match range {
        Some((timestamp_min, timestamp_max)) => {
            format!("{} -> {}", timestamp_min, timestamp_max)
        }
        _ => "Range does not exist for the index.".to_string(),
    }
}

impl IndexStats {
    pub fn from_metadata(
        index_metadata: IndexMetadata,
        splits: Vec<Split>,
    ) -> anyhow::Result<Self> {
        let splits_num_docs = splits
            .iter()
            .map(|split| split.split_metadata.num_docs)
            .sorted()
            .collect_vec();

        let total_num_docs = splits_num_docs.iter().sum::<usize>();

        let splits_bytes = splits
            .iter()
            .map(|split| (split.split_metadata.footer_offsets.end / 1_000_000) as usize)
            .sorted()
            .collect_vec();
        let total_bytes = splits_bytes.iter().sum::<usize>();

        let timestamp_range = if index_metadata
            .index_config()
            .indexing_settings
            .timestamp_field
            .is_some()
        {
            let time_min = splits
                .iter()
                .flat_map(|split| split.split_metadata.time_range.clone())
                .map(|time_range| *time_range.start())
                .min();
            let time_max = splits
                .iter()
                .flat_map(|split| split.split_metadata.time_range.clone())
                .map(|time_range| *time_range.end())
                .max();
            if let (Some(time_min), Some(time_max)) = (time_min, time_max) {
                Some((time_min, time_max))
            } else {
                None
            }
        } else {
            None
        };

        let (num_docs_descriptive, num_bytes_descriptive) = if !splits.is_empty() {
            (
                DescriptiveStats::maybe_new(&splits_num_docs),
                DescriptiveStats::maybe_new(&splits_bytes),
            )
        } else {
            (None, None)
        };
        let index_config = index_metadata.into_index_config();

        Ok(Self {
            index_id: index_config.index_id.clone(),
            index_uri: index_config.index_uri.clone(),
            num_published_splits: splits.len(),
            num_published_docs: total_num_docs,
            size_published_docs: total_bytes,
            timestamp_field_name: index_config.indexing_settings.timestamp_field,
            timestamp_range,
            num_docs_descriptive,
            num_bytes_descriptive,
        })
    }

    pub fn display_as_table(&self) -> String {
        let index_stats_table = create_table(self, "General Information");

        let index_stats_table = if let Some(docs_stats) = &self.num_docs_descriptive {
            let doc_stats_table = create_table(docs_stats, "Document count stats");
            index_stats_table.with(Concat::vertical(doc_stats_table))
        } else {
            index_stats_table
        };

        let index_stats_table = if let Some(size_stats) = &self.num_bytes_descriptive {
            let size_stats_table = create_table(size_stats, "Size in MB stats");
            index_stats_table.with(Concat::vertical(size_stats_table))
        } else {
            index_stats_table
        };

        index_stats_table.to_string()
    }
}

fn create_table(table: impl Tabled, header: &str) -> Table {
    Table::new(vec![table])
        .with(Rotate::Left)
        .with(Rotate::Bottom)
        .with(
            Modify::new(Columns::first())
                .with(Format::new(|column| column.color(GREEN_COLOR).to_string())),
        )
        .with(
            Modify::new(Segment::all())
                .with(Alignment::left())
                .with(Alignment::top()),
        )
        .with(Panel(header, 0))
        .with(Style::psql())
        .with(Panel("\n", 0))
}

#[derive(Debug)]
pub struct DescriptiveStats {
    mean_val: f32,
    std_val: f32,
    min_val: usize,
    max_val: usize,
    q1: f32,
    q25: f32,
    q50: f32,
    q75: f32,
    q99: f32,
}

impl Tabled for DescriptiveStats {
    const LENGTH: usize = 2;

    fn fields(&self) -> Vec<String> {
        vec![
            format!(
                "{} ± {} in [{} … {}]",
                self.mean_val, self.std_val, self.min_val, self.max_val
            ),
            format!(
                "[{}, {}, {}, {}, {}]",
                self.q1, self.q25, self.q50, self.q75, self.q99,
            ),
        ]
    }

    fn headers() -> Vec<String> {
        vec![
            "Mean ± σ in [min … max]:".to_string(),
            "Quantiles [1%, 25%, 50%, 75%, 99%]:".to_string(),
        ]
    }
}

impl DescriptiveStats {
    pub fn maybe_new(values: &[usize]) -> Option<DescriptiveStats> {
        if values.is_empty() {
            return None;
        }
        Some(DescriptiveStats {
            mean_val: mean(values),
            std_val: std_deviation(values),
            min_val: *values.iter().min().expect("Values should not be empty."),
            max_val: *values.iter().max().expect("Values should not be empty."),
            q1: percentile(values, 1),
            q25: percentile(values, 50),
            q50: percentile(values, 50),
            q75: percentile(values, 75),
            q99: percentile(values, 75),
        })
    }
}

pub async fn ingest_docs_cli(args: IngestDocsArgs) -> anyhow::Result<()> {
    debug!(args=?args, "ingest-docs");
    quickwit_telemetry::send_telemetry_event(TelemetryEvent::Ingest).await;

    let config = load_quickwit_config(&args.config_uri).await?;

    let source_params = if let Some(filepath) = args.input_path_opt.as_ref() {
        SourceParams::file(filepath)
    } else {
        SourceParams::stdin()
    };
    let source_config = SourceConfig {
        source_id: CLI_INGEST_SOURCE_ID.to_string(),
        num_pipelines: 1,
        enabled: true,
        source_params,
    };
    run_index_checklist(&config.metastore_uri, &args.index_id, Some(&source_config)).await?;
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let metastore = metastore_uri_resolver
        .resolve(&config.metastore_uri)
        .await?;

    if args.overwrite {
        let index_service =
            IndexService::new(metastore.clone(), quickwit_storage_uri_resolver().clone());
        index_service.clear_index(&args.index_id).await?;
    }
    let indexer_config = IndexerConfig {
        ..Default::default()
    };
    start_actor_runtimes(&HashSet::from_iter([QuickwitService::Indexer]))?;
    let universe = Universe::new();
    let indexing_server = IndexingService::new(
        config.node_id.clone(),
        config.data_dir_path.clone(),
        indexer_config,
        metastore,
        quickwit_storage_uri_resolver().clone(),
    )
    .await?;
    let (indexing_server_mailbox, indexing_server_handle) =
        universe.spawn_builder().spawn(indexing_server);
    let pipeline_id = indexing_server_mailbox
        .ask_for_res(SpawnPipeline {
            index_id: args.index_id.clone(),
            source_config,
            pipeline_ord: 0,
        })
        .await?;
    let pipeline_handle = indexing_server_mailbox
        .ask_for_res(DetachPipeline { pipeline_id })
        .await?;

    let is_stdin_atty = atty::is(atty::Stream::Stdin);
    if args.input_path_opt.is_none() && is_stdin_atty {
        let eof_shortcut = match env::consts::OS {
            "windows" => "CTRL+Z",
            _ => "CTRL+D",
        };
        println!(
            "Please, enter JSON documents one line at a time.\nEnd your input using {}.",
            eof_shortcut
        );
    }
    let statistics =
        start_statistics_reporting_loop(pipeline_handle, args.input_path_opt.is_none()).await?;
    // Shutdown the indexing server.
    universe
        .send_exit_with_success(&indexing_server_mailbox)
        .await?;
    indexing_server_handle.join().await;
    if statistics.num_published_splits > 0 {
        println!(
            "Now, you can query the index with the following command:\nquickwit index search \
             --index {} --config ./config/quickwit.yaml --query \"my query\"",
            args.index_id
        );
    }

    if args.clear_cache {
        println!("Clearing local cache directory...");
        clear_cache_directory(&config.data_dir_path).await?;
    }

    match statistics.num_invalid_docs {
        0 => Ok(()),
        _ => bail!("Failed to ingest all the documents."),
    }
}

pub async fn toggle_ingest_api_index_cli(args: ToggleIngestApiArgs) -> anyhow::Result<()> {
    let config = load_quickwit_config(&args.config_uri).await?;
    let metastore = quickwit_metastore_uri_resolver()
        .resolve(&config.metastore_uri)
        .await?;
    metastore
        .toggle_source(&args.index_id, INGEST_API_SOURCE_ID, args.enable)
        .await?;
    let toggled_state_name = if args.enable { "enabled" } else { "disabled" };
    println!(
        "Ingest API successfully {} for index `{}`.",
        toggled_state_name, args.index_id
    );
    Ok(())
}

pub async fn search_index(args: SearchIndexArgs) -> anyhow::Result<SearchResponse> {
    debug!(args=?args, "search-index");
    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let storage_uri_resolver = quickwit_storage_uri_resolver();
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let metastore = metastore_uri_resolver
        .resolve(&quickwit_config.metastore_uri)
        .await?;
    let search_request = SearchRequest {
        index_id: args.index_id,
        query: args.query.clone(),
        search_fields: args.search_fields.unwrap_or_default(),
        snippet_fields: args.snippet_fields.unwrap_or_default(),
        start_timestamp: args.start_timestamp,
        end_timestamp: args.end_timestamp,
        max_hits: args.max_hits as u64,
        start_offset: args.start_offset as u64,
        sort_order: None,
        sort_by_field: args.sort_by_score.then_some("_score".to_string()),
        aggregation_request: args.aggregation,
    };
    let search_response: SearchResponse =
        single_node_search(&search_request, &*metastore, storage_uri_resolver.clone()).await?;
    Ok(search_response)
}

pub async fn search_index_cli(args: SearchIndexArgs) -> anyhow::Result<()> {
    let search_response: SearchResponse = search_index(args).await?;
    let search_response_rest = SearchResponseRest::try_from(search_response)?;
    let search_response_json = serde_json::to_string_pretty(&search_response_rest)?;
    println!("{}", search_response_json);
    Ok(())
}

pub async fn merge_cli(args: MergeArgs) -> anyhow::Result<()> {
    debug!(args=?args, "run-merge-operations");
    let config = load_quickwit_config(&args.config_uri).await?;
    run_index_checklist(&config.metastore_uri, &args.index_id, None).await?;
    let indexer_config = IndexerConfig {
        ..Default::default()
    };
    let metastore_uri_resolver = quickwit_metastore_uri_resolver();
    let metastore = metastore_uri_resolver
        .resolve(&config.metastore_uri)
        .await?;
    let storage_resolver = quickwit_storage_uri_resolver().clone();
    start_actor_runtimes(&HashSet::from_iter([QuickwitService::Indexer]))?;
    let node_id = config.node_id.clone();
    let indexing_server = IndexingService::new(
        config.node_id,
        config.data_dir_path,
        indexer_config,
        metastore,
        storage_resolver,
    )
    .await?;
    let universe = Universe::new();
    let (indexing_service_mailbox, indexing_service_handle) =
        universe.spawn_builder().spawn(indexing_server);
    let pipeline_id = IndexingPipelineId {
        index_id: args.index_id,
        source_id: args.source_id,
        node_id,
        pipeline_ord: 0,
    };
    indexing_service_mailbox
        .ask_for_res(SpawnMergePipeline {
            pipeline_id: pipeline_id.clone(),
        })
        .await?;
    let pipeline_handle = indexing_service_mailbox
        .ask_for_res(DetachPipeline { pipeline_id })
        .await?;
    let (pipeline_exit_status, _pipeline_statistics) = pipeline_handle.join().await;
    indexing_service_handle.quit().await;
    if !pipeline_exit_status.is_success() {
        bail!(pipeline_exit_status);
    }
    Ok(())
}

pub async fn delete_index_cli(args: DeleteIndexArgs) -> anyhow::Result<()> {
    debug!(args=?args, "delete-index");
    quickwit_telemetry::send_telemetry_event(TelemetryEvent::Delete).await;

    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let metastore = quickwit_metastore_uri_resolver()
        .resolve(&quickwit_config.metastore_uri)
        .await?;
    let index_service = IndexService::new(metastore, quickwit_storage_uri_resolver().clone());
    let affected_files = index_service
        .delete_index(&args.index_id, args.dry_run)
        .await?;
    if args.dry_run {
        if affected_files.is_empty() {
            println!("Only the index will be deleted since it does not contains any data file.");
            return Ok(());
        }
        println!(
            "The following files will be removed from the index `{}`",
            args.index_id
        );
        for file_entry in affected_files {
            println!(" - {}", file_entry.file_name);
        }
        return Ok(());
    }
    if let Err(error) =
        remove_indexing_directory(&quickwit_config.data_dir_path, args.index_id.clone()).await
    {
        warn!(error= ?error, "Failed to remove indexing directory.");
    }
    println!("Index `{}` successfully deleted.", args.index_id);
    Ok(())
}

pub async fn garbage_collect_index_cli(args: GarbageCollectIndexArgs) -> anyhow::Result<()> {
    debug!(args=?args, "garbage-collect-index");
    quickwit_telemetry::send_telemetry_event(TelemetryEvent::GarbageCollect).await;

    let quickwit_config = load_quickwit_config(&args.config_uri).await?;
    let metastore = quickwit_metastore_uri_resolver()
        .resolve(&quickwit_config.metastore_uri)
        .await?;
    let index_service = IndexService::new(metastore, quickwit_storage_uri_resolver().clone());
    let removal_info = index_service
        .garbage_collect_index(&args.index_id, args.grace_period, args.dry_run)
        .await?;
    if removal_info.removed_split_entries.is_empty() && removal_info.failed_split_ids.is_empty() {
        println!("No dangling files to garbage collect.");
        return Ok(());
    }

    if args.dry_run {
        println!("The following files will be garbage collected.");
        for file_entry in removal_info.removed_split_entries {
            println!(" - {}", file_entry.file_name);
        }
        return Ok(());
    }

    if !removal_info.failed_split_ids.is_empty() {
        println!("The following splits were attempted to be removed, but failed.");
        for split_id in removal_info.failed_split_ids.iter() {
            println!(" - {}", split_id);
        }
        println!(
            "{} Splits were unable to be removed.",
            removal_info.failed_split_ids.len()
        );
    }

    let deleted_bytes: u64 = removal_info
        .removed_split_entries
        .iter()
        .map(|entry| entry.file_size_in_bytes)
        .sum();
    println!(
        "{}MB of storage garbage collected.",
        deleted_bytes / 1_000_000
    );

    if removal_info.failed_split_ids.is_empty() {
        println!("Index `{}` successfully garbage collected.", args.index_id);
    } else if removal_info.removed_split_entries.is_empty()
        && !removal_info.failed_split_ids.is_empty()
    {
        println!("Failed to garbage collect index `{}`.", args.index_id);
    } else {
        println!("Index `{}` partially garbage collected.", args.index_id);
    }

    Ok(())
}

/// Starts a tokio task that displays the indexing statistics
/// every once in awhile.
pub async fn start_statistics_reporting_loop(
    pipeline_handle: ActorHandle<IndexingPipeline>,
    is_stdin: bool,
) -> anyhow::Result<IndexingStatistics> {
    let mut stdout_handle = stdout();
    let start_time = Instant::now();
    let mut throughput_calculator = ThroughputCalculator::new(start_time);
    let mut report_interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        // TODO fixme. The way we wait today is a bit lame: if the indexing pipeline exits, we will
        // stil wait up to an entire heartbeat...  Ideally we should  select between two
        // futures.
        report_interval.tick().await;
        // Try to receive with a timeout of 1 second.
        // 1 second is also the frequency at which we update statistic in the console
        let observation = pipeline_handle.observe().await;

        // Let's not display live statistics to allow screen to scroll.
        if observation.state.num_docs > 0 {
            display_statistics(
                &mut stdout_handle,
                &mut throughput_calculator,
                &observation.state,
            )?;
        }

        if observation.obs_type == ObservationType::PostMortem {
            break;
        }
    }
    let (pipeline_exit_status, pipeline_statistics) = pipeline_handle.join().await;
    if !pipeline_exit_status.is_success() {
        bail!(pipeline_exit_status);
    }
    // If we have received zero docs at this point,
    // there is no point in displaying report.
    if pipeline_statistics.num_docs == 0 {
        return Ok(pipeline_statistics);
    }

    if is_stdin {
        display_statistics(
            &mut stdout_handle,
            &mut throughput_calculator,
            &pipeline_statistics,
        )?;
    }
    // display end of task report
    println!();
    let secs = Duration::from_secs(start_time.elapsed().as_secs());
    if pipeline_statistics.num_invalid_docs == 0 {
        println!(
            "Indexed {} documents in {}.",
            pipeline_statistics.num_docs.separate_with_commas(),
            format_duration(secs)
        );
    } else {
        let num_indexed_docs = (pipeline_statistics.num_docs
            - pipeline_statistics.num_invalid_docs)
            .separate_with_commas();

        let error_rate = (pipeline_statistics.num_invalid_docs as f64
            / pipeline_statistics.num_docs as f64)
            * 100.0;

        println!(
            "Indexed {} out of {} documents in {}. Failed to index {} document(s). {}\n",
            num_indexed_docs,
            pipeline_statistics.num_docs.separate_with_commas(),
            format_duration(secs),
            pipeline_statistics.num_invalid_docs.separate_with_commas(),
            colorize_error_rate(error_rate),
        );
    }

    Ok(pipeline_statistics)
}

fn colorize_error_rate(error_rate: f64) -> ColoredString {
    let error_rate_message = format!("({:.1}% error rate)", error_rate);
    if error_rate < 1.0 {
        error_rate_message.yellow()
    } else if error_rate < 5.0 {
        error_rate_message.truecolor(255, 181, 46) //< Orange
    } else {
        error_rate_message.red()
    }
}

/// A struct to print data on the standard output.
struct Printer<'a> {
    pub stdout: &'a mut Stdout,
}

impl<'a> Printer<'a> {
    pub fn print_header(&mut self, header: &str) -> io::Result<()> {
        write!(&mut self.stdout, " {}", header.bright_blue())?;
        Ok(())
    }

    pub fn print_value(&mut self, fmt_args: fmt::Arguments) -> io::Result<()> {
        write!(&mut self.stdout, " {}", fmt_args)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.stdout.flush()
    }
}

fn display_statistics(
    stdout: &mut Stdout,
    throughput_calculator: &mut ThroughputCalculator,
    statistics: &IndexingStatistics,
) -> anyhow::Result<()> {
    let elapsed_duration = time::Duration::try_from(throughput_calculator.elapsed_time())?;
    let elapsed_time = format!(
        "{:02}:{:02}:{:02}",
        elapsed_duration.whole_hours(),
        elapsed_duration.whole_minutes() % 60,
        elapsed_duration.whole_seconds() % 60
    );
    let throughput_mb_s = throughput_calculator.calculate(statistics.total_bytes_processed);
    let mut printer = Printer { stdout };
    printer.print_header("Num docs")?;
    printer.print_value(format_args!("{:>7}", statistics.num_docs))?;
    printer.print_header("Parse errs")?;
    printer.print_value(format_args!("{:>5}", statistics.num_invalid_docs))?;
    printer.print_header("PublSplits")?;
    printer.print_value(format_args!("{:>3}", statistics.num_published_splits))?;
    printer.print_header("Input size")?;
    printer.print_value(format_args!(
        "{:>5}MB",
        statistics.total_bytes_processed / 1_000_000
    ))?;
    printer.print_header("Thrghput")?;
    printer.print_value(format_args!("{:>5.2}MB/s", throughput_mb_s))?;
    printer.print_header("Time")?;
    printer.print_value(format_args!("{}\n", elapsed_time))?;
    printer.flush()?;
    Ok(())
}

/// ThroughputCalculator is used to calculate throughput.
struct ThroughputCalculator {
    /// Stores the time series of processed bytes value.
    processed_bytes_values: VecDeque<(Instant, u64)>,
    /// Store the time this calculator started
    start_time: Instant,
}

impl ThroughputCalculator {
    /// Creates new instance.
    pub fn new(start_time: Instant) -> Self {
        let processed_bytes_values: VecDeque<(Instant, u64)> = (0..THROUGHPUT_WINDOW_SIZE)
            .map(|_| (start_time, 0u64))
            .collect();
        Self {
            processed_bytes_values,
            start_time,
        }
    }

    /// Calculates the throughput.
    pub fn calculate(&mut self, current_processed_bytes: u64) -> f64 {
        self.processed_bytes_values.pop_front();
        let current_instant = Instant::now();
        let (first_instant, first_processed_bytes) = *self.processed_bytes_values.front().unwrap();
        let elapsed_time = (current_instant - first_instant).as_millis() as f64 / 1_000f64;
        self.processed_bytes_values
            .push_back((current_instant, current_processed_bytes));
        (current_processed_bytes - first_processed_bytes) as f64
            / 1_000_000f64
            / elapsed_time.max(1f64) as f64
    }

    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod test {

    use std::ops::RangeInclusive;
    use std::str::FromStr;

    use quickwit_metastore::SplitMetadata;

    use super::*;
    use crate::cli::{build_cli, CliCommand};

    pub fn split_metadata_for_test(
        split_id: &str,
        num_docs: usize,
        time_range: RangeInclusive<i64>,
        size: u64,
    ) -> SplitMetadata {
        let mut split_metadata = SplitMetadata::for_test(split_id.to_string());
        split_metadata.num_docs = num_docs;
        split_metadata.time_range = Some(time_range);
        split_metadata.footer_offsets = 0..size;
        split_metadata
    }

    #[test]
    fn test_parse_ingest_api_args() -> anyhow::Result<()> {
        {
            let app = build_cli().no_binary_name(true);
            let matches = app.try_get_matches_from(vec![
                "index",
                "ingest-api",
                "--config",
                "/config.yaml",
                "--index",
                "foo",
                "--enable",
            ])?;
            let command = CliCommand::parse_cli_args(&matches)?;
            let expected_command =
                CliCommand::Index(IndexCliCommand::ToggleIngestApi(ToggleIngestApiArgs {
                    config_uri: Uri::from_str("file:///config.yaml").unwrap(),
                    index_id: "foo".to_string(),
                    enable: true,
                }));
            assert_eq!(command, expected_command);
        }
        {
            let app = build_cli().no_binary_name(true);
            let matches = app.try_get_matches_from(vec![
                "index",
                "ingest-api",
                "--config",
                "/config.yaml",
                "--index",
                "foo",
                "--disable",
            ])?;
            let command = CliCommand::parse_cli_args(&matches)?;
            let expected_command =
                CliCommand::Index(IndexCliCommand::ToggleIngestApi(ToggleIngestApiArgs {
                    config_uri: Uri::from_str("file:///config.yaml").unwrap(),
                    index_id: "foo".to_string(),
                    enable: false,
                }));
            assert_eq!(command, expected_command);
        }
        {
            let app = build_cli().no_binary_name(true);
            let matches = app.try_get_matches_from(vec![
                "index",
                "ingest-api",
                "--config",
                "/config.yaml",
                "--index",
                "foo",
                "--enable",
                "--disable",
            ]);
            assert!(matches.is_err());
        }
        {
            let app = build_cli().no_binary_name(true);
            let matches = app.try_get_matches_from(vec![
                "index",
                "ingest-api",
                "--config",
                "/config.yaml",
                "--index",
                "foo",
            ]);
            assert!(matches.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_index_stats() -> anyhow::Result<()> {
        let index_id = "index-stats-env".to_string();
        let split_id = "test_split_id".to_string();
        let index_uri = "s3://some-test-bucket";

        let index_metadata = IndexMetadata::for_test(&index_id, index_uri);
        let split_metadata = split_metadata_for_test(&split_id, 100_000, 1111..=2222, 15_000_000);

        let split_data = Split {
            split_metadata,
            split_state: quickwit_metastore::SplitState::Published,
            update_timestamp: 0,
            publish_timestamp: Some(0),
        };

        let index_stats = IndexStats::from_metadata(index_metadata, vec![split_data])?;

        assert_eq!(index_stats.index_id, index_id);
        assert_eq!(index_stats.index_uri.as_str(), index_uri);
        assert_eq!(index_stats.num_published_splits, 1);
        assert_eq!(index_stats.num_published_docs, 100_000);
        assert_eq!(index_stats.size_published_docs, 15);
        assert_eq!(
            index_stats.timestamp_field_name,
            Some("timestamp".to_string())
        );
        assert_eq!(index_stats.timestamp_range, Some((1111, 2222)));

        Ok(())
    }

    #[test]
    fn test_descriptive_stats() -> anyhow::Result<()> {
        let split_id = "stat-test-split".to_string();
        let template_split = Split {
            split_state: quickwit_metastore::SplitState::Published,
            update_timestamp: 0,
            publish_timestamp: Some(0),
            split_metadata: SplitMetadata::default(),
        };

        let split_metadata_1 = split_metadata_for_test(&split_id, 70_000, 10..=12, 60_000_000);
        let split_metadata_2 = split_metadata_for_test(&split_id, 120_000, 11..=15, 145_000_000);
        let split_metadata_3 = split_metadata_for_test(&split_id, 90_000, 15..=22, 115_000_000);
        let split_metadata_4 = split_metadata_for_test(&split_id, 40_000, 22..=22, 55_000_000);

        let mut split_1 = template_split.clone();
        split_1.split_metadata = split_metadata_1;
        let mut split_2 = template_split.clone();
        split_2.split_metadata = split_metadata_2;
        let mut split_3 = template_split.clone();
        split_3.split_metadata = split_metadata_3;
        let mut split_4 = template_split;
        split_4.split_metadata = split_metadata_4;

        let splits = vec![split_1, split_2, split_3, split_4];

        let splits_num_docs = splits
            .iter()
            .map(|split| split.split_metadata.num_docs)
            .sorted()
            .collect_vec();

        let splits_bytes = splits
            .iter()
            .map(|split| (split.split_metadata.footer_offsets.end / 1_000_000) as usize)
            .sorted()
            .collect_vec();

        let num_docs_descriptive = DescriptiveStats::maybe_new(&splits_num_docs);
        let num_bytes_descriptive = DescriptiveStats::maybe_new(&splits_bytes);
        let desciptive_stats_none = DescriptiveStats::maybe_new(&[]);

        assert!(num_docs_descriptive.is_some());
        assert!(num_bytes_descriptive.is_some());

        assert!(desciptive_stats_none.is_none());

        Ok(())
    }
}

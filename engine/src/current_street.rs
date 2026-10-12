//! Compact, self-contained policy for the solve's starting street only.
//! Later-street chance branches are explicit boundaries, never silently omitted.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::br::ExploitReport;
use crate::cards;
use crate::cfr::Solver;
use crate::config::{SolveConfig, Street};
use crate::game::Game;
use crate::nlhe::NlheGame;
use crate::solution::{ActionLabelData, RootCombo};
use crate::tree::{ActionLabel, NodeKind, Terminal};

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyMeta {
    pub iterations: u64,
    pub metric: String,
    pub metric_value: f32,
    pub metric_pct_of_pot: f32,
    pub wall_seconds: f64,
    pub engine_version: String,
    pub payoff_unit: String,
    /// Allocated CFR regret and strategy arrays, measured at export time.
    #[serde(default)]
    pub storage_bytes: usize,
    /// Precomputed chance maps, measured at export time.
    #[serde(default)]
    pub chance_map_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Child {
    Decision { node: u32 },
    StreetBoundary { node: u32, street: Street },
    HandTerminal { node: u32, kind: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAction {
    /// Stable within this config and tree: the same token accepted by resolve_line.
    pub action_id: String,
    pub label: ActionLabelData,
    /// Total chips put in by the actor on this street after the action. Includes calls
    /// and all-ins, whose labels themselves do not carry amounts.
    pub amount_to: f64,
    /// Additional chips committed by this action (zero for folds and checks).
    pub amount_put: f64,
    /// Public state after the action, including terminal/boundary edges.
    pub result_pot: f64,
    pub result_stacks: [f64; 2],
    pub child: Child,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyNode {
    pub node: u32,
    pub player: u8,
    pub pot: f64,
    pub stacks: [f64; 2],
    pub actions: Vec<PolicyAction>,
    /// f32 average probabilities, action-major, aligned with combos[player].
    pub strategy: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurrentStreetPolicy {
    pub format_version: u32,
    pub config: SolveConfig,
    pub meta: PolicyMeta,
    pub street: Street,
    pub root_node: u32,
    pub full_tree_node_count: u32,
    /// Exact canonical combo order for each actor. It is constant on this street.
    pub combos: [Vec<RootCombo>; 2],
    pub nodes: Vec<PolicyNode>,
}

impl CurrentStreetPolicy {
    pub fn from_solver(solver: &Solver<NlheGame>, wall_seconds: f64) -> Result<Self, String> {
        Self::from_solver_with_report(solver, wall_seconds, solver.exploitability())
    }

    /// Use an already measured report to avoid another expensive full-tree best
    /// response traversal when the CLI has just printed its final report.
    pub fn from_solver_with_report(
        solver: &Solver<NlheGame>,
        wall_seconds: f64,
        report: ExploitReport,
    ) -> Result<Self, String> {
        Self::from_profile_with_report(
            solver.game(),
            &solver.average(),
            solver.iterations(),
            solver.storage_bytes(),
            wall_seconds,
            report,
        )
    }

    /// Export public street geometry and rows from a complete supplied profile.
    /// This avoids allocating CFR arrays when evaluating a saved continuation.
    pub fn from_profile_with_report<P: crate::br::StrategyProfile>(
        game: &NlheGame,
        profile: &P,
        iterations: u64,
        storage_bytes: usize,
        wall_seconds: f64,
        report: ExploitReport,
    ) -> Result<Self, String> {
        let tree = game.tree();
        let config = game.config().clone();
        let street = config.starting_street()?;
        let root_node = game.root();
        let combos = std::array::from_fn(|p| {
            game.live_combos(root_node, p as u8)
                .iter()
                .zip(game.combo_indices(root_node, p as u8))
                .map(|(&(a, b), &index)| RootCombo {
                    index,
                    cards: format!("{}{}", cards::card_to_string(a), cards::card_to_string(b)),
                })
                .collect::<Vec<_>>()
        });
        let mut nodes = Vec::new();
        for idx in 0..tree.len() as u32 {
            let node = tree.node(idx);
            if node.street != street {
                continue;
            }
            let NodeKind::Decision { player, actions } = &node.kind else {
                continue;
            };
            // A current-street action cannot alter the board. Fail if this invariant
            // ever changes rather than silently mislabeling the probability vector.
            if game.combo_indices(idx, *player) != game.combo_indices(root_node, *player) {
                return Err(format!(
                    "combo ordering varies at current-street node {idx}"
                ));
            }
            let policy_actions = actions
                .iter()
                .map(|action| {
                    let child_node = tree.node(action.child);
                    let child = match &child_node.kind {
                        NodeKind::Decision { .. } if child_node.street == street => {
                            Child::Decision { node: action.child }
                        }
                        NodeKind::Chance { street: next, .. } => Child::StreetBoundary {
                            node: action.child,
                            street: *next,
                        },
                        NodeKind::Terminal(Terminal::Fold { .. }) => Child::HandTerminal {
                            node: action.child,
                            kind: "fold".into(),
                        },
                        NodeKind::Terminal(Terminal::Showdown { .. }) => Child::HandTerminal {
                            node: action.child,
                            kind: "showdown".into(),
                        },
                        _ => return Err(format!("unexpected child at node {}", action.child)),
                    };
                    let amount_to = match action.label {
                        ActionLabel::Fold | ActionLabel::Check => 0.0,
                        ActionLabel::Bet(to) | ActionLabel::Raise(to) => to,
                        ActionLabel::Call | ActionLabel::AllIn => {
                            config.effective_stack - child_node.stacks[*player as usize]
                        }
                    };
                    let amount_put =
                        node.stacks[*player as usize] - child_node.stacks[*player as usize];
                    Ok(PolicyAction {
                        action_id: action.label.token(),
                        label: action.label.into(),
                        amount_to,
                        amount_put,
                        result_pot: child_node.pot,
                        result_stacks: child_node.stacks,
                        child,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut strategy = vec![0.0; policy_actions.len() * combos[*player as usize].len()];
            profile.strategy_into(idx, &mut strategy);
            if strategy.len() != policy_actions.len() * combos[*player as usize].len() {
                return Err(format!("strategy length mismatch at node {idx}"));
            }
            nodes.push(PolicyNode {
                node: idx,
                player: *player,
                pot: node.pot,
                stacks: node.stacks,
                actions: policy_actions,
                strategy,
            });
        }
        Ok(Self {
            format_version: FORMAT_VERSION,
            config,
            meta: PolicyMeta {
                iterations,
                metric: if game.config().tournament.is_some() {
                    "nash_conv"
                } else {
                    "exploitability"
                }
                .into(),
                metric_value: report.chips,
                metric_pct_of_pot: report.pct_of_pot,
                wall_seconds,
                engine_version: env!("CARGO_PKG_VERSION").into(),
                payoff_unit: if game.config().tournament.is_some() {
                    "cste"
                } else {
                    "chips"
                }
                .into(),
                storage_bytes,
                chance_map_bytes: game.chance_map_bytes(),
            },
            street,
            root_node,
            full_tree_node_count: tree.len() as u32,
            combos,
            nodes,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|e| format!("cannot create {path:?}: {e}"))?;
        self.to_writer(BufWriter::new(file))
    }

    /// Write beside the destination, sync the file, then publish by rename. A
    /// crash during serialization leaves the previous complete policy intact.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("clock error: {e}"))?
            .as_nanos();
        let name = path
            .file_name()
            .ok_or_else(|| format!("policy path has no filename: {path:?}"))?
            .to_string_lossy();
        let tmp = path.with_file_name(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
        let result = (|| {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| format!("cannot create {tmp:?}: {e}"))?;
            let mut writer = BufWriter::new(file);
            self.to_writer(&mut writer)?;
            writer
                .flush()
                .map_err(|e| format!("cannot flush {tmp:?}: {e}"))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| format!("cannot sync {tmp:?}: {e}"))?;
            std::fs::rename(&tmp, path).map_err(|e| format!("cannot publish {path:?}: {e}"))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    pub fn to_writer<W: Write>(&self, writer: W) -> Result<(), String> {
        serde_json::to_writer(writer, self)
            .map_err(|e| format!("cannot write current-street policy: {e}"))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("cannot open {path:?}: {e}"))?;
        Self::from_reader(BufReader::new(file))
    }

    pub fn from_reader<R: Read>(reader: R) -> Result<Self, String> {
        let policy: Self = serde_json::from_reader(reader)
            .map_err(|e| format!("cannot parse current-street policy: {e}"))?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != FORMAT_VERSION {
            return Err(format!(
                "unsupported current-street format version {}",
                self.format_version
            ));
        }
        if self.street != self.config.starting_street()? {
            return Err("policy street differs from config".into());
        }
        let ids: HashSet<u32> = self.nodes.iter().map(|n| n.node).collect();
        if ids.len() != self.nodes.len() || !ids.contains(&self.root_node) {
            return Err("policy has duplicate node IDs or no root".into());
        }
        for (p, list) in self.combos.iter().enumerate() {
            if list.is_empty()
                || list.iter().map(|c| c.index).collect::<HashSet<_>>().len() != list.len()
            {
                return Err(format!("player {p} has empty or duplicate combo order"));
            }
        }
        for node in &self.nodes {
            if node.player > 1 || node.actions.is_empty() {
                return Err(format!("node {} has invalid player/actions", node.node));
            }
            let width = self.combos[node.player as usize].len();
            if node.strategy.len() != node.actions.len() * width {
                return Err(format!("node {} has invalid strategy length", node.node));
            }
            for i in 0..width {
                let sum: f32 = (0..node.actions.len())
                    .map(|a| node.strategy[a * width + i])
                    .sum();
                if !sum.is_finite()
                    || (sum - 1.0).abs() > 1e-4
                    || (0..node.actions.len()).any(|a| node.strategy[a * width + i] < 0.0)
                {
                    return Err(format!(
                        "node {} combo {i} has invalid probabilities",
                        node.node
                    ));
                }
            }
            for action in &node.actions {
                if let Child::Decision { node: child } = action.child {
                    if !ids.contains(&child) {
                        return Err(format!(
                            "node {} points to missing decision {child}",
                            node.node
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::{DcfrParams, Solver};
    use crate::config::Sizings;
    use crate::nlhe::NlheGame;
    use crate::solution::Solution;

    fn config(board: &str) -> SolveConfig {
        let mut cfg = SolveConfig {
            board: board.into(),
            oop_range: "KK,A4s,A5s".into(),
            ip_range: "TT,JJ".into(),
            effective_stack: 20.0,
            starting_pot: 5.0,
            raise_cap: 0,
            max_iterations: 1,
            ..SolveConfig::default()
        };
        cfg.sizings.oop.flop.bet = Sizings::new(&[50.0], false);
        cfg.sizings.ip.flop.bet = Sizings::new(&[50.0], false);
        cfg
    }

    #[test]
    fn roundtrip_probabilities_and_action_edges() {
        let mut cfg = config("Ks 7d 2c 8h 3d");
        cfg.oop_range = "random".into();
        cfg.ip_range = "random".into();
        let mut solver = Solver::new(NlheGame::new(&cfg).unwrap());
        solver.run(1, &DcfrParams::from_config(&cfg), 1, |_, _, _| {});
        let policy = CurrentStreetPolicy::from_solver(&solver, 0.1).unwrap();
        assert_eq!(policy.street, Street::River);
        assert_eq!(policy.meta.iterations, 1);
        assert_eq!(policy.meta.metric, "exploitability");
        assert_eq!(policy.meta.payoff_unit, "chips");
        assert_eq!(policy.nodes[0].node, policy.root_node);
        assert!(!policy.combos[0].is_empty());
        assert!(!policy.combos[1].is_empty());
        for node in &policy.nodes {
            let n = policy.combos[node.player as usize].len();
            assert_eq!(node.strategy.len(), n * node.actions.len());
            for i in 0..n {
                let sum: f32 = (0..node.actions.len())
                    .map(|a| node.strategy[a * n + i])
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "node {} combo {i}: {sum}",
                    node.node
                );
            }
            for action in &node.actions {
                assert!(
                    (action.amount_put
                        - (node.stacks[node.player as usize]
                            - action.result_stacks[node.player as usize]))
                        .abs()
                        < 1e-9
                );
                if matches!(action.label, ActionLabelData::Fold | ActionLabelData::Check) {
                    assert_eq!(action.amount_put, 0.0);
                }
                if let Child::Decision { node: child } = action.child {
                    assert!(policy.nodes.iter().any(|n| n.node == child));
                }
            }
        }
        assert!(policy
            .nodes
            .iter()
            .flat_map(|n| &n.actions)
            .any(|a| matches!(a.child, Child::HandTerminal { .. })));
        let mut bytes = Vec::new();
        policy.to_writer(&mut bytes).unwrap();
        assert_eq!(
            CurrentStreetPolicy::from_reader(bytes.as_slice()).unwrap(),
            policy
        );
    }

    #[test]
    fn flop_export_stops_at_turn_and_is_smaller_than_full_solution() {
        let cfg = config("2s 7d Jc");
        let mut solver = Solver::new(NlheGame::new(&cfg).unwrap());
        solver.run(1, &DcfrParams::from_config(&cfg), 1, |_, _, _| {});
        let policy = CurrentStreetPolicy::from_solver(&solver, 0.1).unwrap();
        assert_eq!(policy.street, Street::Flop);
        assert!(policy
            .nodes
            .iter()
            .all(|n| n.node == 0 || n.node < policy.full_tree_node_count));
        assert!(policy
            .nodes
            .iter()
            .flat_map(|n| &n.actions)
            .any(|a| matches!(
                a.child,
                Child::StreetBoundary {
                    street: Street::Turn,
                    ..
                }
            )));
        let mut compact = Vec::new();
        policy.to_writer(&mut compact).unwrap();
        let mut full = Vec::new();
        Solution::from_solver(&solver, 0.1)
            .to_writer(&mut full)
            .unwrap();
        assert!(
            compact.len() * 5 < full.len(),
            "compact {} full {}",
            compact.len(),
            full.len()
        );
    }
}

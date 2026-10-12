//! Research-only JSON batch API. EV comparisons use an unlocked chip-EV game.
use clap::Args;
use engine::{
    br::{self, StrategyProfile},
    cards::{self, Card},
    cfr::{DcfrParams, Solver},
    config::SolveConfig,
    current_street::CurrentStreetPolicy,
    evaluator::{eval5, eval7},
    game::{Game, NodeInfo},
    nlhe::NlheGame,
    solution::{ActionLabelData, Solution},
    tree::NodeKind,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Args)]
pub struct ResearchArgs {
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Deserialize)]
struct EquityRequest {
    board: String,
    ranges: [BTreeMap<String, f64>; 2],
    samples: usize,
    #[serde(default)]
    exact: bool,
}
fn six_rank(c: [Card; 6]) -> u32 {
    (0..6)
        .map(|omit| {
            let v: Vec<_> = c
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != omit)
                .map(|(_, x)| *x)
                .collect();
            eval5(&v.try_into().unwrap())
        })
        .max()
        .unwrap()
}
fn score(a: u32, b: u32) -> f64 {
    if a > b {
        1.0
    } else if a == b {
        0.5
    } else {
        0.0
    }
}
fn random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
fn parse_hand(h: &str) -> Result<[Card; 2], String> {
    cards::parse_cards(h)?
        .try_into()
        .map_err(|_| "combo must have two cards".into())
}
fn equity(req: EquityRequest) -> Result<Value, String> {
    let board = cards::parse_cards(&req.board)?;
    if board.len() != 3 {
        return Err("equity batch requires flop".into());
    }
    if !(32..=1_000_000).contains(&req.samples) {
        return Err("samples outside 32..1000000".into());
    }
    let ranges: Vec<Vec<_>> = req
        .ranges
        .iter()
        .map(|r| {
            r.iter()
                .map(|(h, w)| {
                    let c = parse_hand(h)?;
                    if !w.is_finite() || *w <= 0.0 || c.iter().any(|x| board.contains(x)) {
                        return Err("invalid weight or board-blocked combo".into());
                    }
                    let rank = eval5(&[board[0], board[1], board[2], c[0], c[1]]);
                    let turns: Vec<u32> = (0..52u8)
                        .map(|t| {
                            if board.contains(&t) || c.contains(&t) {
                                0
                            } else {
                                six_rank([board[0], board[1], board[2], c[0], c[1], t])
                            }
                        })
                        .collect();
                    Ok((h.clone(), c, *w, rank, turns))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<_, _>>()?;
    let mut out = [BTreeMap::new(), BTreeMap::new()];
    for player in 0..2 {
        for (name, h, _, own, own_turns) in &ranges[player] {
            let opp: Vec<_> = ranges[1 - player]
                .iter()
                .filter(|(_, v, _, _, _)| !h.contains(&v[0]) && !h.contains(&v[1]))
                .collect();
            let mass: f64 = opp.iter().map(|(_, _, w, _, _)| *w).sum();
            if mass <= 0.0 {
                out[player].insert(name.clone(), json!({"positive_joint_reach":false}));
                continue;
            }
            let strength: f64 = opp
                .iter()
                .map(|(_, _, w, rank, _)| w * score(*own, *rank))
                .sum::<f64>()
                / mass;
            let mut hist = [0.0; 10];
            let mut positive = 0.0;
            let mut negative = 0.0;
            let mut turn_values = Vec::new();
            for t in 0..52u8 {
                if board.contains(&t) || h.contains(&t) {
                    continue;
                }
                let tr = own_turns[t as usize];
                let mut tm = 0.0;
                let mut win = 0.0;
                for (_, v, w, rank, turns) in &opp {
                    if v.contains(&t) {
                        continue;
                    }
                    let before = score(*own, *rank);
                    let after = score(tr, turns[t as usize]);
                    tm += w;
                    win += w * after;
                    positive += w * (after - before).max(0.0) / (45.0 * mass);
                    negative += w * (before - after).max(0.0) / (45.0 * mass);
                }
                if tm > 0.0 {
                    let x = win / tm;
                    let probability = tm / (45.0 * mass);
                    hist[(x * 10.0).min(9.0) as usize] += probability;
                    turn_values.push((t, x, probability));
                }
            }
            let mut sum = 0.0;
            let mut square = 0.0;
            let mut count = 0usize;
            let mut half = 0.0;
            if req.exact {
                for (_, v, w, _, _) in &opp {
                    let deck: Vec<_> = (0..52u8)
                        .filter(|c| !board.contains(c) && !h.contains(c) && !v.contains(c))
                        .collect();
                    for (i, t) in deck.iter().enumerate() {
                        for r in &deck[i + 1..] {
                            let x = score(
                                eval7(&[board[0], board[1], board[2], h[0], h[1], *t, *r]),
                                eval7(&[board[0], board[1], board[2], v[0], v[1], *t, *r]),
                            );
                            sum += w * x / (mass * 990.0);
                            count += 1;
                        }
                    }
                }
            } else {
                let mut cumulative = Vec::new();
                let mut c = 0.0;
                for (_, _, w, _, _) in &opp {
                    c += w / mass;
                    cumulative.push(c)
                }
                let mut seed = 0x5249564552474c41u64;
                for c in board.iter().chain(h.iter()) {
                    seed = seed.wrapping_mul(1099511628211) ^ (*c as u64)
                }
                let decks: Vec<Vec<_>> = opp
                    .iter()
                    .map(|(_, v, _, _, _)| {
                        (0..52u8)
                            .filter(|c| !board.contains(c) && !h.contains(c) && !v.contains(c))
                            .collect()
                    })
                    .collect();
                for i in 0..req.samples {
                    let u = random(&mut seed);
                    let j = cumulative.partition_point(|x| *x < u).min(opp.len() - 1);
                    let v = opp[j].1;
                    let deck = &decks[j];
                    let ti = (random(&mut seed) * 45.0) as usize;
                    let ri = (random(&mut seed) * 44.0) as usize;
                    let r = deck[if ri >= ti { ri + 1 } else { ri }];
                    let t = deck[ti];
                    let x = score(
                        eval7(&[board[0], board[1], board[2], h[0], h[1], t, r]),
                        eval7(&[board[0], board[1], board[2], v[0], v[1], t, r]),
                    );
                    sum += x;
                    square += x * x;
                    if i < req.samples / 2 {
                        half += x
                    }
                    count += 1;
                }
                sum /= count as f64;
            }
            let se = if req.exact {
                0.0
            } else {
                ((square / count as f64 - sum * sum).max(0.0) / count as f64).sqrt()
            };
            let turn_mean: f64 = turn_values.iter().map(|(_, x, p)| x * p).sum();
            let turn_std = turn_values
                .iter()
                .map(|(_, x, p)| (x - turn_mean).powi(2) * p)
                .sum::<f64>()
                .sqrt();
            out[player].insert(name.clone(),json!({"positive_joint_reach":true,"current_strength":strength,"showdown_equity":sum,
                "equity_standard_error":se,"sample_count":count,"equity_half_sample_delta":if req.exact {0.0} else {(half/(req.samples/2) as f64-sum).abs()},
                "future_strength_hist":hist,"future_strength_mean":turn_mean,"future_strength_std":turn_std,
                "positive_potential":positive,"negative_potential":negative,"turn_strength":turn_values}));
        }
    }
    Ok(
        json!({"format_version":2,"method":if req.exact {"exact_legal_pair_runouts"} else {"weighted_opponent_legal_runout_sampling"},"players":out}),
    )
}
/// Fully validated immutable continuation. Missing rows never receive uniform fallback.
struct SavedProfile<'a> {
    rows: HashMap<u32, &'a [f32]>,
}
impl<'a> SavedProfile<'a> {
    fn validate(sol: &'a Solution, game: &NlheGame) -> Result<Self, String> {
        if sol.node_count as usize != game.tree().len() {
            return Err("saved node count mismatch".into());
        }
        if sol.meta.payoff_unit != "chips" {
            return Err("saved payoff unit must be chips".into());
        }
        let by_node: HashMap<_, _> = sol.nodes.iter().map(|ns| (ns.node, ns)).collect();
        let mut rows = HashMap::new();
        for ns in &sol.nodes {
            if rows.insert(ns.node, ns.strategy.as_slice()).is_some() {
                return Err("duplicate saved decision node".into());
            }
        }
        for id in 0..game.tree().len() as u32 {
            if let NodeKind::Decision { player, actions } = &game.tree().node(id).kind {
                let ns = by_node
                    .get(&id)
                    .ok_or_else(|| format!("missing saved decision node {id}"))?;
                let width = game.combo_count(id, *player);
                let labels: Vec<ActionLabelData> = actions.iter().map(|a| a.label.into()).collect();
                if ns.player != *player
                    || ns.actions != labels
                    || ns.combo_count as usize != width
                    || ns.strategy.len() != actions.len() * width
                {
                    return Err(format!("saved decision structure mismatch at {id}"));
                }
                for i in 0..width {
                    let mut sum = 0.0;
                    for a in 0..actions.len() {
                        let p = ns.strategy[a * width + i];
                        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                            return Err(format!("invalid saved probability at {id}"));
                        }
                        sum += p;
                    }
                    if (sum - 1.0).abs() > 1e-4 {
                        return Err(format!("invalid saved probability sum at {id}"));
                    }
                }
            }
        }
        for p in 0..2u8 {
            let combos = game.live_combos(game.root(), p);
            let indices = game.combo_indices(game.root(), p);
            if combos.len() != sol.root_combos[p as usize].len() {
                return Err("saved root combos mismatch".into());
            }
            for (i, &(a, b)) in combos.iter().enumerate() {
                let stored = &sol.root_combos[p as usize][i];
                if stored.index != indices[i]
                    || stored.cards
                        != format!("{}{}", cards::card_to_string(a), cards::card_to_string(b))
                {
                    return Err("saved root combo order mismatch".into());
                }
            }
        }
        Ok(Self { rows })
    }
}
impl StrategyProfile for SavedProfile<'_> {
    fn strategy_into(&self, node: u32, out: &mut [f32]) {
        out.copy_from_slice(
            self.rows
                .get(&node)
                .expect("validated complete saved decision row"),
        );
    }
}
struct Overlay<'a, P: StrategyProfile> {
    base: &'a P,
    rows: HashMap<u32, Vec<f32>>,
}
impl<P: StrategyProfile> StrategyProfile for Overlay<'_, P> {
    fn strategy_into(&self, node: u32, out: &mut [f32]) {
        if let Some(row) = self.rows.get(&node) {
            out.copy_from_slice(row)
        } else {
            self.base.strategy_into(node, out)
        }
    }
}
fn atomic_json(path: &std::path::Path, value: &Value) -> Result<(), String> {
    let temporary = path.with_extension("tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::rename(temporary, path).map_err(|e| e.to_string())
}
fn candidate_rows(
    payload: &CurrentStreetPolicy,
    teacher: &CurrentStreetPolicy,
    actors: &[u8],
) -> Result<HashMap<u32, Vec<f32>>, String> {
    if payload.config != teacher.config
        || payload.combos != teacher.combos
        || payload.nodes.len() != teacher.nodes.len()
    {
        return Err("candidate state/combos/nodes mismatch".into());
    }
    let mut rows = HashMap::new();
    for node in teacher.nodes.iter().filter(|n| actors.contains(&n.player)) {
        let c = payload
            .nodes
            .iter()
            .find(|n| n.node == node.node)
            .ok_or("missing candidate node")?;
        if c.actions != node.actions
            || c.player != node.player
            || c.strategy.len() != node.strategy.len()
        {
            return Err(format!(
                "candidate action semantics mismatch at node {}",
                node.node
            ));
        }
        let width = teacher.combos[node.player as usize].len();
        for i in 0..width {
            let col: Vec<_> = (0..c.actions.len())
                .map(|a| c.strategy[a * width + i])
                .collect();
            let sum = col.iter().sum::<f32>();
            if col.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                || (sum - 1.0).abs() > 1e-4
            {
                return Err("candidate has invalid or missing combo row".into());
            }
        }
        rows.insert(c.node, c.strategy.clone());
    }
    Ok(rows)
}
/// Evaluate one legal turn boundary under the original flop payoff convention.
/// Reach factors exclude chance; the chance edge is applied once to root mass.
fn boundary_value<P: StrategyProfile>(
    game: &NlheGame,
    teacher: &CurrentStreetPolicy,
    profile: &P,
    actor: u8,
    query: &Value,
) -> Result<Value, String> {
    let history: Vec<String> =
        serde_json::from_value(query["history"].clone()).map_err(|e| e.to_string())?;
    let turn = cards::parse_card(query["turn"].as_str().ok_or("turn missing")?)?;
    let mut node = teacher.root_node;
    let mut reach = [game.root_weights(0).to_vec(), game.root_weights(1).to_vec()];
    for token in &history {
        let current = teacher
            .nodes
            .iter()
            .find(|n| n.node == node)
            .ok_or("history extends past flop")?;
        let a = current
            .actions
            .iter()
            .position(|a| &a.action_id == token)
            .ok_or("unknown history action")?;
        let who = current.player as usize;
        let width = reach[who].len();
        let mut row = vec![0.0; current.strategy.len()];
        profile.strategy_into(node, &mut row);
        for i in 0..width {
            reach[who][i] *= row[a * width + i];
        }
        node = game.child(node, a);
    }
    let count = match game.node(node) {
        NodeInfo::Chance { num_outcomes } => num_outcomes,
        _ => return Err("history must end at turn chance boundary".into()),
    };
    if game.board_at(node).contains(&turn) {
        return Err("turn repeats public card".into());
    }
    let edge = (0..count)
        .map(|k| game.chance_outcome(node, k))
        .find(|e| game.board_at(e.child).len() == 4 && game.board_at(e.child).contains(&turn))
        .ok_or("illegal turn")?;
    if !matches!(game.node(edge.child), NodeInfo::Decision { .. }) {
        return Err("all-in runout has no turn policy".into());
    }
    let live: [Vec<f32>; 2] = std::array::from_fn(|p| {
        edge.parent_of_child[p]
            .iter()
            .map(|&i| reach[p][i as usize])
            .collect()
    });
    let compatible = game.compatible_mass(edge.child, actor, &live[(1 - actor) as usize]);
    let mass: f32 = live[actor as usize]
        .iter()
        .zip(&compatible)
        .map(|(a, b)| a * b)
        .sum();
    if mass <= 0.0 {
        return Err("zero compatible boundary reach".into());
    }
    let mut values = vec![0.0; live[actor as usize].len()];
    br::subtree_values(
        game,
        edge.child,
        actor,
        profile,
        &live[(1 - actor) as usize],
        false,
        &mut values,
    );
    let conditional: f32 = live[actor as usize]
        .iter()
        .zip(&values)
        .map(|(a, b)| a * b)
        .sum::<f32>()
        / mass;
    let geometry = game.node_at(edge.child);
    let contribution = game.config().effective_stack - geometry.stacks[actor as usize];
    let offset = geometry.pot / 2.0 - game.config().starting_pot / 2.0 - contribution;
    let probability = mass * edge.weight / game.normalizer();
    let mut result = json!({"history":history,"turn":cards::card_to_string(turn),"conditional_reference_ev_bb":conditional,
        "root_probability":probability,"turn_to_flop_offset_bb":offset,"pot":geometry.pot,"stacks":geometry.stacks});
    if let Some(override_ev) = query.get("override_turn_ev_bb") {
        let value = override_ev
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("invalid boundary override")?;
        // Reject physically impossible utility labels as well as non-finite labels.
        let limit = geometry.pot + 2.0 * geometry.effective_stack();
        if value.abs() > limit {
            return Err("boundary override outside chip bounds".into());
        }
        result["override_flop_ev_bb"] = json!(value + offset);
        result["root_delta_bb"] = json!(probability as f64 * (value + offset - conditional as f64));
    }
    Ok(result)
}

fn evaluate(request: &Value, output: &PathBuf) -> Result<Value, String> {
    let cfg: SolveConfig =
        serde_json::from_value(request["config"].clone()).map_err(|e| e.to_string())?;
    if cfg.tournament.is_some() || !cfg.locks.is_empty() || cfg.rake.percent != 0.0 {
        return Err("evaluation requires unlocked, rake-free chip-EV config".into());
    }
    for field in ["load_solution", "export_solution"] {
        if request.get(field).is_some()
            && request[field].as_str().filter(|s| !s.is_empty()).is_none()
        {
            return Err(format!("{field} must be a nonempty path string"));
        }
    }
    let load_clock = Instant::now();
    let loaded = request["load_solution"]
        .as_str()
        .map(Solution::load)
        .transpose()?;
    let load_seconds = load_clock.elapsed().as_secs_f64();
    if let Some(sol) = &loaded {
        if sol.config != cfg {
            return Err("loaded solution config mismatch".into());
        }
        if request.get("checkpoints").is_some() || request.get("export_solution").is_some() {
            return Err("loaded evaluation cannot run checkpoints or export a new solve".into());
        }
    }
    let checkpoints: Vec<u64> = if loaded.is_some() {
        vec![]
    } else {
        let values: Vec<u64> =
            serde_json::from_value(request["checkpoints"].clone()).map_err(|e| e.to_string())?;
        if values.is_empty() || values.windows(2).any(|w| w[0] >= w[1]) || values[0] == 0 {
            return Err("checkpoints must increase".into());
        }
        values
    };
    let construction_clock = Instant::now();
    let game = NlheGame::new(&cfg)?;
    let construction_seconds = construction_clock.elapsed().as_secs_f64();
    if let Some(sol) = &loaded {
        let profile = SavedProfile::validate(sol, &game)?;
        let report = br::ExploitReport {
            br: [0.0; 2],
            gain: [0.0; 2],
            chips: sol.meta.exploitability_chips,
            pct_of_pot: sol.meta.exploitability_pct_of_pot,
        };
        let teacher = CurrentStreetPolicy::from_profile_with_report(
            &game,
            &profile,
            sol.meta.iterations,
            0,
            construction_clock.elapsed().as_secs_f64(),
            report,
        )?;
        validate_evaluation_request(request, &game, &teacher, &profile)?;
        if request["validate_only"].as_bool().unwrap_or(false) {
            return Ok(json!({"status":"validated","decision_nodes":teacher.nodes.len()}));
        }
        let mut result = evaluate_profile(
            request,
            &cfg,
            &game,
            &profile,
            teacher,
            vec![],
            &construction_clock,
            0,
            None,
        )?;
        result["timings"]["load_solution_seconds"] = json!(load_seconds);
        result["timings"]["game_construction_seconds"] = json!(construction_seconds);
        result["wall_seconds"] = json!(construction_clock.elapsed().as_secs_f64() + load_seconds);
        result["reference_source"] = json!("full_solution");
        result["cfr_iterations_run"] = json!(0);
        return Ok(result);
    }
    let params = DcfrParams::from_config(&cfg);
    let mut solver = Solver::new(game);
    let clock = Instant::now();
    // Validate candidate geometry and probabilities before spending iterations
    // or computing action values. Only public geometry is used at this point.
    let preflight = CurrentStreetPolicy::from_solver_with_report(
        &solver,
        0.0,
        br::ExploitReport {
            br: [0.0, 0.0],
            gain: [0.0, 0.0],
            chips: 0.0,
            pct_of_pot: 0.0,
        },
    )?;
    validate_evaluation_request(request, solver.game(), &preflight, &solver.average())?;
    if request["validate_only"].as_bool().unwrap_or(false) {
        return Ok(
            json!({"status":"validated","decision_nodes":preflight.nodes.len(),"wall_seconds":clock.elapsed().as_secs_f64()}),
        );
    }
    let mut audits = Vec::new();
    let mut last_teacher = None;
    let mut last_values = None;
    for iteration in checkpoints {
        let solve_clock = Instant::now();
        solver.run(iteration - solver.iterations(), &params, 0, |_, _, _| {});
        let solve_seconds = solve_clock.elapsed().as_secs_f64();
        let report_clock = Instant::now();
        let profile = solver.average();
        let ev = [
            br::expected_value(solver.game(), 0, &profile),
            br::expected_value(solver.game(), 1, &profile),
        ];
        let br = [
            br::best_response_value(solver.game(), 0, &profile),
            br::best_response_value(solver.game(), 1, &profile),
        ];
        let report = br::ExploitReport {
            br,
            gain: [br[0] - ev[0], br[1] - ev[1]],
            chips: br[0] + br[1],
            pct_of_pot: 100.0 * (br[0] + br[1]) / cfg.starting_pot as f32,
        };
        last_values = Some((ev, br));
        let report_seconds = report_clock.elapsed().as_secs_f64();
        let export = CurrentStreetPolicy::from_solver_with_report(
            &solver,
            clock.elapsed().as_secs_f64(),
            report,
        )?;
        let path = output.with_extension(format!("iter_{iteration:06}.json"));
        fs::write(
            &path,
            serde_json::to_vec(&export).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        last_teacher = Some(export);
        audits.push(json!({"iterations":iteration,"wall_seconds":clock.elapsed().as_secs_f64(),"cfr_seconds":solve_seconds,"reference_report_seconds":report_seconds,"export":path,
            "ev":ev,"br":report.br,"exploitability_pct_pot":report.pct_of_pot}));
        atomic_json(output, &json!({"status":"running","checkpoints":audits}))?;
    }
    let game = solver.game();
    let base = solver.average();
    let teacher = last_teacher.ok_or("missing teacher checkpoint")?;
    let mut result = evaluate_profile(
        request,
        &cfg,
        game,
        &base,
        teacher,
        audits,
        &clock,
        solver.storage_bytes(),
        last_values,
    )?;
    if let Some(path) = request["export_solution"].as_str() {
        let export_clock = Instant::now();
        let ev: [f32; 2] = std::array::from_fn(|i| {
            result["reference_values"]["ev_bb"][i].as_f64().unwrap() as f32
        });
        let br: [f32; 2] = std::array::from_fn(|i| {
            result["reference_values"]["br_bb"][i].as_f64().unwrap() as f32
        });
        let report = br::ExploitReport {
            br,
            gain: [br[0] - ev[0], br[1] - ev[1]],
            chips: br[0] + br[1],
            pct_of_pot: 100.0 * (br[0] + br[1]) / cfg.starting_pot as f32,
        };
        let tmp = PathBuf::from(path).with_extension("solution.tmp");
        let mut writer = BufWriter::new(fs::File::create(&tmp).map_err(|e| e.to_string())?);
        Solution::stream_solver_with_values(
            &solver,
            &mut writer,
            clock.elapsed().as_secs_f64(),
            ev,
            report,
        )?;
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);
        fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        result["full_solution"] = json!({"path":path,"bytes":fs::metadata(path).map_err(|e| e.to_string())?.len(),
            "decision_nodes":(0..game.tree().len() as u32).filter(|&id| matches!(game.tree().node(id).kind,NodeKind::Decision{..})).count(),"capture_and_serialization_seconds":export_clock.elapsed().as_secs_f64()});
    }
    result["timings"]["game_construction_seconds"] = json!(construction_seconds);
    result["wall_seconds"] = json!(construction_clock.elapsed().as_secs_f64());
    result["cfr_iterations_run"] = json!(solver.iterations());
    Ok(result)
}

fn validate_evaluation_request<P: StrategyProfile>(
    request: &Value,
    game: &NlheGame,
    preflight: &CurrentStreetPolicy,
    base: &P,
) -> Result<(), String> {
    for candidate in request["candidates"].as_array().into_iter().flatten() {
        let actor = candidate["actor"].as_u64().ok_or("actor missing")?;
        if actor > 1 {
            return Err("actor must be 0 or 1".into());
        }
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        candidate_rows(&payload, preflight, &[actor as u8])?;
    }
    for candidate in request["joint_candidates"].as_array().into_iter().flatten() {
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        candidate_rows(&payload, preflight, &[0, 1])?;
    }
    for candidate in request["boundary_evaluations"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let actor = candidate["actor"]
            .as_u64()
            .filter(|a| *a <= 1)
            .ok_or("invalid boundary actor")? as u8;
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        let rows = candidate_rows(&payload, preflight, &[actor])?;
        let profile = Overlay { base, rows };
        let mut seen = std::collections::HashSet::new();
        for q in candidate["queries"]
            .as_array()
            .ok_or("boundary queries missing")?
        {
            let key = serde_json::to_string(&json!([q["history"], q["turn"]]))
                .map_err(|e| e.to_string())?;
            if !seen.insert(key) {
                return Err("duplicate boundary override/query".into());
            }
            boundary_value(game, preflight, &profile, actor, q)?;
        }
    }
    Ok(())
}

fn evaluate_profile<P: StrategyProfile>(
    request: &Value,
    cfg: &SolveConfig,
    game: &NlheGame,
    base: &P,
    teacher: CurrentStreetPolicy,
    audits: Vec<Value>,
    clock: &Instant,
    storage_bytes: usize,
    cached_values: Option<([f32; 2], [f32; 2])>,
) -> Result<Value, String> {
    let reference_clock = Instant::now();
    let (reference_ev, reference_br) = cached_values.unwrap_or_else(|| {
        (
            [
                br::expected_value(game, 0, base),
                br::expected_value(game, 1, base),
            ],
            [
                br::best_response_value(game, 0, base),
                br::best_response_value(game, 1, base),
            ],
        )
    });
    let reference_seconds = reference_clock.elapsed().as_secs_f64();
    let evaluation_clock = Instant::now();
    let action_clock = Instant::now();
    let mut action_values = serde_json::Map::new();
    if request["action_values"].as_bool().unwrap_or(false) {
        let nodes: HashMap<_, _> = teacher.nodes.iter().map(|n| (n.node, n)).collect();
        let mut pending = vec![(
            teacher.root_node,
            [game.root_weights(0).to_vec(), game.root_weights(1).to_vec()],
        )];
        while let Some((id, reaches)) = pending.pop() {
            let node = nodes[&id];
            let actor = node.player;
            let width = teacher.combos[actor as usize].len();
            let opponent = 1 - actor;
            let mut actions = serde_json::Map::new();
            for (a, action) in node.actions.iter().enumerate() {
                let mut forced = vec![0.0; node.strategy.len()];
                forced[a * width..(a + 1) * width].fill(1.0);
                let overlay = Overlay {
                    base,
                    rows: HashMap::from([(id, forced)]),
                };
                let mut values = vec![0.0; width];
                br::subtree_values(
                    game,
                    id,
                    actor,
                    &overlay,
                    &reaches[opponent as usize],
                    false,
                    &mut values,
                );
                let own = game.live_combos(id, actor);
                let other = game.live_combos(id, opponent);
                let mut normalized = Vec::new();
                for (i, &(x, y)) in own.iter().enumerate() {
                    let mass: f32 = other
                        .iter()
                        .zip(&reaches[opponent as usize])
                        .filter(|((u, v), _)| *u != x && *u != y && *v != x && *v != y)
                        .map(|(_, w)| *w)
                        .sum();
                    normalized.push(if mass > 0.0 {
                        Some(values[i] / mass)
                    } else {
                        None
                    });
                }
                actions.insert(action.action_id.clone(), json!(normalized));
                if let engine::current_street::Child::Decision { node: child } = action.child {
                    let mut next = reaches.clone();
                    for i in 0..width {
                        next[actor as usize][i] *= node.strategy[a * width + i]
                    }
                    pending.push((child, next));
                }
            }
            action_values.insert(id.to_string(), Value::Object(actions));
        }
    }
    let action_seconds = action_clock.elapsed().as_secs_f64();
    let unilateral_clock = Instant::now();
    let mut results = Vec::new();
    for candidate in request["candidates"].as_array().into_iter().flatten() {
        let actor_id = candidate["actor"].as_u64().ok_or("actor missing")?;
        if actor_id > 1 {
            return Err("actor must be 0 or 1".into());
        }
        let actor = actor_id as u8;
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        let rows = candidate_rows(&payload, &teacher, &[actor])?;
        let profile = Overlay { base, rows };
        let reference_ev = reference_ev[actor as usize];
        let candidate_ev = br::expected_value(game, actor, &profile);
        let reference_br = reference_br[(1 - actor) as usize];
        let candidate_br = br::best_response_value(game, 1 - actor, &profile);
        results.push(json!({"id":candidate["id"],"actor":actor,"reference_ev_bb":reference_ev,"candidate_ev_bb":candidate_ev,
            "fixed_opponent_loss_bb":reference_ev-candidate_ev,"security_loss_bb":candidate_br-reference_br,
            "fixed_opponent_loss_pct_pot":100.0*(reference_ev-candidate_ev)/cfg.starting_pot as f32,
            "security_loss_pct_pot":100.0*(candidate_br-reference_br)/cfg.starting_pot as f32}));
    }
    let unilateral_seconds = unilateral_clock.elapsed().as_secs_f64();
    let joint_clock = Instant::now();
    let mut joint_results = Vec::new();
    for candidate in request["joint_candidates"].as_array().into_iter().flatten() {
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        let rows = candidate_rows(&payload, &teacher, &[0, 1])?;
        let profile = Overlay { base, rows };
        let ev = [
            br::expected_value(game, 0, &profile),
            br::expected_value(game, 1, &profile),
        ];
        let best_response = [
            br::best_response_value(game, 0, &profile),
            br::best_response_value(game, 1, &profile),
        ];
        joint_results.push(json!({"id":candidate["id"],"candidate_ev_bb":ev,
            "candidate_ev_pct_pot":[100.0*ev[0]/cfg.starting_pot as f32,100.0*ev[1]/cfg.starting_pot as f32],
            "best_response_bb":best_response,
            "unilateral_gain_bb":[best_response[0]-ev[0],best_response[1]-ev[1]],
            "note":"Both-candidate play; not a unilateral approximation-loss comparison"}));
    }
    let joint_seconds = joint_clock.elapsed().as_secs_f64();
    let boundary_clock = Instant::now();
    let mut boundary_results = Vec::new();
    for candidate in request["boundary_evaluations"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let actor = candidate["actor"]
            .as_u64()
            .ok_or("boundary actor missing")? as u8;
        let payload: CurrentStreetPolicy =
            serde_json::from_value(candidate["policy"].clone()).map_err(|e| e.to_string())?;
        let rows = candidate_rows(&payload, &teacher, &[actor])?;
        let profile = Overlay { base, rows };
        let mut queries = Vec::new();
        let mut delta = 0.0;
        for query in candidate["queries"]
            .as_array()
            .ok_or("boundary queries missing")?
        {
            let result = boundary_value(game, &teacher, &profile, actor, query)?;
            delta += result["root_delta_bb"].as_f64().unwrap_or(0.0);
            queries.push(result);
        }
        let fixed = br::expected_value(game, actor, &profile);
        boundary_results.push(json!({"id":candidate["id"],"actor":actor,"fixed_continuation_ev_bb":fixed,
            "queries":queries,"override_root_delta_bb":delta,"partial_override_ev_bb":fixed as f64+delta,
            "scope":"Fixed-policy EV only; unsupplied continuations retain reference. No best-response claim."}));
    }
    Ok(
        json!({"status":"complete","format_version":2,"checkpoints":audits,"unilateral":results,"both_candidates":joint_results,"boundary_evaluations":boundary_results,"wall_seconds":clock.elapsed().as_secs_f64(),"storage_bytes":storage_bytes,"reference_values":{"ev_bb":reference_ev,"br_bb":reference_br},"timings":{"reference_values_seconds":reference_seconds,"action_values_seconds":action_seconds,
                "unilateral_seconds":unilateral_seconds,"joint_seconds":joint_seconds,
                "boundary_seconds":boundary_clock.elapsed().as_secs_f64(),
                "candidate_evaluation_seconds":evaluation_clock.elapsed().as_secs_f64()},"reference_policy":teacher,"action_values":action_values}),
    )
}
/// Export public geometry without iterations or EV/best-response evaluation.
/// Uniform placeholder rows are replaced by the standalone flop generator.
fn template(request: &Value) -> Result<Value, String> {
    let cfg: SolveConfig =
        serde_json::from_value(request["config"].clone()).map_err(|e| e.to_string())?;
    if cfg.tournament.is_some() || !cfg.locks.is_empty() || cfg.rake.percent != 0.0 {
        return Err("template requires unlocked, rake-free chip-EV config".into());
    }
    let clock = Instant::now();
    let solver = Solver::new(NlheGame::new(&cfg)?);
    let mut policy = CurrentStreetPolicy::from_solver_with_report(
        &solver,
        clock.elapsed().as_secs_f64(),
        br::ExploitReport {
            br: [0.0; 2],
            gain: [0.0; 2],
            chips: 0.0,
            pct_of_pot: 0.0,
        },
    )?;
    policy.meta.metric = "untrained_template".into();
    Ok(json!({"status":"complete", "template":policy,
        "iterations":0, "values_measured":false,
        "wall_seconds":clock.elapsed().as_secs_f64(),
        "storage_bytes":solver.storage_bytes()}))
}

pub fn run(args: ResearchArgs) -> Result<(), String> {
    let request: Value =
        serde_json::from_str(&fs::read_to_string(args.request).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let threads = request["threads"].as_u64().unwrap_or(4);
    if !(1..=64).contains(&threads) {
        return Err("threads outside 1..64".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .map_err(|e| e.to_string())?;
    let result = match request["operation"].as_str() {
        Some("equity") => equity(serde_json::from_value(request).map_err(|e| e.to_string())?)?,
        Some("evaluate") => evaluate(&request, &args.output)?,
        Some("template") => template(&request)?,
        _ => return Err("unknown research operation".into()),
    };
    atomic_json(&args.output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn continuation_cfg() -> SolveConfig {
        let mut cfg = SolveConfig::default();
        cfg.board = "2c 7d Ts 4h 9c".into();
        cfg.oop_range = "AA,KK,QQ".into();
        cfg.ip_range = "AA,KK,QQ".into();
        cfg.effective_stack = 2.0;
        cfg.starting_pot = 4.0;
        cfg.raise_cap = 0;
        cfg.sizings.oop.river.bet = engine::config::Sizings::new(&[100.0], false);
        cfg.sizings.ip.river.bet = engine::config::Sizings::new(&[100.0], false);
        cfg
    }

    #[test]
    fn full_continuation_rows_values_and_rejection_are_exact() {
        let cfg = continuation_cfg();
        let mut solver = Solver::new(NlheGame::new(&cfg).unwrap());
        solver.run(40, &DcfrParams::from_config(&cfg), 0, |_, _, _| {});
        let original = Solution::from_solver(&solver, 0.0);
        let mut bytes = vec![];
        original.to_writer(&mut bytes).unwrap();
        let loaded = Solution::from_reader(bytes.as_slice()).unwrap();
        let profile = SavedProfile::validate(&loaded, solver.game()).unwrap();
        for node in &original.nodes {
            let mut row = vec![0.0; node.strategy.len()];
            profile.strategy_into(node.node, &mut row);
            assert_eq!(row, solver.average_strategy(node.node));
        }
        for actor in 0..2 {
            assert_eq!(
                br::expected_value(solver.game(), actor, &profile),
                solver.expected_value(actor)
            );
            assert_eq!(
                br::best_response_value(solver.game(), actor, &profile),
                br::best_response_value(solver.game(), actor, &solver.average())
            );
        }
        let mut missing = loaded.clone();
        missing.nodes.pop();
        assert!(SavedProfile::validate(&missing, solver.game()).is_err());
        let mut duplicate = loaded.clone();
        duplicate.nodes.push(duplicate.nodes[0].clone());
        assert!(SavedProfile::validate(&duplicate, solver.game()).is_err());
        let mut bad = loaded.clone();
        bad.nodes[0].strategy[0] = f32::NAN;
        assert!(SavedProfile::validate(&bad, solver.game()).is_err());
        let mut bad = loaded.clone();
        bad.root_combos[0].swap(0, 1);
        assert!(SavedProfile::validate(&bad, solver.game()).is_err());
    }

    #[test]
    fn saved_flop_continuation_preserves_turn_and_river_rows_and_boundaries() {
        let mut cfg = continuation_cfg();
        cfg.board = "2c 7d Ts".into();
        cfg.oop_range = "KcKd:1,KhKs:0.5".into();
        cfg.ip_range = "AhAd:1".into();
        let mut solver = Solver::new(NlheGame::new(&cfg).unwrap());
        solver.run(2, &DcfrParams::from_config(&cfg), 0, |_, _, _| {});
        let solution = Solution::from_solver(&solver, 0.0);
        let profile = SavedProfile::validate(&solution, solver.game()).unwrap();
        let teacher = CurrentStreetPolicy::from_solver(&solver, 0.0).unwrap();
        assert!(solution.nodes.len() > teacher.nodes.len());
        for ns in &solution.nodes {
            let mut row = vec![0.0; ns.strategy.len()];
            profile.strategy_into(ns.node, &mut row);
            assert_eq!(row, solver.average_strategy(ns.node));
        }
        for actor in 0..2 {
            let q = json!({"history":["check","check"],"turn":"4h"});
            assert_eq!(
                boundary_value(solver.game(), &teacher, &profile, actor, &q).unwrap(),
                boundary_value(solver.game(), &teacher, &solver.average(), actor, &q).unwrap()
            );
            assert_eq!(
                br::expected_value(solver.game(), actor, &profile),
                solver.expected_value(actor)
            );
        }
    }

    #[test]
    fn research_loaded_batch_matches_fresh_and_runs_no_cfr() {
        let cfg = continuation_cfg();
        let dir =
            std::env::temp_dir().join(format!("research_continuation_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let solution = dir.join("full.json");
        let output = dir.join("measurement.json");
        let initial = evaluate(
            &json!({"config":cfg,"checkpoints":[40],"export_solution":solution}),
            &output,
        )
        .unwrap();
        let teacher: CurrentStreetPolicy =
            serde_json::from_value(initial["reference_policy"].clone()).unwrap();
        let mut bad = teacher.clone();
        // Always fold facing a bet; choose check otherwise. This loses value for both roles.
        for node in &mut bad.nodes {
            let a = node
                .actions
                .iter()
                .position(|a| a.action_id == "fold")
                .unwrap_or(0);
            let width = bad.combos[node.player as usize].len();
            node.strategy.fill(0.0);
            node.strategy[a * width..(a + 1) * width].fill(1.0);
        }
        let candidates = json!([
            {"id":"self0","actor":0,"policy":teacher}, {"id":"self1","actor":1,"policy":teacher},
            {"id":"bad0","actor":0,"policy":bad}, {"id":"bad1","actor":1,"policy":bad}]);
        let fresh = evaluate(
            &json!({"config":cfg,"checkpoints":[40],"candidates":candidates,"action_values":true}),
            &output,
        )
        .unwrap();
        let saved = evaluate(&json!({"config":cfg,"load_solution":solution,"candidates":candidates,"action_values":true}),&output).unwrap();
        assert_eq!(saved["cfr_iterations_run"], 0);
        assert_eq!(saved["storage_bytes"], 0);
        assert_eq!(saved["unilateral"], fresh["unilateral"]);
        assert_eq!(saved["action_values"], fresh["action_values"]);
        assert_eq!(
            saved["reference_policy"]["nodes"],
            fresh["reference_policy"]["nodes"]
        );
        for i in 0..2 {
            assert_eq!(saved["unilateral"][i]["fixed_opponent_loss_bb"], 0.0);
        }
        for i in 2..4 {
            assert!(
                saved["unilateral"][i]["fixed_opponent_loss_bb"]
                    .as_f64()
                    .unwrap()
                    > 0.0,
                "{}",
                saved["unilateral"]
            );
        }
        let mut changed = cfg.clone();
        changed.starting_pot += 1.0;
        assert!(
            evaluate(&json!({"config":changed,"load_solution":solution}), &output)
                .unwrap_err()
                .contains("config mismatch")
        );
        let mut incomplete: Solution = Solution::load(&solution).unwrap();
        incomplete.nodes.pop();
        incomplete.save(&solution).unwrap();
        assert!(
            evaluate(&json!({"config":cfg,"load_solution":solution}), &output)
                .unwrap_err()
                .contains("missing saved")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn policy_chip_geometry_survives_json_roundtrip() {
        let mut cfg = SolveConfig::default();
        cfg.board = "2c 7d Ts 4h 9c".into();
        cfg.oop_range = "KcKd".into();
        cfg.ip_range = "AhAd".into();
        cfg.effective_stack = 2.875;
        cfg.starting_pot = 5.625;
        cfg.allin_threshold = 100.0;
        cfg.sizings.oop.river.bet.percents = vec![33.0];
        cfg.sizings.ip.river.bet.percents = vec![33.0];
        let params = DcfrParams::from_config(&cfg);
        let mut solver = Solver::new(NlheGame::new(&cfg).unwrap());
        solver.run(10, &params, 0, |_, _, _| {});
        let teacher = CurrentStreetPolicy::from_solver(&solver, 0.0).unwrap();
        // Reproduce the JSON Value hop in real research requests, including
        // subtraction-derived chip values with long decimal representations.
        let encoded = serde_json::to_string(&teacher).unwrap();
        let value: Value = serde_json::from_str(&encoded).unwrap();
        let mut candidate: CurrentStreetPolicy = serde_json::from_value(value).unwrap();
        assert_eq!(candidate.nodes, teacher.nodes);
        assert!(candidate_rows(&candidate, &teacher, &[0, 1]).is_ok());
        candidate.nodes[0].actions[0].amount_put += 0.01;
        assert!(candidate_rows(&candidate, &teacher, &[0, 1]).is_err());
    }

    #[test]
    fn subtraction_derived_chip_number_is_parsed_exactly() {
        let value: f64 = serde_json::from_str("1.8599999999999994").unwrap();
        assert_eq!(value.to_bits(), 1.8599999999999994_f64.to_bits());
    }
    #[test]
    fn boundary_override_matches_direct_full_tree_walk() {
        use engine::br::UniformProfile;
        let mut cfg = SolveConfig::default();
        cfg.board = "2c 7d Ts".into();
        cfg.oop_range = "KcKd:1,KhKs:0.5".into();
        cfg.ip_range = "AhAd:1".into();
        cfg.effective_stack = 3.0;
        cfg.starting_pot = 5.625;
        cfg.allin_threshold = 100.0;
        let solver = Solver::new(NlheGame::new(&cfg).unwrap());
        let teacher = CurrentStreetPolicy::from_solver_with_report(
            &solver,
            0.0,
            br::ExploitReport {
                br: [0.0; 2],
                gain: [0.0; 2],
                chips: 0.0,
                pct_of_pot: 0.0,
            },
        )
        .unwrap();
        let game = solver.game();
        let uniform = UniformProfile(game);
        let mut node = teacher.root_node;
        for _ in 0..2 {
            let current = teacher.nodes.iter().find(|n| n.node == node).unwrap();
            let a = current
                .actions
                .iter()
                .position(|a| a.action_id == "check")
                .unwrap();
            node = game.child(node, a);
        }
        let num = match game.node(node) {
            NodeInfo::Chance { num_outcomes } => num_outcomes,
            _ => panic!(),
        };
        let card = cards::parse_card("4h").unwrap();
        let edge = (0..num)
            .map(|k| game.chance_outcome(node, k))
            .find(|e| game.board_at(e.child).contains(&card))
            .unwrap();
        let (actor, actions) = match game.node(edge.child) {
            NodeInfo::Decision {
                player,
                num_actions,
            } => (player, num_actions),
            _ => panic!(),
        };
        let width = game.combo_count(edge.child, actor);
        let mut forced = vec![0.0; actions * width];
        forced[(actions - 1) * width..].fill(1.0);
        let changed = Overlay {
            base: &uniform,
            rows: HashMap::from([(edge.child, forced)]),
        };
        let q = json!({"history":["check","check"],"turn":"4h"});
        let changed_value = boundary_value(game, &teacher, &changed, actor, &q).unwrap();
        let offset = changed_value["turn_to_flop_offset_bb"].as_f64().unwrap();
        let q = json!({"history":["check","check"],"turn":"4h","override_turn_ev_bb":changed_value["conditional_reference_ev_bb"].as_f64().unwrap()-offset});
        let result = boundary_value(game, &teacher, &uniform, actor, &q).unwrap();
        let direct = (br::expected_value(game, actor, &changed)
            - br::expected_value(game, actor, &uniform)) as f64;
        assert!((direct - result["root_delta_bb"].as_f64().unwrap()).abs() < 2e-6);
        assert!(boundary_value(
            game,
            &teacher,
            &uniform,
            actor,
            &json!({"history":["check","check"],"turn":"2c"})
        )
        .is_err());
        assert!(boundary_value(
            game,
            &teacher,
            &uniform,
            actor,
            &json!({"history":["check"],"turn":"4h"})
        )
        .is_err());
    }
}

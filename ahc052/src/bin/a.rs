#![allow(unused_imports)]
use ac_library::*;
use itertools::*;
use proconio::{fastout, input, marker::Chars};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet, VecDeque};
use superslice::Ext;

use std::time::{Duration, Instant};

pub const N: usize = 30;

pub const CMD: [char; 5] = ['U', 'D', 'L', 'R', 'S'];
pub const CMD_U: usize = 0;
pub const CMD_D: usize = 1;
pub const CMD_L: usize = 2;
pub const CMD_R: usize = 3;
pub const CMD_S: usize = 4;

type KeyConfig = Vec<Vec<char>>;

#[derive(Clone)]
pub struct Info {
    start_time: Instant,
    time_limit: Duration,
    best_answer: (i64, Vec<Vec<char>>, Vec<usize>), // 最適解
}

impl Info {
    fn new() -> Self {
        Self {
            time_limit: Duration::from_millis(1800),
            start_time: Instant::now(),
            best_answer: (0, vec![vec!['S'; 10]; 10], vec![]),
        }
    }

    pub fn is_time_up(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    pub fn update_best_answer(&mut self, score: i64, key: &KeyConfig, operation: &Vec<usize>) -> bool {
        if score > self.best_answer.0 {
            self.best_answer = (score, key.clone(), operation.clone());
            true
        } else {
            false
        }
    }

    pub fn print_answer(&self) {
        for i in 0..10 {
            for j in 0..9 {
                print!("{} ", self.best_answer.1[i][j]);
            }
            for j in 9..10 {
                print!("{}", self.best_answer.1[i][j]);
            }
            println!()
        }
        for &i in self.best_answer.2.iter() {
            println!("{}", i);
        }
    }
}

pub struct Input {
    n: usize,                    // n=30
    m: usize,                    // m=10 robot
    k: usize,                    // k=10 button
    robots: Vec<(usize, usize)>, // 初期のロボット
    graph: Vec<Vec<HashSet<(usize, usize)>>>,
    dsu: Dsu,
    groups: HashMap<usize, Vec<usize>>, // 各連結成分に属するロボットのリスト
}

// #[derive(Clone)]
pub struct State {
    key_config: KeyConfig,
    robot_positions: Vec<(usize, usize)>, // ロボットの場所
    operation: Vec<usize>,                // 操作列
    yuka: HashSet<(usize, usize)>,        // 各床についてワックスを塗ったか
    yuka_counter: Vec<Vec<usize>>,        // 各床についてワックスを塗った回数
    score: i64,                           // スコア
    rng: rand::rngs::ThreadRng,
}

impl Input {
    fn from_stdin() -> Self {
        input! {
            n: usize,
            m: usize,
            k: usize,
            robots: [(usize,usize); m],
            v: [Chars; n],
            h: [Chars; n-1],
        }

        let mut graph = vec![vec![HashSet::new(); n]; n];

        let mut dsu = Dsu::new(n * n);

        iproduct!(0..n, 0..n).for_each(|(i, j)| {
            let u = i * n + j;

            if i > 0 && h[i - 1][j] == '0' {
                graph[i][j].insert((i - 1, j));

                let v = (i - 1) * n + j;
                dsu.merge(u, v);
            }
            if i + 1 < n && h[i][j] == '0' {
                graph[i][j].insert((i + 1, j));

                let v = (i + 1) * n + j;
                dsu.merge(u, v);
            }
            if j > 0 && v[i][j - 1] == '0' {
                graph[i][j].insert((i, j - 1));

                let v = i * n + j - 1;
                dsu.merge(u, v);
            }
            if j + 1 < n && v[i][j] == '0' {
                graph[i][j].insert((i, j + 1));

                let v = i * n + j + 1;
                dsu.merge(u, v);
            }
        });

        let groups = robots
            .iter()
            .enumerate()
            .map(|(rb, &(i, j))| (rb, i * n + j))
            .fold(HashMap::new(), |mut acc, (rb, u)| {
                let root = dsu.leader(u);
                acc.entry(root).or_insert(vec![]).push(rb);
                acc
            });

        Self { n, m, k, robots, graph, dsu, groups }
    }
}

impl State {
    pub fn new(input: &Input) -> Self {
        let mut yuka = HashSet::new();
        for (i, j) in input.robots.iter() {
            yuka.insert((*i, *j));
        }
        Self {
            key_config: vec![vec!['S'; 10]; 10],
            robot_positions: input.robots.clone(),
            operation: vec![],
            yuka,
            yuka_counter: vec![vec![0; N]; N],
            score: 0,
            rng: rand::thread_rng(),
        }
    }

    fn reset_sosa(&mut self, input: &Input) {
        self.robot_positions = input.robots.clone();
        self.operation = vec![];
        self.yuka = HashSet::new();
        for (i, j) in input.robots.iter() {
            self.yuka.insert((*i, *j));
        }
        self.score = 0;
        self.yuka_counter = vec![vec![0; N]; N];
    }

    pub fn operate(&mut self, key: usize, graph: &Vec<Vec<HashSet<(usize, usize)>>>) {
        for (robot, cmd) in self.key_config[key].iter().enumerate() {
            let (ni, nj) = self.robot_positions[robot];
            match cmd {
                'U' => {
                    let next = (ni.saturating_sub(1), nj);
                    if graph[ni][nj].contains(&next) {
                        self.yuka.insert(next);
                        self.robot_positions[robot].0 -= 1
                    }
                },
                'D' => {
                    let next = (ni.saturating_add(1), nj);
                    if graph[ni][nj].contains(&next) {
                        self.yuka.insert(next);
                        self.robot_positions[robot].0 += 1
                    }
                },
                'L' => {
                    let next = (ni, nj.saturating_sub(1));
                    if graph[ni][nj].contains(&next) {
                        self.yuka.insert(next);
                        self.robot_positions[robot].1 -= 1
                    }
                },
                'R' => {
                    let next = (ni, nj.saturating_add(1));
                    if graph[ni][nj].contains(&next) {
                        self.yuka.insert(next);
                        self.robot_positions[robot].1 += 1
                    }
                },
                'S' => {},
                _ => unreachable!(),
            }
        }

        self.operation.push(key);
    }

    pub fn update_score(&mut self) {
        let r = N * N - self.yuka.len();
        self.score = if r == 0 {
            (3 * N * N - self.operation.len()) as i64
        } else {
            self.yuka.len() as i64
        };
    }
}

fn get_cmd_i(u: (usize, usize), v: (usize, usize)) -> usize {
    if u.0 + 1 == v.0 && u.1 == v.1 {
        CMD_D
    } else if u.0 == v.0 && u.1 + 1 == v.1 {
        CMD_R
    } else if u.1 == v.1 && u.0 >= 1 && u.0 - 1 == v.0 {
        CMD_U
    } else if u.0 == v.0 && u.1 >= 1 && u.1 - 1 == v.1 {
        CMD_L
    } else if u == v {
        CMD_S
    } else {
        panic!("invalid edge");
    }
}

fn dfs(
    graph: &Vec<Vec<HashSet<(usize, usize)>>>,
    u: (usize, usize),
    visited: &mut HashSet<(usize, usize)>,
    history: &mut Vec<(usize, usize)>,
    parent: (usize, usize),
    yuka_counter: &Vec<Vec<usize>>,
) {
    history.push(u);
    visited.insert(u);

    for &v in graph[u.0][u.1]
        .iter()
        .sorted()
        //.sorted_by_key(|&(i, j)| std::cmp::Reverse(yuka_counter[*i][*j]))
        .sorted_by_key(|&(i, j)| yuka_counter[*i][*j])
    {
        if visited.contains(&v) {
            continue;
        }
        dfs(graph, v, visited, history, u, yuka_counter);
    }
    history.push(parent);
}

mod solver {

    use rand::seq::SliceRandom;
    use std::vec;

    use super::*;

    ///
    /// キー割り当てを生成する
    pub fn generate_key_config(input: &Input, info: &mut Info, state: &mut State) {
        for group in input.groups.values() {
            for (index, &robot) in group.iter().enumerate() {
                for i in 0..10 {
                    state.key_config[i][robot] = CMD[(i + index) % 5];
                }
            }
        }
    }

    //
    // 操作順序の決定
    //
    fn decide_order_to_operate(input: &Input, info: &mut Info, state: &mut State, order: &Vec<usize>) {
        for &i in order.iter() {
            state.operate(i, &input.graph);
            if state.yuka.len() == input.n * input.n {
                break;
            }
        }
    }

    fn decide_robot_order(input: &Input, info: &mut Info, state: &mut State, rb: usize, order_candidate: &mut Vec<Vec<usize>>) {
        order_candidate.push(vec![]);
        let order_candidate_tail = order_candidate.len() - 1;
        let mut order = vec![];
        {
            let mut visited = HashSet::new();
            // TODO: parent を直す
            dfs(
                &input.graph,
                input.robots[rb],
                &mut visited,
                &mut order,
                input.robots[rb],
                &state.yuka_counter,
            );
            order.pop(); // 最後の親への戻りを削除
        }

        let mut cmd_to_key = vec![usize::MAX; 5];
        let mut keys = vec!['S'; 10];

        let mut now = state.robot_positions[rb];

        for (t, &n_pos) in order.iter().enumerate() {
            state.yuka_counter[now.0][now.1] += 1;

            let cmd_i = get_cmd_i(now, n_pos);

            if cmd_to_key[cmd_i] == usize::MAX {
                let p = if order_candidate.len() > 1 {
                    let key = order_candidate[order_candidate_tail - 1][t];
                    if keys[key] == 'S' {
                        key
                    } else {
                        keys.iter().position(|&c| c == 'S').unwrap()
                    }
                } else {
                    keys.iter().position(|&c| c == 'S').unwrap()
                };

                cmd_to_key[cmd_i] = p;
                keys[p] = CMD[cmd_i];
            }
            order_candidate[order_candidate_tail].push(cmd_to_key[cmd_i]);

            now = n_pos;
        }
        // eprintln!("Robot {} order length: {}", rb, order_candidate[order_candidate_tail].len());
        for i in 0..10 {
            state.key_config[i][rb] = keys[i];
        }
    }

    fn decide_robots_order(input: &Input, info: &mut Info, state: &mut State, order: &Vec<usize>) {
        let mut order_candidate = vec![];

        for &i in order {
            decide_robot_order(input, info, state, i, &mut order_candidate);
        }

        // -------------------------------------------
        for order in order_candidate.iter() {
            if info.is_time_up() {
                break;
            }
            decide_order_to_operate(input, info, state, order);
            state.update_score();
            info.update_best_answer(state.score, &state.key_config, &state.operation);
            state.reset_sosa(input);
        }
    }

    pub fn solve(input: &Input, info: &mut Info) {
        // // キー割り当て作成
        // generate_key_config(input, info, state);

        // // 操作順序の決定
        // decide_order_to_operate(input, info, state);

        for _ in 0..50 {
            if info.is_time_up() {
                break;
            }
            let mut state = State::new(&input);

            let mut order: Vec<usize> = (0..10).collect();
            order.shuffle(&mut state.rng);
            decide_robots_order(input, info, &mut state, &order);
        }
    }

    pub fn find_best_answer(input: &Input, info: &mut Info) {
        solve(input, info);
    }
}

fn main() {
    let input = Input::from_stdin();
    let mut info = Info::new();

    solver::find_best_answer(&input, &mut info);
    info.print_answer();

    eprintln!("#time: {}", info.start_time.elapsed().as_millis());
    eprintln!("#score: {}", info.best_answer.0);
}

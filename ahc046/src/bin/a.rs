#![allow(unused_imports)]
use ac_library::*;
use itertools::*;
use proconio::{fastout, input, marker::Chars};
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::{Duration, Instant};
use std::{collections::VecDeque, usize};
use superslice::Ext;

const N: usize = 20;
const M: usize = 40;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Pos {
    x: usize,
    y: usize,
}

#[derive(Clone)]
struct Input {
    n: usize,       // n=20 (座標)
    m: usize,       // m=40 (目的地の個数)
    goal: Vec<Pos>, // 目的地の座標
}

impl Input {
    fn from_stdin() -> Self {
        input! {
            n: usize, // 都市の個数 n=800
            m: usize, // 都市のグループの個数 1 <= m <= 400
        }

        let mut goal = Vec::with_capacity(m);
        for _ in 0..m {
            input! {
                x: usize,
                y: usize,
            }
            goal.push(Pos { x, y });
        }

        Self { n, m, goal }
    }
}

#[derive(Clone, Eq, PartialEq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    fn into_char(&self) -> char {
        match self {
            Dir::Up => 'U',
            Dir::Down => 'D',
            Dir::Left => 'L',
            Dir::Right => 'R',
        }
    }
}
impl std::fmt::Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.into_char())
    }
}

#[derive(Clone)]
enum Action {
    Move(Dir),
    Slide(Dir),
    Apply(Dir),
}
impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Apply(dir) => write!(f, "A {}", dir),
            Action::Move(dir) => write!(f, "M {}", dir),
            Action::Slide(dir) => write!(f, "S {}", dir),
        }
    }
}

#[derive(Clone)]
struct Info {
    state: Vec<Vec<char>>,       // 座標の状態
    next: usize,                 // 次の目的地
    last_visited: usize,         // 最後に訪れた目的地
    now_pos: Pos,                // 現在の座標
    action_hisotry: Vec<Action>, // 行動履歴
    position_hisotry: Vec<Pos>,  // 座標履歴
    start_time: Instant,
    time_limit: Duration,
    best_answer: (usize, Vec<Action>), // 最適解
    candidate: FxHashMap<Pos, usize>,  // 壁の建設候補
    rng: rand::rngs::ThreadRng,
}

impl Info {
    fn new(input: &Input) -> Self {
        let rng = rand::prelude::ThreadRng::default();
        Self {
            state: vec![vec!['.'; input.n]; input.n],
            next: 1,
            last_visited: 0,
            now_pos: input.goal[0],
            action_hisotry: vec![],
            position_hisotry: vec![],
            time_limit: Duration::from_millis(1920),
            start_time: Instant::now(),
            best_answer: (0, vec![]),
            candidate: FxHashMap::default(),
            rng,
        }
    }

    fn is_time_up(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    fn update_best_answer(&mut self) {
        let new_score = calculate_score(self);
        eprintln!("new score: {}", new_score);
        if new_score > self.best_answer.0 {
            self.best_answer = (new_score, self.action_hisotry.clone());
        }
    }

    fn clear(&mut self, input: &Input) {
        self.state = vec![vec!['.'; self.state.len()]; self.state.len()];
        self.next = 1;
        self.last_visited = 0;
        self.now_pos = input.goal[0];
        self.action_hisotry.clear();
        self.position_hisotry.clear();
        // self.candidate.clear();
    }
}

/// 出力した行動列の長さを T、訪れることが出来た目的地の数をm としたとき、以下のスコアが得られる。
/// m<M−1 の場合、 m+1
/// m=M−1 の場合、M+2NM−T
fn calculate_score(info: &Info) -> usize {
    if info.last_visited < M - 1 {
        info.last_visited
    } else {
        M + 2 * N * M - info.action_hisotry.len()
    }
}

fn print_result(info: &Info) {
    for action in info.best_answer.1.iter() {
        println!("{}", action);
    }
}

/// 2点間のマンハッタン距離を計算する
fn calculate_distance(a: &Pos, b: &Pos) -> usize {
    a.x.abs_diff(b.x) + a.y.abs_diff(b.y)
}

/// 現在地から進む方向において最初にぶつかる壁を探す
fn find_latest_wall(dir: &Dir, info: &Info) -> Pos {
    match dir {
        Dir::Up => {
            for i in (0..info.now_pos.x).rev() {
                if info.state[i][info.now_pos.y] == '#' {
                    return Pos { x: i + 1, y: info.now_pos.y };
                }
            }
            Pos { x: 0, y: info.now_pos.y }
        },
        Dir::Down => {
            for i in info.now_pos.x + 1..info.state.len() {
                if info.state[i][info.now_pos.y] == '#' {
                    return Pos { x: i - 1, y: info.now_pos.y };
                }
            }
            Pos {
                x: info.state.len() - 1,
                y: info.now_pos.y,
            }
        },
        Dir::Left => {
            for i in (0..info.now_pos.y).rev() {
                if info.state[info.now_pos.x][i] == '#' {
                    return Pos { x: info.now_pos.x, y: i + 1 };
                }
            }
            Pos { x: info.now_pos.x, y: 0 }
        },
        Dir::Right => {
            for i in info.now_pos.y + 1..info.state[0].len() {
                if info.state[info.now_pos.x][i] == '#' {
                    return Pos { x: info.now_pos.x, y: i - 1 };
                }
            }
            Pos {
                x: info.now_pos.x,
                y: info.state[0].len() - 1,
            }
        },
    }
}

fn select_building_wall(input: &Input, info: &mut Info, action: Action, now: Pos) -> Option<Action> {
    let dir = match action {
        Action::Move(dir) => dir,
        Action::Slide(dir) => dir,
        _ => unreachable!(),
    };

    let neighboring_positions = [
        (Dir::Up, Pos { x: now.x.saturating_sub(1), y: now.y }),
        (
            Dir::Down,
            Pos {
                x: (now.x + 1).min(input.n - 1),
                y: now.y,
            },
        ),
        (Dir::Left, Pos { x: now.x, y: now.y.saturating_sub(1) }),
        (
            Dir::Right,
            Pos {
                x: now.x,
                y: (now.y + 1).min(input.n - 1),
            },
        ),
    ]
    .iter()
    .filter(|&(_, pos)| pos != &now)
    .filter(|&(n_dir, _)| *n_dir != dir)
    .map(|(_, pos)| *pos)
    .collect::<Vec<_>>();

    // 壁の建設候補地が上下左右にある場合は、壁を建設する
    let rnd_val = info.rng.gen_range(0..100);
    for pos in neighboring_positions {
        if let Some(goal) = info.candidate.get(&pos) {
            if info.state[pos.x][pos.y] == '.' && goal > &info.last_visited && rnd_val < 25 {
                return Some(Action::Apply(dir.clone()));
            }
        }
    }
    None
}

/// 最適な行動を選択する
fn select_action(input: &Input, info: &mut Info, next_goal: &Pos) -> Option<Action> {
    let now = info.now_pos;

    // 現在地から次の目的地までのマンハッタン距離
    let current_dist = calculate_distance(&now, &next_goal);

    // 各方向への移動後の距離を計算
    let mut best_action = None;
    let mut min_dist = current_dist;

    // ------------------------------------------------------------------------------------------------
    // Move actions
    // ------------------------------------------------------------------------------------------------
    let moves = [
        (Dir::Up, Pos { x: now.x.saturating_sub(1), y: now.y }),
        (
            Dir::Down,
            Pos {
                x: (now.x + 1).min(input.n - 1),
                y: now.y,
            },
        ),
        (Dir::Left, Pos { x: now.x, y: now.y.saturating_sub(1) }),
        (
            Dir::Right,
            Pos {
                x: now.x,
                y: (now.y + 1).min(input.n - 1),
            },
        ),
    ];

    for (dir, pos) in moves.iter() {
        if *pos == now {
            continue;
        }

        // 壁かつ目的地の場合は壁を壊す
        if info.state[pos.x][pos.y] == '#' && *pos == *next_goal {
            return Some(Action::Apply(dir.clone()));
        }

        // TODO: 壁にぶつかる場合の最適な行動を考える
        // 壁かつ目的地ではない場合は壁を避けたいがとりあえず壊す
        if info.state[pos.x][pos.y] == '#' {
            return Some(Action::Apply(dir.clone()));
        }

        let dist = calculate_distance(&pos, &next_goal);

        if dist < min_dist {
            min_dist = dist;
            best_action = Some(Action::Move(dir.clone()));
        }
    }

    // ------------------------------------------------------------------------------------------------
    // Slide actions
    // ------------------------------------------------------------------------------------------------
    let slides = [
        (Dir::Up, find_latest_wall(&Dir::Up, info)),
        (Dir::Down, find_latest_wall(&Dir::Down, info)),
        (Dir::Left, find_latest_wall(&Dir::Left, info)),
        (Dir::Right, find_latest_wall(&Dir::Right, info)),
    ];

    for (dir, pos) in slides.iter() {
        if *pos == now {
            continue;
        } // Skip if already at edge

        let dist = calculate_distance(&pos, &next_goal);

        if dist < min_dist {
            min_dist = dist;
            best_action = Some(Action::Slide(dir.clone()));
        }
    }

    if let Some(action) = &best_action {
        // 壁の建設候補地が上下左右にある場合は、壁を建設する
        if let Some(build_wall_action) = select_building_wall(input, info, action.clone(), now) {
            return Some(build_wall_action);
        }
    }

    best_action
}

/// 以下の2つの方法の内，次の目的地との距離が近くなるのはどれかを判定し実行する
/// 1. Move: 上下左右どれかに1マスだけ移動する
/// 2. Slide: 上下左右どれかに可能な限り移動する (現在の座標 (x,y) とすると (0,y) or (x,0) or (n,y) or (x,n) となるように移動する)
fn do_best_action(input: &Input, info: &mut Info, next_goal: &Pos) {
    if let Some(action) = select_action(input, info, next_goal) {
        match &action {
            Action::Move(dir) => match dir {
                Dir::Up => info.now_pos.x -= 1,
                Dir::Down => info.now_pos.x += 1,
                Dir::Left => info.now_pos.y -= 1,
                Dir::Right => info.now_pos.y += 1,
            },
            Action::Slide(dir) => match dir {
                Dir::Up => info.now_pos.x = 0,
                Dir::Down => info.now_pos.x = input.n - 1,
                Dir::Left => info.now_pos.y = 0,
                Dir::Right => info.now_pos.y = input.n - 1,
            },
            Action::Apply(dir) => {
                let pos = match dir {
                    Dir::Up => Pos { x: info.now_pos.x - 1, y: info.now_pos.y },
                    Dir::Down => Pos { x: info.now_pos.x + 1, y: info.now_pos.y },
                    Dir::Left => Pos { x: info.now_pos.x, y: info.now_pos.y - 1 },
                    Dir::Right => Pos { x: info.now_pos.x, y: info.now_pos.y + 1 },
                };
                if info.state[pos.x][pos.y] == '#' {
                    info.state[pos.x][pos.y] = '.';
                } else {
                    info.state[pos.x][pos.y] = '#';
                }
            },
        }

        // eprintln!("Now: {} {}", info.now_pos.x, info.now_pos.y);
        info.position_hisotry.push(info.now_pos);
        info.action_hisotry.push(action);
    }
}

fn create_answer(input: &Input, info: &mut Info) {
    while info.action_hisotry.len() < 2 * input.n * input.m && info.next < input.m {
        let next_goal = input.goal[info.next];
        do_best_action(input, info, &next_goal);

        // Check if we reached the next goal
        if info.now_pos == next_goal {
            info.last_visited = info.next;
            info.next += 1;
        }
    }
}

fn create_candidate_to_build_wall(input: &Input, info: &mut Info) {
    let mut last_pos = info.now_pos;
    let mut pre_dir = None;
    let mut p_m = input.goal.len() - 1;

    for i in (0..info.action_hisotry.len()).rev() {
        let action = &info.action_hisotry[i];
        // 同じ方向への Move Actionが続く場合は last_pos の隣を壁の建設候補とする
        if let Action::Move(dir) = action {
            match pre_dir {
                None => {
                    last_pos = info.position_hisotry[i];
                    pre_dir = Some(dir);
                },
                Some(pre_dir_val) => {
                    let wall_pos = match pre_dir_val {
                        Dir::Up => {
                            if last_pos.x == 0 {
                                None
                            } else {
                                Some(Pos { x: last_pos.x - 1, y: last_pos.y })
                            }
                        },
                        Dir::Down => {
                            if last_pos.x == input.n - 1 {
                                None
                            } else {
                                Some(Pos { x: last_pos.x + 1, y: last_pos.y })
                            }
                        },
                        Dir::Left => {
                            if last_pos.y == 0 {
                                None
                            } else {
                                Some(Pos { x: last_pos.x, y: last_pos.y - 1 })
                            }
                        },
                        Dir::Right => {
                            if last_pos.y == input.n - 1 {
                                None
                            } else {
                                Some(Pos { x: last_pos.x, y: last_pos.y + 1 })
                            }
                        },
                    };
                    if let Some(wall_pos) = wall_pos {
                        // 壁の建設候補を追加
                        if !info.candidate.contains_key(&wall_pos) {
                            info.candidate.insert(wall_pos, p_m);
                        }
                    }

                    //
                    last_pos = info.position_hisotry[i];
                    pre_dir = None;
                },
            }
        } else {
            if let Some(dir) = pre_dir {
                let wall_pos = match dir {
                    Dir::Up => {
                        if last_pos.x == 0 {
                            None
                        } else {
                            Some(Pos { x: last_pos.x - 1, y: last_pos.y })
                        }
                    },
                    Dir::Down => {
                        if last_pos.x == input.n - 1 {
                            None
                        } else {
                            Some(Pos { x: last_pos.x + 1, y: last_pos.y })
                        }
                    },
                    Dir::Left => {
                        if last_pos.y == 0 {
                            None
                        } else {
                            Some(Pos { x: last_pos.x, y: last_pos.y - 1 })
                        }
                    },
                    Dir::Right => {
                        if last_pos.y == input.n - 1 {
                            None
                        } else {
                            Some(Pos { x: last_pos.x, y: last_pos.y + 1 })
                        }
                    },
                };
                if let Some(wall_pos) = wall_pos {
                    // 壁の建設候補を追加
                    if !info.candidate.contains_key(&wall_pos) {
                        info.candidate.insert(wall_pos, p_m);
                    }
                }
            }
            pre_dir = None;
        }
        if info.position_hisotry[i] == input.goal[p_m.saturating_sub(1)] {
            p_m = p_m.saturating_sub(1);
        }
    }
}

fn solve(input: &Input, info: &mut Info) {
    // 初期解作成
    create_answer(input, info);

    // 壁の建設候補を作成
    create_candidate_to_build_wall(input, info);

    while !info.is_time_up() {
        create_answer(input, info);
        info.update_best_answer();
        info.clear(input);
    }

    print_result(info);
    eprintln!("best score: {}", info.best_answer.0);
}

#[fastout]
fn main() {
    let input = Input::from_stdin();
    let mut info = Info::new(&input);
    solve(&input, &mut info);
}

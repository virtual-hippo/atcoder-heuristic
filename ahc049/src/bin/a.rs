use proconio::input;
use rand::Rng;
use std::time::{Duration, Instant};

pub const N: usize = 20;
pub const MAX: usize = 2 * N * N * N;
pub const GATE: Pos = Pos { x: 0, y: 0 };

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Pos {
    x: usize,
    y: usize,
}

#[derive(Clone, Eq, PartialEq)]
pub enum Dir {
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
struct Info {
    start_time: Instant,
    time_limit: Duration,
    best_answer: (usize, Vec<char>), // 最適解
}

impl Info {
    fn new() -> Self {
        Self {
            time_limit: Duration::from_millis(1920),
            start_time: Instant::now(),
            best_answer: (0, vec![]),
        }
    }

    fn is_time_up(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    fn update_best_answer(&mut self, state: &State) {
        let new_score = calculate_score(state);
        if new_score > self.best_answer.0 {
            self.best_answer = (new_score, state.history.clone());
        }
    }
}

#[derive(Clone)]
pub struct Input {
    n: usize,         // n=20 (座標)
    w: Vec<Vec<i64>>, // 重さ
    d: Vec<Vec<i64>>, // 耐久力
}

impl Input {
    fn from_stdin() -> Self {
        input! {
            n: usize,
            w: [[i64; n]; n],
            d: [[i64; n]; n],
        }

        Self { n, w, d }
    }
}

/// (w,d)
#[derive(Clone)]
pub struct Cardboard(i64, i64);

#[derive(Clone)]
pub struct State {
    t: usize,
    office: Vec<Vec<Option<Cardboard>>>,
    // 現在持っている cardboards
    takahashi_cardboards: Vec<Cardboard>,
    pos: Pos,
    history: Vec<char>,
    rng: rand::rngs::ThreadRng,
}

impl State {
    fn new(Input { n, w, d }: &Input) -> Self {
        let mut office = vec![vec![]; *n];
        for i in 0..*n {
            for j in 0..*n {
                office[i].push(Some(Cardboard(w[i][j], d[i][j])));
            }
        }
        office[0][0] = None; // (0,0) は空き

        let takahashi_cardboards = vec![];

        let rng = rand::prelude::ThreadRng::default();

        Self {
            t: 0,
            office,
            takahashi_cardboards,
            pos: GATE,
            history: vec![],
            rng,
        }
    }
}

pub mod helper {
    use super::*;

    pub fn lift(state: &mut State) {
        if let Some(cardboard) = state.office[state.pos.x][state.pos.y].take() {
            state.takahashi_cardboards.push(cardboard);
            state.history.push('1');
        }
    }

    pub fn put(state: &mut State) {
        if let Some(cardboard) = state.takahashi_cardboards.pop() {
            state.office[state.pos.x][state.pos.y] = Some(cardboard);
            state.history.push('2');
        }
    }

    pub fn shift(state: &mut State, dir: Dir) {
        let (dx, dy) = match dir {
            Dir::Up => (-1, 0),
            Dir::Down => (1, 0),
            Dir::Left => (0, -1),
            Dir::Right => (0, 1),
        };
        let new_x = state.pos.x as isize + dx;
        let new_y = state.pos.y as isize + dy;

        if new_x >= 0 && new_x < state.office.len() as isize && new_y >= 0 && new_y < state.office[0].len() as isize {
            state.pos.x = new_x as usize;
            state.pos.y = new_y as usize;
            state.history.push(dir.into_char());
            state.t += 1;
        }
    }
}

fn serach_cardboard(state: &mut State) -> Option<Pos> {
    for x in (0..N).rev() {
        for y in (0..N).rev() {
            if state.office[x][y].is_some() {
                return Some(Pos { x, y });
            }
        }
    }
    None
}

fn move_x(state: &mut State, goal: Pos) -> bool {
    let dx = goal.x as isize - state.pos.x as isize;
    if dx > 0 {
        helper::shift(state, Dir::Down);
        true
    } else if dx < 0 {
        helper::shift(state, Dir::Up);
        true
    } else {
        false
    }
}

fn move_y(state: &mut State, goal: Pos) -> bool {
    let dy = goal.y as isize - state.pos.y as isize;
    if dy > 0 {
        helper::shift(state, Dir::Right);
        true
    } else if dy < 0 {
        helper::shift(state, Dir::Left);
        true
    } else {
        false
    }
}

fn move_x_or_y(goal: Pos, state: &mut State) {
    let rnd_val = state.rng.gen_range(0..2);

    if rnd_val == 0 {
        if !move_x(state, goal) {
            move_y(state, goal);
        }
    } else {
        if !move_y(state, goal) {
            move_x(state, goal);
        }
    }
}

/// state.pos と GATE のマンハッタン距離を求める
fn manhattan_distance(state: &State) -> usize {
    let dx = (state.pos.x as isize - GATE.x as isize).abs() as usize;
    let dy = (state.pos.y as isize - GATE.y as isize).abs() as usize;
    dx + dy
}

fn lift_cargeboard(state: &mut State) {
    // 段ボールがある場合のみチェック
    if let Some(cardboard) = &state.office[state.pos.x][state.pos.y] {
        // 持っている段ボール数が指定の範囲の場合のみ比較
        if 0 < state.takahashi_cardboards.len() && state.takahashi_cardboards.len() < 8 {
            let dist = manhattan_distance(state) as i64;
            let cardboard_weight = cardboard.0;

            let can_lift = state
                .takahashi_cardboards
                .iter()
                .map(|Cardboard(_, d)| *d)
                .all(|d| d > cardboard_weight * dist);

            // 持っている段ボールの耐久力が十分なら持ち上げる
            if can_lift {
                helper::lift(state);
                for i in 0..state.takahashi_cardboards.len() - 1 {
                    state.takahashi_cardboards[i].1 -= cardboard_weight * dist;
                }
            }
        }
    }
}

fn solve(state: &mut State) {
    while let Some(pos) = serach_cardboard(state) {
        // 目的地までの移動
        while state.pos != pos {
            move_x_or_y(pos, state);
        }

        // 段ボールを持ち上げる
        helper::lift(state);

        // 出入り口に戻る
        while state.pos != GATE {
            move_x_or_y(GATE, state);
            lift_cargeboard(state);
        }

        state.takahashi_cardboards = vec![];
    }
}

fn calculate_score(state: &State) -> usize {
    N * N + 2 * N * N * N - state.t
}

fn main() {
    let mut info = Info::new();
    let input = Input::from_stdin();

    while !info.is_time_up() {
        let mut state = State::new(&input);
        solve(&mut state);
        info.update_best_answer(&state);
    }

    for i in 0..info.best_answer.1.len() {
        println!("{}", info.best_answer.1[i]);
    }
    // eprintln!("score: {}", info.best_answer.0);
}

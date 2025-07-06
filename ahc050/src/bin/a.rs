// use itertools::*;
use proconio::{input, marker::Chars};
use rand::Rng;
use std::time::{Duration, Instant};

pub const N: usize = 40;

#[derive(Clone)]
struct Info {
    start_time: Instant,
    time_limit: Duration,
    best_answer: (i64, Vec<(usize, usize)>), // 最適解
}

impl Info {
    fn new() -> Self {
        Self {
            time_limit: Duration::from_millis(1950),
            start_time: Instant::now(),
            best_answer: (0, vec![]),
        }
    }

    fn is_time_up(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    fn update_best_answer(&mut self, score: i64, answer: Vec<(usize, usize)>) -> bool {
        if score > self.best_answer.0 {
            self.best_answer = (score, answer);
            true
        } else {
            false
        }
    }

    fn print_answer(&self) {
        for &(i, j) in &self.best_answer.1 {
            println!("{} {}", i, j);
        }
    }
}

#[derive(Clone)]
pub struct Input {
    _n: usize,         // n=40
    m: usize,          //
    s: Vec<Vec<char>>, // 初期盤面
}

impl Input {
    fn from_stdin() -> Self {
        input! {
            n: usize,
            m: usize,
            s: [Chars; n],
        }

        Self { _n: n, m, s }
    }
}

#[derive(Clone)]
pub struct FieldState {
    pub s: Vec<Vec<char>>,
    // 岩が存在する確率
    pub prob: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct State {
    m: usize, // ブロック数
    field: FieldState,
    // ターンごとの生存確率
    lives: Vec<f64>,
    answer: Vec<(usize, usize)>,
    score: i64, // スコア
    rng: rand::rngs::ThreadRng,
}

impl State {
    fn new(input: &Input) -> Self {
        Self {
            m: input.m,
            field: Self::build_field(input),
            lives: vec![1.0],
            answer: vec![],
            score: 0,
            rng: rand::thread_rng(),
        }
    }

    fn reset(&mut self, input: &Input) {
        self.field = Self::build_field(input);
        self.lives = vec![1.0];
        self.answer = vec![];
        self.score = 0;
    }

    fn reset_prob(&mut self) {
        if self.lives.len() > 1 {
            panic!("reset_prob should be called only at the beginning of the operation");
        }
        let initial_prob = 1.0 / (N * N - self.m) as f64;
        for i in 0..N {
            for j in 0..N {
                if self.field.s[i][j] == '.' {
                    self.field.prob[i][j] = initial_prob;
                }
            }
        }
    }

    fn build_field(input: &Input) -> FieldState {
        let mut prob = vec![vec![0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                if input.s[i][j] == '.' {
                    prob[i][j] = 1.0 / (N * N - input.m) as f64;
                }
            }
        }
        FieldState { s: input.s.clone(), prob }
    }

    fn _print_answer(&self) {
        for &(i, j) in &self.answer {
            println!("{} {}", i, j);
        }
    }

    fn compute_score(&self) -> i64 {
        let ub = (N * N - self.m - 1) as f64;
        let ret = self.lives.iter().sum::<f64>() - 1.0; // 初期の1.0 を減じる
        (ret / ub * 1e6).round() as i64
    }

    /**
     * ロボットが存在する確率を更新する関数
     */
    fn update_prob(&mut self) {
        const DIRS: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let mut next = vec![vec![0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                if self.field.prob[i][j] == 0.0 {
                    continue;
                }
                for &(di, dj) in &DIRS {
                    let mut i2 = i as i64;
                    let mut j2 = j as i64;
                    let n_i64 = N as i64;
                    loop {
                        let next_i = i2 + di;
                        let next_j = j2 + dj;

                        if next_i < 0 || next_i >= n_i64 || next_j < 0 || next_j >= n_i64 {
                            break;
                        }

                        if self.field.s[next_i as usize][next_j as usize] != '.' {
                            break;
                        }

                        i2 = next_i;
                        j2 = next_j;
                    }
                    next[i2 as usize][j2 as usize] += self.field.prob[i][j] * 0.25;
                }
            }
        }
        self.field.prob = next;
    }
}

fn update_each_turn(state: &mut State, pos: (usize, usize)) {
    state.update_prob();

    let (bi, bj) = pos;
    if state.field.s[bi][bj] == '#' {
        panic!("({}, {}) is already blocked (turn {})", bi, bj, state.lives.len() - 1);
    }

    let life = state.lives[state.lives.len() - 1] - state.field.prob[bi][bj];
    state.lives.push(life);
    state.field.prob[bi][bj] = 0.0;
    state.field.s[bi][bj] = '#';

    // -----------------------------------------------------------------------------
    // スコア, 解の更新
    // -----------------------------------------------------------------------------
    state.score = state.compute_score();
    state.answer.push((bi, bj));
}

/**
 * 現在のstate.field.probで最小値を持つ位置を探す
 */
fn find_min_prob_position(state: &State, random_flag: bool) -> Vec<(usize, usize)> {
    let mut min_prob = i64::MAX;
    let mut second_min_prob = i64::MAX;
    let mut min_positions = vec![];
    let mut second_min_positions = vec![];

    for i in 0..N {
        for j in 0..N {
            if state.field.s[i][j] == '.' {
                let prob = (state.field.prob[i][j] * 1e8).round() as i64; // 1e8倍して整数に変換
                if prob < min_prob {
                    second_min_prob = min_prob;
                    second_min_positions = min_positions.clone();
                    min_prob = prob;
                    min_positions = vec![];
                    min_positions.push((i, j));
                } else if prob == min_prob {
                    min_positions.push((i, j));
                } else if prob < second_min_prob {
                    second_min_prob = prob;
                    second_min_positions = vec![];
                    second_min_positions.push((i, j));
                } else if prob == second_min_prob {
                    second_min_positions.push((i, j));
                }
            }
        }
    }

    if random_flag && !second_min_positions.is_empty() {
        second_min_positions
    } else {
        min_positions
    }
}

/**
 * 確率の低い位置を貪欲に選んで解を構築する関数
 */
fn greedy_solve(input: &Input, state: &mut State, random_flag: bool) {
    let total_turns = N * N - input.m;

    // 最初の候補を探す
    state.update_prob();
    let koho = find_min_prob_position(state, false);
    // 状態を戻す
    state.reset_prob();

    let mut next = koho[state.rng.gen_range(0..koho.len())];

    for t in 0..total_turns {
        update_each_turn(state, next);

        let random_flag = t > 100 && random_flag && state.rng.gen_bool(0.4);

        let koho = find_min_prob_position(state, random_flag);
        if koho.is_empty() {
            break;
        }
        next = koho[state.rng.gen_range(0..koho.len())];
    }
}

fn greedy_solve2(input: &Input, state: &mut State, random_flag: bool) {
    let total_turns = N * N - input.m;
    let start = state.answer.len();

    let koho = find_min_prob_position(state, false);

    let mut next = koho[state.rng.gen_range(0..koho.len())];

    for t in start..total_turns {
        update_each_turn(state, next);

        let random_flag = t > 100 && random_flag && state.rng.gen_bool(0.5);

        let koho = find_min_prob_position(state, random_flag);
        if koho.is_empty() {
            break;
        }
        next = koho[state.rng.gen_range(0..koho.len())];
    }
}

fn build_initial_answer(input: &Input, info: &mut Info, state: &mut State) {
    let mut iteration = 0;

    while iteration < 100 && !info.is_time_up() {
        greedy_solve(input, state, iteration > 10 && iteration % 2 == 0);

        info.update_best_answer(state.score, state.answer.clone());

        state.reset(input);
        iteration += 1;
    }

    let mut state2 = State::new(input);
    for i in 0..800 {
        update_each_turn(&mut state2, info.best_answer.1[i]);
    }
    let stored_state = state2.clone();
    *state = state2;

    while iteration < 60 && !info.is_time_up() {
        greedy_solve2(input, state, iteration > 10 && iteration % 2 == 0);

        info.update_best_answer(state.score, state.answer.clone());

        *state = stored_state.clone();
        iteration += 1;
    }
    eprintln!("Total iterations: {} in build_initial_answer", iteration);
}

fn evaluate(input: &Input, order: &Vec<(usize, usize)>) -> i64 {
    let mut state = State::new(input);
    for &pos in order {
        update_each_turn(&mut state, pos);
    }
    state.score
}

fn solve(input: &Input, info: &mut Info, state: &mut State) {
    // 初期解を構築
    build_initial_answer(input, info, state);

    let mut p = info.best_answer.1.clone();

    // 初期解の評価
    let mut current_score = info.best_answer.0;

    // 温度パラメータ
    let start_temp = 3000.0;
    let end_temp = 10.0;

    let mut iter_count = 0;

    while !info.is_time_up() {
        iter_count += 1;

        // 温度の計算
        let progress = info.start_time.elapsed().as_millis() as f64 / info.time_limit.as_millis() as f64;
        let temp = start_temp * ((end_temp / start_temp) as f64).powf(progress);

        // 近傍解の生成（2つの要素をswap）
        let len = p.len();
        let i = state.rng.gen_range(0..len - 400);
        let j = state.rng.gen_range(0..len - 400);

        if i == j {
            continue;
        }

        // swap
        p.swap(i, j);

        // 新しい解の評価
        let new_score = evaluate(input, &p);

        if info.is_time_up() {
            break;
        }

        // 受理判定
        let delta = new_score - current_score;
        if delta > 0 || state.rng.gen::<f64>() < (delta as f64 / temp).exp() {
            // 受理
            current_score = new_score;

            if new_score > info.best_answer.0 {
                info.update_best_answer(current_score, p.clone());
                eprintln!(
                    "Update best score: {} (iter: {}, time: {:.3}s)",
                    info.best_answer.0,
                    iter_count,
                    info.start_time.elapsed().as_secs_f64()
                );
            }
        } else {
            // 棄却（元に戻す）
            p.swap(i, j);
        }
    }
    eprintln!("Total iterations: {}", iter_count);
    eprintln!("Final best score: {}", info.best_answer.0);
}

fn main() {
    let input = Input::from_stdin();
    let mut info = Info::new();
    let mut state = State::new(&input);

    solve(&input, &mut info, &mut state);
    info.print_answer();
    eprintln!("Score = {}", info.best_answer.0);
}

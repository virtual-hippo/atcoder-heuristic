use ac_library::*;
use itertools::*;
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
            time_limit: Duration::from_millis(1960),
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
}

#[derive(Clone)]
pub struct Input {
    n: usize,          // n=40
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

        Self { n, m, s }
    }
}

#[derive(Clone)]
pub struct FieldState {
    pub s: Vec<Vec<char>>,
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

    fn print_answer(&self) {
        for &(i, j) in &self.answer {
            println!("{} {}", i, j);
        }
    }

    fn compute_score(&self) -> i64 {
        let ub = (N * N - self.m - 1) as f64;
        let ret = self.lives.iter().sum::<f64>() - 1.0; // 初期の1.0 を減じる
        (ret / ub * 1e6).round() as i64
    }
}

fn update_each_turn(input: &Input, state: &mut State, pos: (usize, usize)) {
    // -----------------------------------------------------------------------------
    // 盤面状態の更新
    // -----------------------------------------------------------------------------
    let mut next = vec![vec![0.0; N]; N];
    let mut life = state.lives[state.lives.len() - 1];
    for i in 0..N {
        for j in 0..N {
            if state.field.prob[i][j] == 0.0 {
                continue;
            }
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let mut i2 = i as i64;
                let mut j2 = j as i64;
                let n_i64 = N as i64;
                while 0 <= i2 + di
                    && i2 + di < n_i64
                    && 0 <= j2 + dj
                    && j2 + dj < n_i64
                    && state.field.s[(i2 + di) as usize][(j2 + dj) as usize] == '.'
                {
                    i2 += di;
                    j2 += dj;
                }
                next[i2 as usize][j2 as usize] += state.field.prob[i][j] * 0.25;
            }
        }
    }
    state.field.prob = next;
    let (bi, bj) = pos;
    if state.field.s[bi][bj] == '#' {
        panic!("({}, {}) is already blocked (turn {})", bi, bj, state.lives.len() - 1);
    }
    life -= state.field.prob[bi][bj];
    state.lives.push(life);
    state.field.prob[bi][bj] = 0.0;
    state.field.s[bi][bj] = '#';

    // -----------------------------------------------------------------------------
    // スコア, 解の更新
    // -----------------------------------------------------------------------------
    state.score = state.compute_score();
    state.answer.push((bi, bj));
}

fn solve(input: &Input, info: &mut Info, state: &mut State) {
    let p = iproduct!(0..N, 0..N).filter(|&(i, j)| input.s[i][j] != '#').collect::<Vec<_>>();

    for &pos in p.iter() {
        update_each_turn(input, state, pos);
    }
}

fn main() {
    let input = Input::from_stdin();
    let mut info = Info::new();
    let mut state = State::new(&input);

    solve(&input, &mut info, &mut state);
    state.print_answer();
    eprintln!("Score = {}", state.score);
}

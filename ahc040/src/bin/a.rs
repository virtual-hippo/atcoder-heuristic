#![allow(unused_imports)]
use proconio::source::line::LineSource;
use proconio::{fastout, input, marker::Chars};
use rand::Rng;
use rand_distr::{num_traits::float, Distribution, Normal};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    io::{self, BufReader},
    time::{Duration, Instant},
};

const U: char = 'U';
const L: char = 'L';
const ROTATE: u8 = 1;
const NOT_ROTATE: u8 = 0;

#[derive(Debug, Eq, PartialEq)]
pub struct Prdb {
    pub p: usize,
    pub r: u8,
    pub d: char,
    pub b: i64,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Square {
    /// 番号
    pub i: usize,

    pub w: i64,
    pub h: i64,
}

impl Ord for Square {
    // `w` + `h` の合計 (面積で降順にソートできるようにする)
    fn cmp(&self, other: &Self) -> Ordering {
        (other.w + other.h)
            .cmp(&(self.w + self.h))
            .then_with(|| self.i.cmp(&other.i))
    }
}

impl PartialOrd for Square {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct State {
    pub prdbs: Vec<Prdb>,
    pub prdbs_p: Vec<usize>,
    pub score: i64,
    pub right: i64,
    pub bottom: i64,
    pub x_tree: ahc040_lib::SegmentTree,
    pub y_tree: ahc040_lib::SegmentTree,
    pub right_each_square: Vec<i64>,
    pub bottom_each_square: Vec<i64>,
}

pub const MAX_RANGE: i64 = 10_000_000_000;

#[fastout]
fn main() {
    // 実行時間上限
    let time_limit = Duration::from_millis(2300);
    let start_time = Instant::now();

    let mut stdin = LineSource::new(BufReader::new(io::stdin()));
    macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));

    // 長方形の個数 N は30≤N≤100 を満たす。
    // 操作回数T はN/2≤T≤4N を満たす。
    // 計測時に発生する誤差の標準偏差
    // σ は1000≤σ≤10000 を満たす整数値である。
    // 横幅と縦幅の計測値wi′​,hi′​ は 1 以上 109 以下の整数値である。

    input! {
        n: usize,
        t: usize,
        sigma: i64,
        mut wh: [(i64,i64); n],
    }

    for i in 0..n {
        wh[i].0 += sigma;
        wh[i].1 += sigma;
    }

    // とりあえず 1 度出力する
    {
        let state = optimize(n, &wh, start_time, time_limit, sigma);
        query(&state.prdbs);
    }
    let t = t - 1;

    // 実測値を求める
    let mut set = HashSet::new();
    let cnt = if n < t { n } else { t - 1 };
    {
        let mut squares = wh
            .iter()
            .enumerate()
            .map(|(i, (w, h))| Square { i, w: *w, h: *h })
            .collect::<Vec<Square>>();
        squares.sort();

        for i in 0..cnt {
            let prdbs = vec![Prdb {
                p: squares[i].i,
                r: 0,
                d: U,
                b: -1,
            }];
            wh[squares[i].i] = query(&prdbs);
            set.insert(squares[i].i);
        }
    }
    let t = t - cnt;

    // 実測値で上書きする
    let wh = {
        let mut ret = vec![];
        for i in 0..n {
            let (w, h) = wh[i];
            if set.contains(&i) {
                ret.push((w, h));
                continue;
            }
            ret.push((w, h));
        }
        ret
    };

    //let mut rng: rand::prelude::ThreadRng = rand::thread_rng();

    for _ in 0..t {
        let state = optimize(n, &wh, start_time, time_limit, sigma);
        query(&state.prdbs);
    }
}

fn optimize(
    n: usize,
    wh: &Vec<(i64, i64)>,
    start_time: Instant,
    time_limit: Duration,
    sigma: i64,
) -> State {
    let squares = wh
        .iter()
        .enumerate()
        .map(|(i, (w, h))| Square { i, w: *w, h: *h })
        .collect::<Vec<Square>>();

    let sum_w_and_h: i64 = wh.iter().map(|(w, h)| *w + *h).sum();

    let mut state = State {
        prdbs: vec![],
        prdbs_p: vec![],
        right: 0,
        bottom: 0,
        x_tree: ahc040_lib::SegmentTree::new(),
        y_tree: ahc040_lib::SegmentTree::new(),
        score: sum_w_and_h,
        right_each_square: vec![0; n],
        bottom_each_square: vec![0; n],
    };

    // 初期構築
    for p in 0..n {
        if start_time.elapsed() >= time_limit {
            break;
        }
        if let Some((prdb, best_score)) = jikken::execute(&mut state, &squares[p], p) {
            let (w, h) = if prdb.r == NOT_ROTATE {
                (squares[p].w, squares[p].h)
            } else {
                (squares[p].h, squares[p].w)
            };

            match prdb.d {
                U => {
                    let right = if prdb.b == -1 {
                        0
                    } else {
                        state.right_each_square[prdb.b as usize]
                    };

                    let (l, r) = (right, right + w);

                    let now_bottom = state.x_tree.query(0, MAX_RANGE, l, r - 1);
                    let updated_bottom = now_bottom + h as i64;
                    state
                        .x_tree
                        .update(0, MAX_RANGE, l, r - 1, now_bottom + h as i64);

                    state
                        .y_tree
                        .update(0, MAX_RANGE, state.bottom, updated_bottom - 1, r);
                    state.y_tree.update(
                        0,
                        MAX_RANGE,
                        updated_bottom - sigma,
                        updated_bottom - 1,
                        r,
                    );

                    state.bottom = state.bottom.max(updated_bottom);
                    state.right = state.right.max(r as i64);
                    state.right_each_square[p] = r;
                    state.bottom_each_square[p] = updated_bottom;
                }
                L => {
                    let bottom = if prdb.b == -1 {
                        0
                    } else {
                        state.bottom_each_square[prdb.b as usize]
                    };

                    let (l, r) = (bottom, bottom + h);

                    let now_right = state.y_tree.query(0, MAX_RANGE, l, r - 1);
                    // 更新対象区間についての更新後の右端
                    let updated_right = now_right + w as i64;
                    state
                        .y_tree
                        .update(0, MAX_RANGE, l, r - 1, now_right + w as i64);

                    state
                        .x_tree
                        .update(0, MAX_RANGE, now_right, updated_right - 1, r);
                    state
                        .x_tree
                        .update(0, MAX_RANGE, updated_right - sigma, updated_right - 1, r);

                    state.right = state.right.max(updated_right);
                    state.bottom = state.bottom.max(r as i64);
                    state.bottom_each_square[p] = r;
                    state.right_each_square[p] = updated_right;
                }
                _ => unreachable!(),
            }

            state.prdbs.push(prdb);
            state.prdbs_p.push(p);
            state.score = best_score;
            if cfg!(debug_assertions) {
                query(&state.prdbs);

                println!(
                            "# p: {} W: {} H: {} state.x_tree.query: {} state.y_tree.query: {} state.bottom_each_square[p]: {} state.right_each_square[p]: {}",
                            p,
                            state.right,
                            state.bottom,
                            state.x_tree.query(0, MAX_RANGE, 175234, 216115),
                            state.y_tree.query(0, MAX_RANGE, 323813, 334585),
                            state.bottom_each_square[p],
                            state.right_each_square[p],
                        );
            }
        }
    }
    return state;
}

mod jikken {
    use super::*;

    pub fn execute(state: &mut State, square: &Square, p: usize) -> Option<(Prdb, i64)> {
        let mut best_score = state.score;
        let mut prdb = None;

        for r in [ROTATE, NOT_ROTATE] {
            let (w, h) = if r == ROTATE {
                (square.h, square.w)
            } else {
                (square.w, square.h)
            };
            for d in [U, L] {
                for prdbs_pi in 0..state.prdbs_p.len() {
                    let b = state.prdbs_p[prdbs_pi] as i64;
                    let score = estimate(state, w, h, d, b);

                    if best_score >= score {
                        best_score = score;
                        prdb = Some(Prdb { p, r, d, b });
                    }
                }

                // b = -1 のとき
                {
                    let b = -1;
                    let score = estimate(state, w, h, d, b);

                    if best_score >= score {
                        best_score = score;
                        prdb = Some(Prdb { p, r, d, b });
                    }
                }
            }
        }

        if let Some(prdb) = prdb {
            Some((prdb, best_score))
        } else {
            None
        }
    }

    fn estimate(state: &mut State, w: i64, h: i64, d: char, b: i64) -> i64 {
        match d {
            U => {
                let right = if b == -1 {
                    0
                } else {
                    state.right_each_square[b as usize]
                };
                estimate_u(state, w, h, right)
            }
            L => {
                let bottom = if b == -1 {
                    0
                } else {
                    state.bottom_each_square[b as usize]
                };
                estimate_l(state, w, h, bottom)
            }
            _ => unreachable!(),
        }
    }

    fn estimate_u(state: &mut State, w: i64, h: i64, right: i64) -> i64 {
        let (l, r) = (right, right + w);
        let now_bottom = state.x_tree.query(0, MAX_RANGE, l, r - 1);

        // 更新対象区間についての更新後の下端
        let updated_bottom = now_bottom + h as i64;

        let diff_h = (state.bottom.max(updated_bottom) - state.bottom).max(0);
        let diff_w = (state.right.max(r as i64) - state.right).max(0);

        state.score + diff_h + diff_w - w - h
    }

    fn estimate_l(state: &mut State, w: i64, h: i64, bottom: i64) -> i64 {
        let (l, r) = (bottom, bottom + h);
        let now_right = state.y_tree.query(0, MAX_RANGE, l, r - 1);

        // 更新対象区間についての更新後の右端
        let updated_right = now_right + w as i64;

        let diff_w = (state.right.max(updated_right) - state.right).max(0);
        let diff_h = (state.bottom.max(r as i64) - state.bottom).max(0);

        state.score + diff_w + diff_h - w - h
    }
}

fn query(prdbs: &Vec<Prdb>) -> (i64, i64) {
    let mut stdin = LineSource::new(BufReader::new(io::stdin()));
    macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));

    println!("{}", prdbs.len());
    for prdb in prdbs.iter() {
        println!("{} {} {} {}", prdb.p, prdb.r, prdb.d, prdb.b);
    }
    if cfg!(debug_assertions) {
        return (0, 0);
    } else {
        input! {
            w: i64,
            h: i64,
        }
        return (w, h);
    }
}

mod ahc040_lib {
    #[derive(Default)]
    pub struct SegmentTree {
        left: Option<Box<SegmentTree>>,
        right: Option<Box<SegmentTree>>,
        max: i64,          // このノードがカバーする区間の最大値
        lazy: Option<i64>, // 遅延伝搬用 (置換値を保持)
    }

    impl SegmentTree {
        pub fn new() -> Self {
            SegmentTree {
                left: None,
                right: None,
                max: 0,
                lazy: None, // 遅延値はNoneが初期値 (未設定)
            }
        }

        /// 遅延値を適用
        fn push(&mut self, start: i64, end: i64) {
            if let Some(value) = self.lazy {
                if value > self.max {
                    self.max = value;
                    if start != end {
                        // 葉ノードでない場合
                        self.left
                            .get_or_insert_with(|| Box::new(SegmentTree::new()))
                            .lazy = Some(value);
                        self.right
                            .get_or_insert_with(|| Box::new(SegmentTree::new()))
                            .lazy = Some(value);
                    }
                }
                self.lazy = None; // 遅延値をクリア
            }
        }

        /// 区間 [l, r] を value で置換
        pub fn update(&mut self, start: i64, end: i64, l: i64, r: i64, value: i64) {
            self.push(start, end); // 遅延値を適用

            if r < start || end < l {
                // 更新範囲外
                return;
            }
            if l <= start && end <= r {
                // 現在の区間が完全に含まれる場合
                if value > self.max {
                    // 現在の最大値より大きい場合のみ更新
                    self.lazy = Some(value);
                    self.push(start, end);
                }
                return;
            }
            let mid = (start + end) / 2;
            self.left
                .get_or_insert_with(|| Box::new(SegmentTree::new()))
                .update(start, mid, l, r, value);
            self.right
                .get_or_insert_with(|| Box::new(SegmentTree::new()))
                .update(mid + 1, end, l, r, value);

            // 左右の子ノードの最大値を計算
            self.max = std::cmp::max(
                self.left.as_ref().map_or(0, |l| l.max),
                self.right.as_ref().map_or(0, |r| r.max),
            );
        }

        /// 区間 [l, r] の最大値を取得
        pub fn query(&mut self, start: i64, end: i64, l: i64, r: i64) -> i64 {
            self.push(start, end); // 遅延値を適用

            if r < start || end < l {
                // クエリ範囲外
                return 0;
            }
            if l <= start && end <= r {
                // 現在の区間が完全に含まれる場合
                return self.max;
            }
            let mid = (start + end) / 2;
            let left_max = self
                .left
                .as_mut()
                .map_or(0, |tree| tree.query(start, mid, l, r));
            let right_max = self
                .right
                .as_mut()
                .map_or(0, |tree| tree.query(mid + 1, end, l, r));
            std::cmp::max(left_max, right_max)
        }
    }
}

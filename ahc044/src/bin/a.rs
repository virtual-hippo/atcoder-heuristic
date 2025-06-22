#![allow(unused_imports)]
use itertools::*;
use proconio::{fastout, input};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

// left: 奇数回
// right: 偶数回
#[derive(Copy, Clone)]
struct Next(usize, usize);

#[fastout]
fn main() {
    input! {
        n: usize,
        _l: usize,
        t: [i64; n],
    }

    let sorted_t: Vec<(usize, i64)> = t
        .iter()
        .enumerate()
        .sorted_by(|(_, a), (_, b)| b.cmp(a))
        .map(|(i, &v)| (i, v))
        .collect();

    let mut ans = vec![Next(0, 0); n];
    for i in 1..n - 1 {
        let pi = sorted_t[i - 1].0;
        let ti = sorted_t[i].0;
        let ni = sorted_t[i + 1].0;
        ans[ti].0 = ni;
        ans[ti].1 = pi;
    }
    // 0
    {
        let ti = sorted_t[0].0;
        let ni = sorted_t[1].0;
        ans[ti].0 = ni;
        ans[ti].1 = ti;
    }
    // n
    {
        let pi = sorted_t[n - 2].0;
        let ti = sorted_t[n - 1].0;
        ans[ti].0 = sorted_t[0].0;
        ans[ti].1 = pi;
    }

    // -----------------------------------------------------------

    for i in 0..n {
        println!("{} {}", ans[i].0, ans[i].1);
    }
}

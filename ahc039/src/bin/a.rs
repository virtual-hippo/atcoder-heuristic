use proconio::{fastout, input};

use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[fastout]
fn main() {
    input! {
        n: usize,
    }
    let saba = init::fetch_fish(n);
    let iwashi = init::fetch_fish(n);

    solve(&saba, &iwashi);
}

struct Point(usize, usize);

#[derive(Clone)]
pub struct State {
    /// 各マスについて使うかどうかの状態保持
    grid: Vec<Vec<bool>>,
    /// 1 マスあたりの辺の長さ
    grid_size: usize,
    /// 各マスごとの魚の数
    fish_cnt_each_grid: Vec<Vec<(usize, usize)>>,
    /// 多角形の辺の合計
    len: i64,
}

impl State {
    fn default(d: usize, saba: &Vec<(usize, usize)>, iwashi: &Vec<(usize, usize)>) -> Self {
        let grid = vec![vec![true; d]; d];
        let grid_size = 100_000 / d;
        // すべてのマスを選択するので辺の長さの合計は 100_000 * 4
        let len = 100_000 * 4;
        let fish_cnt_each_grid = init::count_fish_each_grid(grid_size, saba, iwashi);
        State {
            grid,
            grid_size,
            fish_cnt_each_grid,
            len,
        }
    }

    fn divide_grid(&self, saba: &Vec<(usize, usize)>, iwashi: &Vec<(usize, usize)>) -> Self {
        let d = self.get_d() * 2;
        let grid_size = 100_000 / d;
        let mut grid = vec![vec![false; d]; d];
        for i in 0..self.grid.len() {
            for j in 0..self.grid.len() {
                grid[i * 2][j * 2] = self.grid[i][j];
                grid[i * 2][j * 2 + 1] = self.grid[i][j];
                grid[i * 2 + 1][j * 2] = self.grid[i][j];
                grid[i * 2 + 1][j * 2 + 1] = self.grid[i][j];
            }
        }
        let fish_cnt_each_grid = init::count_fish_each_grid(grid_size, saba, iwashi);
        State {
            grid,
            grid_size,
            fish_cnt_each_grid,
            len: self.len,
        }
    }

    fn get_score(&self) -> i64 {
        let mut a = 0;
        let mut b = 0;
        let d = self.get_d();
        for i in 0..d {
            for j in 0..d {
                if self.grid[i][j] {
                    a += self.fish_cnt_each_grid[i][j].0 as i64;
                    b += self.fish_cnt_each_grid[i][j].1 as i64;
                }
            }
        }
        0.max(a - b + 1)
    }

    /// 指定したマスを更新した際のスコアを求める
    fn get_new_score(&mut self, i: usize, j: usize) -> i64 {
        self.grid[i][j] = !self.grid[i][j];
        let score = self.get_score();
        self.grid[i][j] = !self.grid[i][j];
        score
    }

    fn get_d(&self) -> usize {
        100_000 / self.grid_size
    }
}

fn solve(saba: &Vec<(usize, usize)>, iwashi: &Vec<(usize, usize)>) {
    // 実行時間上限
    let time_limit = Duration::from_millis(1930);
    let start_time = Instant::now();

    let mut best_state = State::default(10, saba, iwashi);

    // 10
    best_state = optimization::optimize(start_time, time_limit, 20, &mut best_state);

    // 20
    let mut best_state = best_state.divide_grid(saba, iwashi);
    best_state = optimization::optimize(start_time, time_limit, 200, &mut best_state);

    // 40
    let mut best_state = best_state.divide_grid(saba, iwashi);
    best_state = optimization::optimize(start_time, time_limit, 150, &mut best_state);

    // 80
    let mut best_state = best_state.divide_grid(saba, iwashi);
    best_state = optimization::optimize(start_time, time_limit, 150, &mut best_state);

    // 160
    let mut best_state = best_state.divide_grid(saba, iwashi);
    best_state = optimization::optimize(start_time, time_limit, 150, &mut best_state);

    // 320
    let mut best_state = best_state.divide_grid(saba, iwashi);
    _ = optimization::optimize(start_time, time_limit, 150, &mut best_state);
}

mod output {
    pub fn print_answer(answer: &Vec<(usize, usize)>) {
        println!("{}", answer.len());
        for (x, y) in answer.iter() {
            println!("{} {}", x, y);
        }
    }

    #[allow(dead_code)]
    pub fn print_grid(grid: &Vec<Vec<bool>>) {
        let len = grid.len();
        for i in 0..len {
            for j in 0..len {
                if grid[i][j] {
                    print!("#");
                } else {
                    print!(".");
                }
            }
            println!("");
        }
    }
}

mod graph_to_answer {
    use super::*;

    pub fn graph_to_answer(
        graph: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> Vec<(usize, usize)> {
        let mut answer = vec![];
        let mut visited = HashSet::new();
        let pos = get_start_point(&graph).unwrap();
        dfs(&mut visited, graph, &mut answer, pos);

        // 同一辺上に複数の点が存在しないようにする
        let mut check_same_edge_flags = vec![true; answer.len()];
        for i in 1..answer.len() - 1 {
            if answer[i].0 == answer[i - 1].0 && answer[i].0 == answer[i + 1].0 {
                check_same_edge_flags[i] = false;
            }
            if answer[i].1 == answer[i - 1].1 && answer[i].1 == answer[i + 1].1 {
                check_same_edge_flags[i] = false;
            }
        }
        answer
            .iter()
            .enumerate()
            .filter(|(i, _)| check_same_edge_flags[*i])
            .map(|(_, &(i, j))| (j, 100_000 - i))
            .collect::<Vec<(usize, usize)>>()
    }

    fn get_start_point(
        graph: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    ) -> Option<(usize, usize)> {
        for k in graph.keys() {
            if let Some(points) = graph.get(k) {
                if points.len() == 2 {
                    return Some(*k);
                }
            }
        }
        None
    }

    fn dfs(
        visited: &mut HashSet<(usize, usize)>,
        graph: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        answer: &mut Vec<(usize, usize)>,
        pos: (usize, usize),
    ) {
        visited.insert(pos);
        answer.push(pos);
        if let Some(points) = graph.get(&pos) {
            for point in points.iter() {
                if !visited.contains(point) {
                    dfs(visited, graph, answer, *point);
                    break;
                }
            }
        }
    }
}

mod grid_to_graph {
    use super::*;

    pub fn grid_to_graph(state: &State) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
        let mut graph = HashMap::new();

        for i in 0..state.get_d() {
            for j in 0..state.get_d() {
                if !state.grid[i][j] {
                    continue;
                }
                if let Some([u, v]) = check_top(state, i, j) {
                    let u = (u.0, u.1);
                    let v = (v.0, v.1);
                    graph.entry(u).or_insert_with(|| vec![]).push(v);
                    graph.entry(v).or_insert_with(|| vec![]).push(u);
                }
                if let Some([u, v]) = check_bottom(state, i, j) {
                    let u = (u.0, u.1);
                    let v = (v.0, v.1);
                    graph.entry(u).or_insert_with(|| vec![]).push(v);
                    graph.entry(v).or_insert_with(|| vec![]).push(u);
                }
                if let Some([u, v]) = check_left(state, i, j) {
                    let u = (u.0, u.1);
                    let v = (v.0, v.1);
                    graph.entry(u).or_insert_with(|| vec![]).push(v);
                    graph.entry(v).or_insert_with(|| vec![]).push(u);
                }
                if let Some([u, v]) = check_right(state, i, j) {
                    let u = (u.0, u.1);
                    let v = (v.0, v.1);
                    graph.entry(u).or_insert_with(|| vec![]).push(v);
                    graph.entry(v).or_insert_with(|| vec![]).push(u);
                }
            }
        }
        // TODO: 辺上の点を取り除く
        graph
    }

    fn check_top(state: &State, i: usize, j: usize) -> Option<[Point; 2]> {
        let u = Point(i * state.grid_size, j * state.grid_size);
        let v = Point(i * state.grid_size, j * state.grid_size + state.grid_size);

        if i > 0 && !state.grid[i - 1][j] && state.grid[i][j] {
            return Some([u, v]);
        }
        if i == 0 && state.grid[i][j] {
            return Some([u, v]);
        }
        return None;
    }

    fn check_bottom(state: &State, i: usize, j: usize) -> Option<[Point; 2]> {
        let h = state.grid.len();

        let u = Point(i * state.grid_size + state.grid_size, j * state.grid_size);
        let v = Point(
            i * state.grid_size + state.grid_size,
            j * state.grid_size + state.grid_size,
        );

        if i < h - 1 && !state.grid[i + 1][j] && state.grid[i][j] {
            return Some([u, v]);
        }
        if i == h - 1 && state.grid[i][j] {
            return Some([u, v]);
        }
        return None;
    }

    fn check_left(state: &State, i: usize, j: usize) -> Option<[Point; 2]> {
        let u = Point(i * state.grid_size, j * state.grid_size);
        let v = Point(i * state.grid_size + state.grid_size, j * state.grid_size);

        if j > 0 && !state.grid[i][j - 1] && state.grid[i][j] {
            return Some([u, v]);
        }
        if j == 0 && state.grid[i][j] {
            return Some([u, v]);
        }
        return None;
    }

    fn check_right(state: &State, i: usize, j: usize) -> Option<[Point; 2]> {
        let w = state.grid[0].len();

        let u = Point(i * state.grid_size, j * state.grid_size + state.grid_size);
        let v = Point(
            i * state.grid_size + state.grid_size,
            j * state.grid_size + state.grid_size,
        );

        if j < w - 1 && !state.grid[i][j + 1] && state.grid[i][j] {
            return Some([u, v]);
        }
        if j == w - 1 && state.grid[i][j] {
            return Some([u, v]);
        }
        return None;
    }
}

mod optimization {
    use check_length::check_length;

    use super::*;

    pub fn optimize(
        start_time: Instant,
        time_limit: Duration,
        round: usize,
        state: &mut State,
    ) -> State {
        let mut best_score = 1;
        let mut best_answer = vec![];
        let mut best_state = state.clone();
        for i in 0..round {
            if start_time.elapsed() >= time_limit {
                break;
            }

            update(state, start_time, time_limit);
            let graph = grid_to_graph::grid_to_graph(&state);
            let answer = graph_to_answer::graph_to_answer(&graph);
            if state.get_score() > best_score {
                best_score = state.get_score();
                best_answer = answer.clone();
                best_state = state.clone();
            } else {
                let mm = if round > 50 { 10 } else { 5 };
                if i % mm == 0 {
                    *state = best_state.clone();
                }
            }
            if i % 10 == 0 {
                output::print_answer(&answer);
            }
        }
        if best_answer.len() != 0 {
            output::print_answer(&best_answer);
        }
        best_state
    }

    fn update(state: &mut State, start_time: Instant, time_limit: Duration) {
        for i in 0..state.get_d() {
            for j in 0..state.get_d() {
                if start_time.elapsed() >= time_limit {
                    break;
                }
                update_grid(state, (i, j));
            }
        }
    }

    fn update_grid(state: &mut State, (i, j): (usize, usize)) {
        if !is_connect((i, j), state) {
            return;
        }
        let length_diff = check_length::get_length_diff((i, j), state);
        if !check_length(length_diff, state) {
            return;
        }

        let d = state.get_d();
        let mut rng = rand::prelude::ThreadRng::default();
        let geta = if d <= 20 {
            rng.gen_range(0..25)
        } else if d <= 40 {
            rng.gen_range(0..15)
        } else if d <= 80 {
            rng.gen_range(0..7)
        } else {
            rng.gen_range(0..1)
        };

        if state.get_score() <= state.get_new_score(i, j) + geta {
            state.grid[i][j] = !state.grid[i][j];
            state.len += length_diff;
            return;
        }
    }

    /// 更新して連結性を保てられるか
    fn is_connect(point: (usize, usize), state: &mut State) -> bool {
        if state.grid[point.0][point.1] {
            can_erase(point, &state.grid)
        } else {
            can_add(point, &state.grid)
        }
    }

    fn can_erase((i, j): (usize, usize), grid: &Vec<Vec<bool>>) -> bool {
        let x = shokyo_kanosei::get_x((i, j), grid).map(|b| if b { 1 } else { 0 });
        x[1] + x[3] + x[5] + x[7]
            - x[0] * x[1] * x[3]
            - x[1] * x[2] * x[5]
            - x[3] * x[6] * x[7]
            - x[5] * x[7] * x[8]
            == 1
            && !is_kakomareteru((i, j), grid)
    }

    fn can_add((i, j): (usize, usize), grid: &Vec<Vec<bool>>) -> bool {
        let x = shokyo_kanosei::get_x((i, j), grid).map(|b| if b { 0 } else { 1 });
        x[1] + x[3] + x[5] + x[7]
            - x[0] * x[1] * x[3]
            - x[1] * x[2] * x[5]
            - x[3] * x[6] * x[7]
            - x[5] * x[7] * x[8]
            == 1
            && !is_kakomareteinai((i, j), grid)
    }

    // 上下左右全てがtureか
    fn is_kakomareteru((i, j): (usize, usize), grid: &Vec<Vec<bool>>) -> bool {
        (0 < i && grid[i - 1][j])
            && (i < grid.len() - 1 && grid[i + 1][j])
            && (0 < j && grid[i][j - 1])
            && (j < grid.len() - 1 && grid[i][j + 1])
    }

    // 上下左右全てがfalseか
    fn is_kakomareteinai((i, j): (usize, usize), grid: &Vec<Vec<bool>>) -> bool {
        (i == 0 || !grid[i - 1][j])
            && (i == grid.len() - 1 || !grid[i + 1][j])
            && (j == 0 || !grid[i][j - 1])
            && (j == grid.len() - 1 || !grid[i][j + 1])
    }

    mod shokyo_kanosei {
        pub fn get_x((i, j): (usize, usize), grid: &Vec<Vec<bool>>) -> [bool; 9] {
            let mut x = [
                false, false, false, false, grid[i][j], false, false, false, false,
            ];
            x[0] = if i > 0 && j > 0 {
                grid[i - 1][j - 1]
            } else {
                false
            };
            x[1] = if i > 0 { grid[i - 1][j] } else { false };
            x[2] = if i > 0 && j < grid.len() - 1 {
                grid[i - 1][j + 1]
            } else {
                false
            };
            x[3] = if j > 0 { grid[i][j - 1] } else { false };
            x[5] = if j < grid.len() - 1 {
                grid[i][j + 1]
            } else {
                false
            };

            x[6] = if i < grid.len() - 1 && j > 0 {
                grid[i + 1][j - 1]
            } else {
                false
            };
            x[7] = if i < grid.len() - 1 {
                grid[i + 1][j]
            } else {
                false
            };
            x[8] = if i < grid.len() - 1 && j < grid.len() - 1 {
                grid[i + 1][j + 1]
            } else {
                false
            };
            x
        }
    }

    mod check_length {
        use super::*;

        pub fn check_length(length_diff: i64, state: &mut State) -> bool {
            state.len as i64 + length_diff <= 100_000 * 4
        }

        /// マスを更新した際の長さの差分を求める
        pub fn get_length_diff(point: (usize, usize), state: &mut State) -> i64 {
            let cnt = count(point, state);
            match cnt {
                1 => -2 * state.get_d() as i64,
                2 => 0,
                3 => 2 * state.get_d() as i64,
                _ => unreachable!(),
            }
        }

        // 左右上下で接しているマスの内自身のマスと異なるマスの個数を数える
        fn count(point: (usize, usize), state: &State) -> usize {
            if state.grid[point.0][point.1] {
                count_to_off(point, state)
            } else {
                count_to_on(point, state)
            }
        }

        // マスを更新した際に左右上下で接しているマスの内 true の個数を数える
        fn count_to_off((i, j): (usize, usize), state: &State) -> usize {
            let mut ret = 0;
            if i > 0 && state.grid[i - 1][j] {
                ret += 1;
            }
            if j > 0 && state.grid[i][j - 1] {
                ret += 1;
            }
            if i < state.grid.len() - 1 && state.grid[i + 1][j] {
                ret += 1;
            }
            if j < state.grid.len() - 1 && state.grid[i][j + 1] {
                ret += 1;
            }
            ret
        }

        // マスを更新した際に左右上下で接しているマスの内 false or 壁 の個数を数える
        fn count_to_on((i, j): (usize, usize), state: &State) -> usize {
            let mut ret = 0;
            if i == 0 || !state.grid[i - 1][j] {
                ret += 1;
            }
            if j == 0 || !state.grid[i][j - 1] {
                ret += 1;
            }
            if i == state.grid.len() - 1 || !state.grid[i + 1][j] {
                ret += 1;
            }
            if j == state.grid.len() - 1 || !state.grid[i][j + 1] {
                ret += 1;
            }
            ret
        }
    }
}

mod init {
    use super::*;

    pub fn fetch_fish(n: usize) -> Vec<(usize, usize)> {
        let mut fish = vec![];
        for _ in 0..n {
            input! {
                j: usize,
                i: usize,
            }
            let j = j.min(99_999);
            let i = i.max(1);
            fish.push((100_000 - i, j));
        }
        fish
    }

    pub fn count_fish_each_grid(
        grid_size: usize,
        saba: &Vec<(usize, usize)>,
        iwashi: &Vec<(usize, usize)>,
    ) -> Vec<Vec<(usize, usize)>> {
        let d = 100_000 / grid_size;
        let mut ret = vec![vec![(0, 0); d]; d];
        for &(i, j) in saba.iter() {
            ret[i / grid_size][j / grid_size].0 += 1;
        }
        for &(i, j) in iwashi.iter() {
            ret[i / grid_size][j / grid_size].1 += 1;
        }

        ret
    }
}

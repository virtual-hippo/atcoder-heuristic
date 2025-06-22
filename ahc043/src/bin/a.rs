use itertools::*;
use proconio::input;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use rustc_hash::FxHashSet;

use rand::Rng;

const COST_STATION: i64 = 5000;
const COST_RAIL: i64 = 100;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Pos(usize, usize);
impl Pos {
    fn new((r, c): (usize, usize)) -> Self {
        Self(r, c)
    }
}

#[derive(Clone)]
struct UnionFind {
    n: usize,
    /// 親頂点の番号
    parents: Vec<usize>,
    /// 連結成分の頂点の個数
    sizes: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            n,
            parents: vec![usize::MAX; n * n],
            sizes: vec![1; n * n],
        }
    }

    fn pos_to_index(&self, pos: Pos) -> usize {
        pos.0 * self.n + pos.1
    }

    fn find_root(&mut self, idx: usize) -> usize {
        if self.parents[idx] == usize::MAX {
            return idx;
        }
        let root = self.find_root(self.parents[idx]);
        self.parents[idx] = root;
        root
    }

    fn is_same(&mut self, p: Pos, q: Pos) -> bool {
        let p_idx = self.pos_to_index(p);
        let q_idx = self.pos_to_index(q);
        self.find_root(p_idx) == self.find_root(q_idx)
    }

    fn unite(&mut self, p: Pos, q: Pos) {
        let p_idx = self.pos_to_index(p);
        let q_idx = self.pos_to_index(q);
        let p_root = self.find_root(p_idx);
        let q_root = self.find_root(q_idx);

        if p_root != q_root {
            if self.sizes[p_root] < self.sizes[q_root] {
                self.parents[p_root] = q_root;
                self.sizes[q_root] += self.sizes[p_root];
            } else {
                self.parents[q_root] = p_root;
                self.sizes[p_root] += self.sizes[q_root];
            }
        }
    }
}

fn calc_distance(pos1: &Pos, pos2: &Pos) -> i64 {
    let pos1 = (pos1.0 as i64, pos1.1 as i64);
    let pos2 = (pos2.0 as i64, pos2.1 as i64);

    (pos2.0 - pos1.0).abs() + (pos2.1 - pos1.1).abs()
}

#[derive(Clone, Debug, PartialEq)]
enum Building {
    Station,
    HorizontalRail,
    VerticalRail,
    LeftDownRail,
    LeftUpRail,
    RightUpRail,
    RightDownRail,
}

impl Building {
    fn is_station(&self) -> bool {
        matches!(self, Building::Station)
    }

    fn is_rail(&self) -> bool {
        matches!(
            self,
            Building::HorizontalRail
                | Building::VerticalRail
                | Building::LeftDownRail
                | Building::LeftUpRail
                | Building::RightUpRail
                | Building::RightDownRail
        )
    }

    fn is_up(&self) -> bool {
        matches!(
            self,
            Building::Station
                | Building::VerticalRail
                | Building::LeftUpRail
                | Building::RightUpRail
        )
    }

    fn is_down(&self) -> bool {
        matches!(
            self,
            Building::Station
                | Building::VerticalRail
                | Building::LeftDownRail
                | Building::RightDownRail
        )
    }

    fn is_left(&self) -> bool {
        matches!(
            self,
            Building::Station
                | Building::HorizontalRail
                | Building::LeftDownRail
                | Building::LeftUpRail
        )
    }

    fn is_right(&self) -> bool {
        matches!(
            self,
            Building::Station
                | Building::HorizontalRail
                | Building::RightDownRail
                | Building::RightUpRail
        )
    }
    fn to_string(&self) -> String {
        match self {
            Building::Station => "0".to_string(),
            Building::HorizontalRail => "1".to_string(),
            Building::VerticalRail => "2".to_string(),
            Building::LeftDownRail => "3".to_string(),
            Building::LeftUpRail => "4".to_string(),
            Building::RightUpRail => "5".to_string(),
            Building::RightDownRail => "6".to_string(),
        }
    }
}

#[derive(Clone, PartialEq)]
enum Action {
    DoNothing,
    Build((Building, Pos)),
}

impl Action {
    fn to_string(&self) -> String {
        match self {
            Action::DoNothing => "-1".to_string(),
            Action::Build((building, pos)) => {
                format!("{} {} {}", building.to_string(), pos.0, pos.1)
            }
        }
    }
}

#[derive(Clone)]
struct Answer<'a> {
    actions: &'a Vec<Action>,
    score: i64,
    income: i64,
}

impl<'a> Answer<'a> {
    fn new(actions: &'a Vec<Action>, score: i64, income: i64) -> Self {
        Self {
            actions,
            score,
            income,
        }
    }
}

#[derive(Clone)]
struct Field {
    n: usize,
    fields: Vec<Vec<Option<Building>>>,
    uf: UnionFind,
}

impl Field {
    fn new(n: usize) -> Self {
        Self {
            n,
            fields: vec![vec![None; n]; n],
            uf: UnionFind::new(n),
        }
    }

    fn build(&mut self, building: &Building, Pos(r, c): Pos) {
        // すでに建てられている場合はエラー
        if let Some(ref existing) = self.fields[r][c] {
            debug_assert!(!existing.is_station());
            if building.is_rail() {
                debug_assert!(self.fields[r][c].is_none());
            }
        }
        self.fields[r][c] = Some(building.clone());

        // 隣接する区画と接続
        // 上
        if building.is_up() && r > 0 {
            if let Some(ref other) = self.fields[r - 1][c] {
                if other.is_down() {
                    self.uf.unite(Pos(r, c), Pos(r - 1, c));
                }
            }
        }
        // 下
        if building.is_down() && r < self.n - 1 {
            if let Some(ref other) = self.fields[r + 1][c] {
                if other.is_up() {
                    self.uf.unite(Pos(r, c), Pos(r + 1, c));
                }
            }
        }
        // 左
        if building.is_left() && c > 0 {
            if let Some(ref other) = self.fields[r][c - 1] {
                if other.is_right() {
                    self.uf.unite(Pos(r, c), Pos(r, c - 1));
                }
            }
        }
        // 右
        if building.is_right() && c < self.n - 1 {
            if let Some(ref other) = self.fields[r][c + 1] {
                if other.is_left() {
                    self.uf.unite(Pos(r, c), Pos(r, c + 1));
                }
            }
        }
    }

    // 通勤できるかどうか判定する
    fn is_connected(&mut self, s: Pos, t: Pos) -> bool {
        debug_assert!(calc_distance(&s, &t) > 4);

        let stations_s = self.collect_stations(&s);
        let stations_t = self.collect_stations(&t);

        for station_s in stations_s.iter() {
            for station_t in stations_t.iter() {
                if self.uf.is_same(*station_s, *station_t) {
                    return true;
                }
            }
        }
        false
    }

    fn collect_stations(&self, Pos(rr, cc): &Pos) -> Vec<Pos> {
        itertools::iproduct!(-2_i64..3, -2_i64..3)
            .filter(|&(dr, dc)| dr.abs() + dc.abs() <= 2)
            .map(|(dr, dc)| ((*rr as i64 + dr), (*cc as i64 + dc)))
            .filter(|&(r, c)| {
                r >= 0
                    && r < self.n as i64
                    && c >= 0
                    && c < self.n as i64
                    && self.fields[r as usize][c as usize].is_some()
                    && self.fields[r as usize][c as usize]
                        .as_ref()
                        .unwrap()
                        .is_station()
            })
            .map(|(r, c)| Pos(r as usize, c as usize))
            .collect()
    }

    fn is_connected_rail(&self, s: Pos, t: Pos) -> bool {
        if s.0 != t.0 && s.1 != t.1 {
            panic!("s.0 != t.0 && s.1 != t.1");
        }

        if s.0 == t.0 && s.1 == t.1 {
            panic!("s.0 == t.0 && s.1 == t.1");
        }

        if s.0 == t.0 {
            if s.1 < t.1 {
                if let Some(ref rail_s) = self.fields[s.0][s.1] {
                    if let Some(ref rail_t) = self.fields[t.0][t.1] {
                        if rail_s.is_right() && rail_t.is_left() {
                            return true;
                        }
                    }
                }
            } else {
                if let Some(ref rail_s) = self.fields[s.0][s.1] {
                    if let Some(ref rail_t) = self.fields[t.0][t.1] {
                        if rail_s.is_left() && rail_t.is_right() {
                            return true;
                        }
                    }
                }
            }
        } else if s.1 == t.1 {
            if s.0 < t.0 {
                if let Some(ref rail_s) = self.fields[s.0][s.1] {
                    if let Some(ref rail_t) = self.fields[t.0][t.1] {
                        if rail_s.is_down() && rail_t.is_up() {
                            return true;
                        }
                    }
                }
            } else {
                if let Some(ref rail_s) = self.fields[s.0][s.1] {
                    if let Some(ref rail_t) = self.fields[t.0][t.1] {
                        if rail_s.is_up() && rail_t.is_down() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

struct SolverInput {
    n: usize,
    m: usize,
    k: usize,
    t: usize,
    home: Vec<Pos>,
    workspace: Vec<Pos>,
    distance: Vec<i64>,
}

impl SolverInput {
    fn new() -> Self {
        input! {
            // N は区画の縦・横の数で、 N=50 を満たす。
            n: usize,
            // M はR国の人の数で、 50≤M≤1600 を満たす。
            m: usize,
            // K は鉄道会社Xの初期資金であり、 11000≤K≤20000 を満たす。
            k: usize,
            //T はゲームのターン数であり、 T=800 を満たす。
            t: usize,
        }

        let mut home = Vec::new();
        let mut workspace = Vec::new();
        let mut distance = Vec::new();

        for _ in 0..m {
            input! {
                i: usize,
                j: usize,
            }
            let home_i = Pos(i, j);

            input! {
                i: usize,
                j: usize,
            }
            let workspace_i = Pos(i, j);

            home.push(home_i);
            workspace.push(workspace_i);
            distance.push(calc_distance(&home_i, &workspace_i));
        }

        Self {
            n,
            m,
            k,
            t,
            home,
            workspace,
            distance,
        }
    }
}

struct SolverInfo {
    // 区画の周りの建物の数を降順にソートしたもの
    total_buildings_around_cells: Vec<(usize, (usize, usize))>,
    // 区画の周りで働いている or 住んでいる人々
    people_around_cell: Vec<Vec<Vec<usize>>>,

    // 繋げると点数が高くなる組み合わせを降順にソートしたもの
    yoi_pair_list: Vec<(i64, (Pos, Pos))>,

    // 繋げると収入が増える組み合わせを降順にソートしたもの
    high_income_pair_list: Vec<(i64, (Pos, Pos))>,

    // 寄せたワークスペース
    hosei_home: Vec<Pos>,
    hosei_workspace: Vec<Pos>,
}

// 駅を建てた際に通勤可能になるか判定する
fn can_tsukin_if_build_station(home: &Pos, workspace: &Pos, stations: &(Pos, Pos)) -> bool {
    (calc_distance(&home, &stations.0) <= 2 && calc_distance(&workspace, &stations.1) <= 2)
        || (calc_distance(&home, &stations.1) <= 2 && calc_distance(&workspace, &stations.0) <= 2)
}

impl SolverInfo {
    const TAKE_COUNT: usize = 250;

    fn new(
        input: &SolverInput,
        time_limit: &Duration,
        start_time: &Instant,
        get_shikiichi: fn(usize) -> usize,
        is_hosei: bool,
    ) -> Self {
        let mut field = vec![vec![vec![]; input.n]; input.n];
        for pi in 0..input.m {
            let Pos(i, j) = input.home[pi];
            field[i][j].push(pi);
            let Pos(i, j) = input.workspace[pi];
            field[i][j].push(pi);
        }

        // cell の情報を集める
        let mut total_buildings_around_cell = vec![vec![0; input.n]; input.n];
        let mut people_around_cell: Vec<Vec<Vec<usize>>> = vec![vec![vec![]; input.n]; input.n];
        itertools::iproduct!(0..input.n, 0..input.n, -2_i64..3, -2_i64..3)
            .filter(|&(_i, _j, dr, dc)| dr.abs() + dc.abs() <= 2)
            .filter(|&(i, j, dr, dc)| {
                (i as i64 + dr) >= 0
                    && (i as i64 + dr) < input.n as i64
                    && (j as i64 + dc) >= 0
                    && (j as i64 + dc) < input.n as i64
            })
            .map(|(i, j, dr, dc)| {
                let r = (i as i64 + dr) as usize;
                let c = (j as i64 + dc) as usize;
                (i, j, r, c)
            })
            .for_each(|(i, j, r, c)| {
                total_buildings_around_cell[i][j] += field[r][c].len();
                people_around_cell[i][j].extend(field[r][c].iter());
            });

        let total_buildings_around_cells = iproduct!(0..input.n, 0..input.n)
            .filter(|&(i, j)| !is_hosei || (i % 3 == 1 && j % 3 == 0))
            .map(|(i, j)| (total_buildings_around_cell[i][j], (i, j)))
            .sorted_by(|a, b| b.0.cmp(&a.0))
            .collect::<Vec<_>>();

        // debug_assert_eq!(
        //     total_buildings_around_cells
        //         .iter()
        //         .all(|(_, (i, j))| i % 3 == 1 && j % 3 == 0),
        //     true
        // );

        let yoi_pair_list = Self::create_yoi_pair_list(
            input,
            &total_buildings_around_cells,
            &people_around_cell,
            &start_time,
            &time_limit,
            get_shikiichi,
            //&FxHashSet::default(),
        );

        let high_income_pair_list = Self::_create_high_income_pair_list(
            input,
            &total_buildings_around_cells,
            &people_around_cell,
            &start_time,
            &time_limit,
            get_shikiichi,
            // &FxHashSet::default(),
        );

        Self {
            total_buildings_around_cells,
            people_around_cell,
            yoi_pair_list,
            high_income_pair_list,
            hosei_home: input
                .home
                .iter()
                .map(|pos| if is_hosei { Self::hosei(pos) } else { *pos })
                .collect(),
            hosei_workspace: input
                .workspace
                .iter()
                .map(|pos| if is_hosei { Self::hosei(pos) } else { *pos })
                .collect(),
        }
    }

    // 軸に寄せる
    fn hosei(pos: &Pos) -> Pos {
        let Pos(mut i, mut j) = *pos;

        {
            if i % 3 == 2 {
                i -= 1
            }
            if i % 3 == 0 {
                if i + 1 < 50 {
                    i += 1
                } else {
                    i -= 2
                }
            }
        }

        {
            if j % 3 == 1 {
                j -= 1
            }
            if j % 3 == 2 {
                if j + 1 < 50 {
                    j += 1
                } else {
                    j -= 2
                }
            }
        }

        debug_assert_eq!(i % 3, 1);
        debug_assert_eq!(j % 3, 0);
        Pos(i, j)
    }

    fn create_yoi_pair_list(
        input: &SolverInput,
        total_buildings_around_cells: &Vec<(usize, (usize, usize))>,
        people_around_cell: &Vec<Vec<Vec<usize>>>,
        start_time: &Instant,
        time_limit: &Duration,
        get_shikiichi: fn(usize) -> usize,
        //_tukindekiru_people: &FxHashSet<usize>,
    ) -> Vec<(i64, (Pos, Pos))> {
        let mut yoi_pair_list = Vec::with_capacity(total_buildings_around_cells.len());

        let shikiichi = get_shikiichi(input.m);
        let take_count = Self::TAKE_COUNT.min(total_buildings_around_cells.len());

        for i in 0..(take_count - 1) {
            if start_time.elapsed() >= *time_limit {
                break;
            }

            // 高スコアが見込まれないものは間引く
            if total_buildings_around_cells[i].0 < shikiichi {
                continue;
            }

            for j in i + 1..take_count {
                if start_time.elapsed() >= *time_limit {
                    break;
                }

                // 高スコアが見込まれないものは間引く
                if total_buildings_around_cells[j].0 < shikiichi {
                    continue;
                }
                let pair = (
                    Pos::new(total_buildings_around_cells[i].1),
                    Pos::new(total_buildings_around_cells[j].1),
                );

                let distance = calc_distance(&pair.0, &pair.1);
                let kenchiku_cost = (distance - 1) * COST_RAIL + COST_STATION * 2;
                let money = input.k as i64;

                // スコアが多いものを計算する
                let income = {
                    let building_count = distance + 1;
                    let nothing_count = (input.t as i64) - building_count;

                    let people1 = &people_around_cell[pair.0 .0][pair.0 .1];
                    let people2 = &people_around_cell[pair.1 .0][pair.1 .1];

                    // pos1, pos2 両方に属する人を探す
                    let peoples = people1
                        .iter()
                        //.filter(|&pi1| !_tukindekiru_people.contains(pi1))
                        .filter(|&pi1| people2.contains(pi1))
                        .collect::<Vec<_>>();

                    let income = peoples
                        .iter()
                        .filter(|&&pi| {
                            let home = input.home[*pi];
                            let workspace = input.workspace[*pi];
                            can_tsukin_if_build_station(&home, &workspace, &pair)
                        })
                        .map(|pi| input.distance[**pi])
                        .sum::<i64>();

                    income * nothing_count
                };
                let score = if kenchiku_cost <= money {
                    income + money - kenchiku_cost
                } else {
                    money
                };
                yoi_pair_list.push((score, pair));
            }
        }

        yoi_pair_list.sort_by(|a, b| b.0.cmp(&a.0));

        let mut set = FxHashSet::default();

        let new = {
            let mut new = vec![];
            let mut exists = vec![];
            for &(score, pair) in yoi_pair_list.iter() {
                if exists.len() == 0 {
                    new.push((score, pair));
                    exists.push(pair.0);
                    exists.push(pair.1);
                    continue;
                }
                if start_time.elapsed() > *time_limit {
                    break;
                }

                let mut new1 = pair.0;
                let mut new2 = pair.1;

                for exist_pos in exists.iter() {
                    if start_time.elapsed() > *time_limit {
                        break;
                    }

                    if new1 == pair.0 && calc_distance(&pair.0, exist_pos) <= 2 {
                        new1 = *exist_pos;
                    }
                    if new2 == pair.1 && calc_distance(&pair.1, exist_pos) <= 2 {
                        new2 = *exist_pos;
                    }
                    if new1 != pair.0 && new2 != pair.1 {
                        break;
                    }
                }

                if set.contains(&(new1, new2)) {
                } else {
                    set.insert((new1, new2));
                    new.push((score, (new1, new2)));
                }

                if new1 == pair.0 {
                    exists.push(pair.0);
                }
                if new2 == pair.1 {
                    exists.push(pair.1);
                }
            }
            new
        };

        new.into_iter().take(300).collect()
    }

    fn _create_high_income_pair_list(
        input: &SolverInput,
        total_buildings_around_cells: &Vec<(usize, (usize, usize))>,
        people_around_cell: &Vec<Vec<Vec<usize>>>,
        start_time: &Instant,
        time_limit: &Duration,
        get_shikiichi: fn(usize) -> usize,
        // _tukindekiru_people: &FxHashSet<usize>,
    ) -> Vec<(i64, (Pos, Pos))> {
        let mut high_income_pair_list = Vec::with_capacity(total_buildings_around_cells.len());
        let shikiichi = get_shikiichi(input.m);
        let take_count = Self::TAKE_COUNT.min(total_buildings_around_cells.len());

        for i in 0..(take_count - 1) {
            if start_time.elapsed() >= *time_limit {
                break;
            }

            // 高スコアが見込まれないものは間引く
            if total_buildings_around_cells[i].0 < shikiichi {
                continue;
            }

            for j in i + 1..take_count {
                if start_time.elapsed() >= *time_limit {
                    break;
                }

                // 高スコアが見込まれないものは間引く
                if total_buildings_around_cells[j].0 < shikiichi {
                    continue;
                }
                let pair = (
                    Pos::new(total_buildings_around_cells[i].1),
                    Pos::new(total_buildings_around_cells[j].1),
                );

                let distance = calc_distance(&pair.0, &pair.1);

                // 収入が高いものを計算する
                let income = {
                    let building_count = distance + 1;
                    let nothing_count = (input.t as i64) - building_count;

                    let people1 = &people_around_cell[pair.0 .0][pair.0 .1];
                    let people2 = &people_around_cell[pair.1 .0][pair.1 .1];

                    // pos1, pos2 両方に属する人を探す
                    let peoples = people1
                        .iter()
                        //.filter(|&pi1| !_tukindekiru_people.contains(pi1))
                        .filter(|&pi1| people2.contains(pi1))
                        .collect::<Vec<_>>();

                    let income = peoples
                        .iter()
                        .filter(|&&pi| {
                            let home = input.home[*pi];
                            let workspace = input.workspace[*pi];
                            can_tsukin_if_build_station(&home, &workspace, &pair)
                        })
                        .map(|pi| input.distance[**pi])
                        .sum::<i64>();

                    income * nothing_count
                };
                high_income_pair_list.push((income, pair));
            }
        }

        high_income_pair_list.sort_by(|a, b| b.0.cmp(&a.0));

        let mut set = FxHashSet::default();

        let new = {
            let mut new = vec![];
            let mut exists = vec![];
            for &(score, pair) in high_income_pair_list.iter() {
                if start_time.elapsed() > *time_limit {
                    break;
                }
                if exists.len() == 0 {
                    new.push((score, pair));
                    exists.push(pair.0);
                    exists.push(pair.1);
                    continue;
                }

                let mut new1 = pair.0;
                let mut new2 = pair.1;

                for exist_pos in exists.iter() {
                    if start_time.elapsed() > *time_limit {
                        break;
                    }
                    if new1 == pair.0 && calc_distance(&pair.0, exist_pos) <= 2 {
                        new1 = *exist_pos;
                    }
                    if new2 == pair.1 && calc_distance(&pair.1, exist_pos) <= 2 {
                        new2 = *exist_pos;
                    }
                    if new1 != pair.0 && new2 != pair.1 {
                        break;
                    }
                }

                if set.contains(&(new1, new2)) {
                } else {
                    set.insert((new1, new2));
                    new.push((score, (new1, new2)));
                }

                if new1 == pair.0 {
                    exists.push(pair.0);
                }
                if new2 == pair.1 {
                    exists.push(pair.1);
                }
            }
            new
        };
        new.into_iter().take(300).collect()
    }
}

#[derive(Clone)]
struct SolverState {
    is_connected: Vec<bool>,
    field: Field,
    money: i64,
    actions: Vec<Action>,
    income: i64,
    // 建設できなかった駅
    station_queue: VecDeque<Pos>,
    // 建設住みの駅
    stations: Vec<Pos>,
    //tukindekiru_people: FxHashSet<usize>,
}

impl SolverState {
    fn new(input: &SolverInput) -> Self {
        let mut is_connected = Vec::with_capacity(input.m);
        is_connected.resize(input.m, false);

        let field = Field::new(input.n);

        let actions = Vec::with_capacity(input.t);

        Self {
            is_connected,
            field,
            money: input.k as i64,
            actions,
            income: 0,
            station_queue: VecDeque::new(),
            stations: Vec::new(),
            //tukindekiru_people: FxHashSet::default(),
        }
    }
}
enum SolverError {
    NotEnoughMoney(i64),
    TooManyActions,
}

struct Solver<'a> {
    input: &'a SolverInput,
    info: SolverInfo,
    initial_state: SolverState,
    state: SolverState,
    best_actions: Vec<Action>,
    limit: usize,
}

impl<'a> Solver<'a> {
    const MID: usize = 500;
    const TEMP: i64 = 50;
    fn new(input: &'a SolverInput, time_limit: &Duration, start_time: &Instant) -> Self {
        let initial_state = SolverState::new(input);
        let state = initial_state.clone();
        let get_shikiichi = |m: usize| {
            if m < 200 {
                7
            } else if m < 500 {
                9
            } else if m < 1000 {
                (m * 15) / 1000
            } else {
                (m * 15) / 1000
            }
        };
        Self {
            input,
            info: SolverInfo::new(input, time_limit, start_time, get_shikiichi, input.m > 500),
            initial_state,
            state,
            best_actions: vec![Action::DoNothing; input.t],
            limit: Self::MID,
        }
    }

    fn update_is_connected(&mut self) {
        for i in 0..self.input.m {
            if !self.state.is_connected[i]
                && self
                    .state
                    .field
                    .is_connected(self.input.home[i], self.input.workspace[i])
            {
                self.state.is_connected[i] = true;
                //self.state.tukindekiru_people.insert(i);
            }
        }
    }

    fn calc_income(&mut self) -> i64 {
        (0..self.input.m)
            .filter(|&i| self.state.is_connected[i])
            .map(|i| self.input.distance[i])
            .sum()
    }

    fn build_rail(&mut self, building: &Building, Pos(r, c): Pos) -> Result<(), SolverError> {
        if self.state.money < COST_RAIL {
            return Err(SolverError::NotEnoughMoney(COST_RAIL));
        }
        if self.state.actions.len() >= self.limit {
            return Err(SolverError::TooManyActions);
        }

        // すでにレールが建てられている場合はスキップ
        if let Some(_) = self.state.field.fields[r][c].as_ref() {
            return Ok(());
        }

        self.state.field.build(building, Pos(r, c));
        self.state.money -= COST_RAIL;
        self.state
            .actions
            .push(Action::Build((building.clone(), Pos(r, c))));

        if r > 0 {
            if let Some(exisiting) = self.state.field.fields[r - 1][c].as_ref() {
                if exisiting.is_station() && building.is_up() {
                    self.update_is_connected();
                    self.state.income = self.calc_income();
                }
            }
        }
        if r < self.input.n - 1 {
            if let Some(exisiting) = self.state.field.fields[r + 1][c].as_ref() {
                if exisiting.is_station() && building.is_down() {
                    self.update_is_connected();
                    self.state.income = self.calc_income();
                }
            }
        }
        if c > 0 {
            if let Some(exisiting) = self.state.field.fields[r][c - 1].as_ref() {
                if exisiting.is_station() && building.is_left() {
                    self.update_is_connected();
                    self.state.income = self.calc_income();
                }
            }
        }
        if c < self.input.n - 1 {
            if let Some(exisiting) = self.state.field.fields[r][c + 1].as_ref() {
                if exisiting.is_station() && building.is_right() {
                    self.update_is_connected();
                    self.state.income = self.calc_income();
                }
            }
        }

        self.collect_money();
        Ok(())
    }

    fn build_station(&mut self, Pos(r, c): Pos) -> Result<(), SolverError> {
        if self.state.money < COST_STATION {
            self.state.station_queue.push_back(Pos(r, c));
            return Err(SolverError::NotEnoughMoney(COST_STATION));
        }
        if self.state.actions.len() >= self.limit {
            return Err(SolverError::TooManyActions);
        }

        // すでに駅が建てられている場合はスキップ
        if let Some(existing) = self.state.field.fields[r][c].as_ref() {
            if existing.is_station() {
                return Ok(());
            }
        }

        self.state.field.build(&Building::Station, Pos(r, c));
        self.state.money -= COST_STATION;
        self.state.stations.push(Pos(r, c));
        self.state
            .actions
            .push(Action::Build((Building::Station, Pos(r, c))));

        self.update_is_connected();
        self.state.income = self.calc_income();

        self.collect_money();
        Ok(())
    }

    fn buildnothing(&mut self) -> Result<(), SolverError> {
        if self.state.actions.len() >= self.limit {
            return Err(SolverError::TooManyActions);
        }
        self.state.actions.push(Action::DoNothing);
        self.collect_money();
        Ok(())
    }

    fn collect_money(&mut self) {
        self.state.money += self.state.income;
    }

    fn _replace_donothing(&mut self, start_idx: usize) {
        for i in start_idx..self.state.actions.len() {
            self.state.actions[i] = Action::DoNothing;
        }
    }

    fn connect_points(&mut self, pos0: Pos, pos1: Pos) -> Result<(), SolverError> {
        let mut pre: Option<Pos> = None;

        // 接続済みは繋げない
        if self.state.field.uf.is_same(pos0, pos1) {
            return Ok(());
        }

        if (calc_distance(&pos0, &pos1) as usize) + self.state.actions.len() >= self.limit {
            return Err(SolverError::TooManyActions);
        }

        // 繋げても収入が増えない場合はreturn
        {
            let people1 = &self.info.people_around_cell[pos0.0][pos0.1];
            let people2 = &self.info.people_around_cell[pos1.0][pos1.1];

            // pos1, pos2 それぞれの周辺の人々で繋がっていない人を探す
            let is_not_connected_people_count = people1
                .iter()
                .filter(|&pi| !self.state.is_connected[*pi])
                .count()
                + people2
                    .iter()
                    .filter(|&pi| !self.state.is_connected[*pi])
                    .count();

            if is_not_connected_people_count < 1 {
                return Ok(());
            }
        }

        // TODO: 期待追加収益が見込み費用よりも低い場合はreturn

        // pos0 -> pos1 への垂直のレールを建てる
        {
            if pos0.0 < pos1.0 {
                // 垂直のレールを建てる
                for r in pos0.0 + 1..pos1.0 {
                    let pos = Pos(r, pos0.1);
                    self.build_rail(&Building::VerticalRail, pos)?;

                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }

                // ゴールへ向けて曲がったレールを建てる
                {
                    let pos = Pos(pos1.0, pos0.1);
                    if pos0.1 < pos1.1 {
                        self.build_rail(&Building::RightUpRail, pos)?;
                    } else if pos0.1 > pos1.1 {
                        self.build_rail(&Building::LeftUpRail, pos)?;
                    }
                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }
            } else if pos0.0 > pos1.0 {
                // 垂直のレールを建てる
                for r in ((pos1.0 + 1)..pos0.0).rev() {
                    let pos = Pos(r, pos0.1);
                    self.build_rail(&Building::VerticalRail, pos)?;
                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }

                // ゴールへ向けて曲がったレールを建てる
                {
                    let pos = Pos(pos1.0, pos0.1);
                    if pos0.1 < pos1.1 {
                        self.build_rail(&Building::RightDownRail, pos)?;
                    } else if pos0.1 > pos1.1 {
                        self.build_rail(&Building::LeftDownRail, pos)?;
                    }
                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }
            }
        }

        // pos0 -> pos1 への水平のレールを建てる
        {
            // 水平のレールを建てる
            if pos0.1 < pos1.1 {
                for c in pos0.1 + 1..pos1.1 {
                    let pos = Pos(pos1.0, c);
                    self.build_rail(&Building::HorizontalRail, pos)?;
                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }
            }
            // 水平のレールを建てる
            else if pos0.1 > pos1.1 {
                for c in ((pos1.1 + 1)..pos0.1).rev() {
                    let pos = Pos(pos1.0, c);
                    self.build_rail(&Building::HorizontalRail, pos)?;
                    if let Some(pre) = pre {
                        if !self.state.field.is_connected_rail(pre, pos) {
                            if let Err(_) = self.build_station(pre) {
                                //
                            }
                        }
                    }
                    pre = Some(pos);
                }
            }
        }

        // 駅を建てる
        while let Err(_) = self.build_station(pos0) {
            self.buildnothing()?;
        }
        while let Err(_) = self.build_station(pos1) {
            self.buildnothing()?;
        }

        Ok(())
    }

    fn execute(
        &mut self,
        time_limit: &Duration,
        start_time: &Instant,
        // .0 は周辺に建物が多い区画, .1 は仕事場 or 家
        pos_pair_list_: &Vec<(Pos, Pos)>,
    ) -> Answer {
        let mut pos_pair_queue: VecDeque<(Pos, Pos)> = VecDeque::new();
        for pos_pair in pos_pair_list_.iter() {
            pos_pair_queue.push_back(*pos_pair);
        }

        let mut rng = rand::prelude::ThreadRng::default();
        let random_value = rng.gen_range(0..100);

        while self.state.actions.len() < self.limit && pos_pair_queue.len() > 0 {
            if start_time.elapsed() >= *time_limit {
                break;
            }

            while self.state.station_queue.len() > 0 && self.state.money >= COST_STATION {
                let pos = self.state.station_queue.pop_front().unwrap();
                if let Err(SolverError::TooManyActions) = self.build_station(pos) {
                    break;
                }
            }

            // 建築を放棄した駅がたまりすぎたら貯蓄する
            // 2 - 5 を試したけど 3 が一番良かった
            if self.state.station_queue.len() > 5 {
                if let Err(SolverError::TooManyActions) = self.buildnothing() {
                    break;
                } else {
                    continue;
                }
            }

            // 最後は何もしない
            // HACK: 再検討の余地あり
            if self.state.actions.len() > 730 {
                if let Err(SolverError::TooManyActions) = self.buildnothing() {
                    break;
                } else {
                    continue;
                }
            }

            let (mut pos0, pos1) = pos_pair_queue.pop_front().unwrap();

            // 建築済みの近い駅を探す
            for &station in self.state.stations.iter() {
                if calc_distance(&station, &pos1) < calc_distance(&pos0, &pos1) {
                    pos0 = station;
                }
            }

            let result_connect = self.connect_points(pos0, pos1);

            match result_connect {
                Ok(_) => {
                    if rng.gen_range(30..90) < random_value {
                        pos_pair_queue.make_contiguous().sort_by(|a, b| {
                            calc_distance(&b.0, &b.1).cmp(&calc_distance(&a.0, &a.1))
                        });
                    }
                }
                Err(SolverError::NotEnoughMoney(cost)) => {
                    if cost == COST_RAIL {
                        while self.state.actions.len() < self.limit && self.state.money < COST_RAIL
                        {
                            if let Err(SolverError::TooManyActions) = self.buildnothing() {
                                break;
                            }
                        }
                    }
                    pos_pair_queue.push_front((pos0, pos1));
                }
                Err(SolverError::TooManyActions) => {
                    break;
                }
            }
        }

        debug_assert!(self.state.actions.len() <= self.limit);
        // もしターン数を超えてしまっていたら取り除く
        while self.state.actions.len() > self.limit {
            self.state.actions.pop();
        }

        // もしターン数を超えていない場合はDoNothingを追加
        while self.state.actions.len() < self.limit {
            if let Err(SolverError::TooManyActions) = self.buildnothing() {
                break;
            }
        }

        Answer::new(&self.state.actions, self.state.money, self.state.income)
    }

    fn print_best_actions(&mut self) {
        while self.best_actions.len() < self.input.t {
            self.best_actions.push(Action::DoNothing);
        }
        println!(
            "{}",
            self.best_actions.iter().map(|a| a.to_string()).join("\n")
        );
    }

    // 周辺に建物が多い区画を順に試す
    fn solve2(
        &mut self,
        time_limit: &Duration,
        start_time: &Instant,
        best_score: &mut i64,
        best_income_with_money: &mut i64,
        _best_state_cache: &mut SolverState,
    ) {
        let mut rng = rand::prelude::ThreadRng::default();
        let random_value = rng.gen_range(0..100);
        let limit = self.limit;

        let yoi_pair_list = self
            .info
            .yoi_pair_list
            .iter()
            .take(rng.gen_range(10..21))
            .map(|(_, pair)| *pair)
            .chain(
                self.info
                    .high_income_pair_list
                    .iter()
                    .take(rng.gen_range(10..21))
                    .map(|(_, pair)| *pair),
            )
            .collect::<Vec<_>>();

        let pi_list_list = yoi_pair_list
            .iter()
            .map(|(pos0, pos1)| {
                let mut ret = vec![];

                // HACK 高速化余地あり
                let people0 = &self.info.people_around_cell[pos0.0][pos0.1];
                let people1 = &self.info.people_around_cell[pos1.0][pos1.1];

                // push people0
                for &pi in people0.iter() {
                    let home = self.info.hosei_home[pi];
                    let workspace = self.info.hosei_workspace[pi];
                    let pair = if calc_distance(&home, &pos0) <= 2 {
                        (*pos0, workspace)
                    } else {
                        (*pos0, home)
                    };

                    let distance = calc_distance(&pair.0, &pair.1);

                    debug_assert_eq!(distance > 2, true);

                    if distance > 10 {
                        ret.push(pair);
                    }
                }

                // push people1
                for &pi in people1.iter() {
                    let home = self.info.hosei_home[pi];
                    let workspace = self.info.hosei_workspace[pi];
                    let pair = if calc_distance(&home, &pos1) <= 2 {
                        (*pos1, workspace)
                    } else {
                        (*pos1, home)
                    };

                    let distance = calc_distance(&pair.0, &pair.1);

                    debug_assert_eq!(distance > 2, true);

                    if distance > 10 {
                        ret.push(pair);
                    }
                }

                ret.sort_by(|a, b| calc_distance(&b.0, &b.1).cmp(&calc_distance(&a.0, &a.1)));

                ret
            })
            .collect::<Vec<Vec<(Pos, Pos)>>>();

        let pi_list_list2 = self
            .info
            .total_buildings_around_cells
            .iter()
            .take(rng.gen_range(10..21))
            .map(|(_, pos)| {
                (
                    Pos(pos.0, pos.1),
                    &self.info.people_around_cell[pos.0][pos.1],
                )
            })
            .map(|(pos, people)| {
                let mut ret = vec![];
                for &pi in people.iter() {
                    let home = self.info.hosei_home[pi];
                    let workspace = self.info.hosei_workspace[pi];
                    let pair = if calc_distance(&home, &pos) <= 2 {
                        (pos, workspace)
                    } else {
                        (pos, home)
                    };

                    let distance = calc_distance(&pair.0, &pair.1);

                    debug_assert_eq!(distance > 2, true);

                    if distance > 10 {
                        ret.push(pair);
                    }
                }

                ret.iter()
                    .sorted_by(|a, b| calc_distance(&b.0, &b.1).cmp(&calc_distance(&a.0, &a.1)))
                    .take(rng.gen_range(8..15))
                    .cloned()
                    .collect()
            })
            .collect::<Vec<Vec<(Pos, Pos)>>>();

        let merged_iter = yoi_pair_list
            .iter()
            .chain(pi_list_list.iter().flatten())
            .chain(pi_list_list2.iter().flatten());

        for _i in 0..20 {
            if start_time.elapsed() >= *time_limit {
                break;
            }

            let merged: Vec<_> = merged_iter
                .clone()
                .filter(|_| rng.gen_range(50..150) < random_value)
                .map(|&v| v)
                .collect();
            let Answer {
                actions,
                score,
                income,
            } = self.execute(&time_limit, &start_time, &merged);

            // TODO: Fix hard coding
            if score > *best_score && limit == 800 {
                *best_score = score;
                self.best_actions = actions.clone();
            }

            if limit == Self::MID
                // && self.state.money != (self.input.k as i64)
                && income + self.state.money > 0.max(*best_income_with_money - Self::TEMP)
            {
                *best_income_with_money = income + self.state.money;
                *_best_state_cache = self.state.clone();
            }

            self.state = self.initial_state.clone();
        }
    }

    fn solve(&mut self, time_limit: &Duration, start_time: &Instant) -> i64 {
        let mut best_score = self.input.k as i64;
        let mut best_income_with_money = 0;
        let mut best_state_cache = SolverState::new(&self.input);
        eprintln!("solve start: {} ms", start_time.elapsed().as_millis());

        let pre_time_limit = Duration::from_millis(2750);

        for _i in 0..3 {
            if start_time.elapsed() > *time_limit {
                return best_score;
            }

            // 途中時点の最適な値を求める
            // limit を一時的に MIDにする
            self.limit = Self::MID;
            for _ in 0..18 {
                self.solve2(
                    &pre_time_limit,
                    start_time,
                    &mut best_score,
                    &mut best_income_with_money,
                    &mut best_state_cache,
                );
                self.state = self.initial_state.clone();
            }
            // limit を元に戻す
            self.limit = self.input.t;
            eprintln!(
                "#init serach {}: {} ms",
                _i,
                start_time.elapsed().as_millis()
            );

            if start_time.elapsed() > *time_limit {
                return best_score;
            }

            self.initial_state = best_state_cache.clone();

            // 末尾の DoNothing を削除する
            while best_state_cache.actions.last() == Some(&Action::DoNothing) {
                best_state_cache.actions.pop();
                best_state_cache.money -= best_state_cache.income;
            }

            for _ in 0..3 {
                self.state = self.initial_state.clone();
                self.solve2(
                    time_limit,
                    start_time,
                    &mut best_score,
                    &mut best_income_with_money,
                    &mut best_state_cache,
                );
            }

            self.initial_state = SolverState::new(&self.input);
            self.state = self.initial_state.clone();
        }

        if start_time.elapsed() > *time_limit {
            return best_score;
        }

        // ------------------------------------------
        // m が小さい場合の解法
        // ------------------------------------------
        if self.input.m > 500 {
            return best_score;
        }

        self.initial_state = SolverState::new(&self.input);
        self.state = self.initial_state.clone();
        self.info = SolverInfo::new(
            &self.input,
            time_limit,
            start_time,
            |m: usize| {
                if m < 100 {
                    3
                } else if m < 170 {
                    5
                } else if m < 300 {
                    6
                } else if m < 500 {
                    4
                } else {
                    20
                }
            },
            false,
        );
        // eprintln!("#created info2: {} ms", start_time.elapsed().as_millis());
        for _i in 0..50 {
            if start_time.elapsed() > *time_limit {
                return best_score;
            }

            self.solve2(
                time_limit,
                start_time,
                &mut best_score,
                &mut best_income_with_money,
                &mut best_state_cache,
            );
        }

        best_score
    }
}

fn main() {
    let time_limit = Duration::from_millis(2935);
    let start_time = Instant::now();

    let input = SolverInput::new();

    let mut solver = Solver::new(&input, &time_limit, &start_time);
    let best_score = solver.solve(&time_limit, &start_time);

    println!("#time: {}", start_time.elapsed().as_millis());
    println!("#score: {}", best_score);
    solver.print_best_actions();
}

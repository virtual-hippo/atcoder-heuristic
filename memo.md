# Rustで競プロする際のメモ

## 良く使うライブラリ
```rust
use proconio::input;
use std::collections::HashSet;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::collections::BinaryHeap;
use proconio::marker::Chars;
use itertools::Itertools; // p.iter().permutations(n)
use regex::Regex; // ←あまり使わん
```

## usize
### saturating_sub: ゼロにクリップして減算する
```rust
    let a: usize = 5;
    let b: usize = 10;

    let result = a.saturating_sub(b);
    println!("Result: {}", result); // 0
```
saturating_sub

## cast 系
### char to num
```rust
let c: char = '5';
let num: i32 = c as i32 - 48;
```

## 文字列操作系
### 大文字変換
```rust
let mut s = String::from("Grüße, Jürgen ❤");
s.make_ascii_uppercase();
assert_eq!("GRüßE, JüRGEN ❤", s);
```

### 小数点
```rust
 // 少数第7まで表示
println!("{:.7}", n);
```

### N進数
```rust
// 16進数0埋め, 文字列長2
println!("{:>02X}", n); 

// 8進数にしたときに7が含まれるか
for i in 1..n+1 {
    let d = format!("{}", i);
    let o = format!("{:>03o}", i);
    if d.contains("7") || o.contains("7") {
        //
    } else {
        cnt +=1;
    }
}
```

### str to num
```rust
let str_num = "111";

// 2進数
println!("{}", u64::from_str_radix(str_num, 2).unwrap());
```
```rust
let stri: String = String::from("5");
let num: i32 = stri.parse().unwrap();
```

### num to string
```rust
let num: i32 = 5;
let stri: String = num.to_string();
```

### アルファベットを順に出力
```rust
// https://www.k-cube.co.jp/wakaba/server/ascii_code.html
// ABCDEFGHIJKLMNOPQRSTUVWXYZ
let large_a = 65_u8;
(0..26).for_each(|i| print!("{}", (large_a + i) as char));

// abcdefghijklmnopqrstuvwxyz
let small_a = 97_u8;
(0..26).for_each(|i| print!("{}", (small_a + i) as char));
```

### 回文チェック
```rust
fn is_kaibun(s: &Vec<char>) -> bool{
    let s_len = s.len();
    for i in 0..s_len/2 {
        if s[i] != s[s_len-1-i] {
            return false;
        }
    }
    true
}
```


```rust
let text = "there".to_string();
let rh = RollingHash::new(text.as_bytes());
assert_eq!(rh.is_palindrome(0, text.len()), false);
```

### 大文字と小文字
```rust
let large_chars = vec!['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
let small_chars = vec!['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
```

### 大文字と小文字の相互変換

```rust
fn flip(c: &char) -> char {
    ((*c as u8) ^ 32) as char
}

fn flip(c: &char) -> char {
    if c.is_lowercase() {
        c.to_ascii_uppercase()
    } else if c.is_uppercase() {
        c.to_ascii_lowercase()
    } else {
        unreachable!()
    }
}

```

## HashMap
### entry API
```rust
 *map.entry(a[i]).or_insert(0) += 1;
 map.entry(s[j]).or_insert_with(|| vec![]).push(j);
```

## Vec
### sort
```rust
// 昇順
vec.sort();

// 降順
vec.sort_by(|a, b| b.cmp(a));

// ソート前の位置を保持したままソート
let mut vec_with_ind = vec.iter().enumerate().map(|(i,x)| (i, *x)).collect::<Vec<(usize,usize)>>();
vec_with_ind.sort_by(|a, b| a.1.cmp(&b.1));
```

### min, max
```rust
let min = v.iter().min().unwrap();
let max = v.iter().max().unwrap();

// fold()を使う場合
let min = x_vec.iter().fold(dummy_max, |min, x| std::cmp::min(min, *x));
let max = x_vec.iter().fold(dummy_min, |max, x| std::cmp::max(max, *x));
```

### sum
```rust
let sum = v.iter().sum::<usize>();

// fold()を使う場合
let sum = v.iter().fold(0, |sum, x| sum + *x);
```

### 一部を新しく作る
```rust
let s = vec!['a', 'b', 'c', 'd', 'e']; 
let new_s = s[0..(s.len()-1)/2].iter().map(|&ch| ch).collect();
```

## DP
### 部分和DP
```rust
// a: Vec<usize>
// s: aの合計
let mut dp = vec![vec![false; s+1]; n+1];
dp[0][0] = true;
for i in 1..n+1 {
    for j in 0..s+1 {
        if dp[i-1][j] {
            dp[i][j] = true;
        }
        if j < a[i-1] {
            // 何もしない
        } else if dp[i-1][j-a[i-1]] {
            dp[i][j] = true;
        }
    }
}
```

## 座標系
### 距離の2乗計算(2次元)
```rust
fn calc_d(pos1: (i64, i64), pos2: (i64, i64)) -> i64 {
    (pos2.0 - pos1.0).pow(2) + (pos2.1 - pos1.1).pow(2) 
}
```

### 距離の2乗計算(N次元)
```rust
fn calc_d(pos1: &Vec<i64>, pos2: &Vec<i64>) -> i64 {
    (0..pos1.len()).fold(0, |sum, x| sum + (pos2[x]-pos1[x]).pow(2))
}
```

### 3点が同じ直線上に存在するか
```rust
fn is_triangle(p1: (i64, i64), p2: (i64, i64), p3: (i64, i64)) -> bool {
    let v1 = (p2.0 - p1.0, p2.1 - p1.1);
    let v2 = (p3.0 - p2.0, p3.1 - p2.1);
    v1.0 * v2.1 != v1.1 * v2.0
}
```

## 探索系
### ベル数
```rust
/// [ABC390 D - Stone XOR](https://atcoder.jp/contests/abc390/tasks/abc390_d)

fn dfs(a: &Vec<usize>, i: usize, groups: &mut Vec<Vec<usize>>) {
    if i == a.len() {
        // グループ分け後の処理
        return;
    }

    // 各グループに a[i] を入れてから潜る
    for j in 0..groups.len() {
        groups[j].push(i);

        dfs(a, i + 1, groups);

        groups[j].pop();
    }

    // グループを追加してから潜る
    groups.push(vec![i]);
    dfs(a, i + 1, groups);
    groups.pop();
}
```

### bit全探索
```rust
for bit in 0..(1<<n) {
    for i in 0..n {
        if bit & (1 << i) != 0 {
            //
        } else {
            //
        }
    }
}
```

### N 進法についての全探索
```rust
// 3進法を例にする

// N 進法で表現したときに最大 8 桁までを探索する
let n = 8;

let p3 = (0..n).fold(vec![1_u64], |mut acc, i| {
    let v = acc[i] * 3;
    acc.push(v);
    acc
});

for s in 0..p3[n] {
    // `s / p3[i] % 3` で i 桁目の数字を取り出している
    // 一見何をやっているかイメージしづらいが, 10進法で考えるとイメージしやすい
    for v in (0..n).map(|i| s / p3[i] % 3) {
        match v {
            0 => {
                // do nothing
            },
            1 => {
                // do something
            },
            2 => {
                // do something else
            },
            _ => unreachable!(),
        }
    }
}
```

### 二分探索
#### lower_bound
```rust
// https://docs.rs/superslice/latest/superslice/trait.Ext.html#tymethod.lower_bound
use superslice::Ext;

let a = [10, 11, 13, 13, 15];
assert_eq!(a.lower_bound(&9), 0);
assert_eq!(a.lower_bound(&10), 0);
assert_eq!(a.lower_bound(&11), 1);
assert_eq!(a.lower_bound(&12), 2);
assert_eq!(a.lower_bound(&13), 2);
assert_eq!(a.lower_bound(&14), 4);
assert_eq!(a.lower_bound(&15), 4);
assert_eq!(a.lower_bound(&16), 5);

```

#### upper_bound
```rust
// https://docs.rs/superslice/latest/superslice/trait.Ext.html#tymethod.upper_bound
use superslice::Ext;
let a = [10, 11, 13, 13, 15];
assert_eq!(a.upper_bound(&9), 0);
assert_eq!(a.upper_bound(&10), 1);
assert_eq!(a.upper_bound(&11), 2);
assert_eq!(a.upper_bound(&12), 2);
assert_eq!(a.upper_bound(&13), 4);
assert_eq!(a.upper_bound(&14), 4);
assert_eq!(a.upper_bound(&15), 5);
assert_eq!(a.upper_bound(&16), 5);
```



### BTreeSet で lower bound 的なことをやりたいときに使うやつ
```rs
// https://stackoverflow.com/questions/48575866/how-to-get-the-lower-bound-and-upper-bound-of-an-element-in-a-btreeset
use std::collections::BTreeSet;
fn neighbors(tree: &BTreeSet<usize>, val: usize) -> (Option<&usize>, Option<&usize>) {
    use std::ops::Bound::*;

    let mut before = tree.range((Unbounded, Excluded(val)));
    let mut after = tree.range((Excluded(val), Unbounded));

    (before.next_back(), after.next())
}
```

## 数学
### 最大公約数
```rust
use num_integer::gcd;
```

```rust
fn gcd(x: usize, y: usize) -> usize {
    let mut xy = (x, y);
    while xy.0 >= 1 && xy.1 >= 1 {
        if xy.0 < xy.1 {
            xy.1 %= xy.0;
        } else {
            xy.0 %= xy.1;
        }
    }
    if xy.0 >= 1 {
        xy.0
    } else {
        xy.1
    }
}
```

### 最小公倍数
```rust
fn lcm(x: usize, y: usize) -> usize {
    let d = gcd(x,y);
    x / d * y
}
```

### 期待値を mod 998244353 で出力
```rust
    // https://strangerxxx.hateblo.jp/entry/20230419/1681873929
    // 期待値 = p / q
    let denominator = modular_inv(q, MOD);
    (p * denominator) % MOD

```

### モジュラ逆数
```rust
// モジュラ逆数を求める
fn modular_inv(a: usize, m: usize) -> usize {
    power(a, m - 2, m)
}

// aのb乗をmで割った余りを返す関数
fn power(a: usize, b: usize, m: usize) -> usize {
    let mut p = a;
    let mut ret = 1;
    for i in 0..30 {
        let wari = 1 << i;
        if (b / wari) % 2 == 1 {
            ret = (ret * p) % m;
        }
        p = (p * p) % m;
    }
    ret
}
```

### 約数列挙

```rust
// 約数列挙(1 ~ M) までの自然数についてそれぞれ約数を列挙する
const M: usize = 100_005;
let mut divs = vec![vec![]; M];
for i in 1..=M {
    let mut j = i * 2;
    while j < M {
        divs[j].push(i);
        j += i;
    }
}
```

## グリッド問題の盆栽

```rust
enum Dir {
    UP,
    DOWN,
    LEFT,
    RIGHT,
}

fn can_move(s: &Vec<Vec<char>>, (i, j): (usize, usize), dir: Dir) -> bool {
    match dir {
        Dir::UP => i > 0 && s[i - 1][j] == '.',
        Dir::DOWN => i < s.len() - 1 && s[i + 1][j] == '.',
        Dir::LEFT => j > 0 && s[i][j - 1] == '.',
        Dir::RIGHT => j < s[0].len() - 1 && s[i][j + 1] == '.',
    }
}

```


## bit 操作

### k 桁目のビットを取得する

```rust
let num = 0b101101; // 例: 2進数 101101 (10進数で 45)
let k = 0;
assert_eq!((num >> k) & 1, 1);
let k = 1;
assert_eq!((num >> k) & 1, 0);
let k = 2;
assert_eq!((num >> k) & 1, 1);
let k = 3;
assert_eq!((num >> k) & 1, 1);
```

### k 桁目のビットを1にする
```rust
let mut num = 0b101101; // 例: 2進数 101101 (10進数で 45)

let k = 1;
num |= 1 << k;
assert_eq!(num, 0b101111);

let k = 2;
num |= 1 << k;
assert_eq!(num, 0b101111);
```

### k 桁目のビットを0にする
```rust
let mut num = 0b101101; // 例: 2進数 101101 (10進数で 45)

let k = 0;
num &= !(1 << k);
assert_eq!(num, 0b101100);

let k = 2;
num &= !(1 << k);
assert_eq!(num, 0b101000);
```

## グラフ
### 木の重心

```rust
// abc348-E
// https://atcoder.jp/contests/abc348/editorial/9706
// 木の重心を求める
fn dfs(
    graph: &Vec<Vec<usize>>,
    // 親頂点
    parent: usize,
    u: usize,
    // 木全体の頂点の重みの総和
    total: &u64,
    // 重心
    x: &mut usize,
    // 頂点の重み
    c: &Vec<u64>,
) -> u64 {
    let mut ret = c[u];
    let mut mx = 0;
    for &v in &graph[u] {
        if v == parent {
            continue;
        }
        // 部分木の頂点の重みの総和
        let now = dfs(graph, u, v, total, x, c);
        ret += now;
        mx = mx.max(now);
    }
    // 親頂点の部分木の頂点の重みの総和
    let parent_size = *total - ret;
    mx = mx.max(parent_size);

    // どの部分木の頂点の重みの総和も 1/2 * total の時, 重心になる
    if mx * 2 <= *total {
        *x = u;
    }
    ret
}
```

## うまく分類できないの
### 区間スケジュール問題
```rust
// https://atcoder.jp/contests/abc131/tasks/abc131_d
input! {
    n: usize,
    mut ab: [(usize, usize); n],
}
ab.sort_by(|(_,a), (_,b)| a.cmp(b));
let mut current_time = 0;
for (a, b) in ab.iter() {
    current_time += *a;
    if current_time > *b {
        println!("No");
        return;
    }
}
println!("Yes");
```

### インタラクティブな問題のおなじない
```rust
use proconio::input;
use proconio::source::line::LineSource;
use std::io::{self, BufReader};

fn main() {
    let mut stdin = LineSource::new(BufReader::new(io::stdin()));
    macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));

    input! {
        n: usize,
    }
}
```

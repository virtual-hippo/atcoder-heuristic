use proconio::{fastout, input};

#[fastout]
fn main() {
    // input! {
    //     n: usize,
    // }
    let mut root = SegmentTree::new(0, 10_000_000_000);

    // 更新操作
    // root.update(1, 5); // 位置1に5を設定
    // root.update(3, 5); // 位置3に7を設定
    // root.update(2, 8); // 位置1に5を設定
    // root.update(5, 8); // 位置3に7を設定

    // クエリ操作
    let result = root.query_max(1, 2); // 位置1から3の範囲の最大値
    println!("Maximum in range [1, 2]: {}", result);
    println!("{}", i64::MAX);
}

use std::cell::RefCell;
use std::cmp::{max, min};
use std::rc::Rc;

#[derive(Debug)]
struct SegmentTree {
    start: u64,
    end: u64,
    sum: i64,
    max: i64,
    left: Option<Rc<RefCell<SegmentTree>>>,
    right: Option<Rc<RefCell<SegmentTree>>>,
}

impl SegmentTree {
    // コンストラクタ
    fn new(start: u64, end: u64) -> Self {
        SegmentTree {
            start,
            end,
            sum: 0,
            max: 0,
            left: None,
            right: None,
        }
    }

    // 更新操作
    fn update(&mut self, idx: u64, val: i64) {
        if self.start == self.end {
            self.sum = val;
            self.max = val;
            return;
        }

        let mid = (self.start + self.end) / 2;
        if idx <= mid {
            if self.left.is_none() {
                self.left = Some(Rc::new(RefCell::new(SegmentTree::new(self.start, mid))));
            }
            self.left.as_ref().unwrap().borrow_mut().update(idx, val);
        } else {
            if self.right.is_none() {
                self.right = Some(Rc::new(RefCell::new(SegmentTree::new(mid + 1, self.end))));
            }
            self.right.as_ref().unwrap().borrow_mut().update(idx, val);
        }

        self.sum = self.left.as_ref().map_or(0, |left| left.borrow().sum)
            + self.right.as_ref().map_or(0, |right| right.borrow().sum);
        self.max = max(
            self.left.as_ref().map_or(0, |left| left.borrow().max),
            self.right.as_ref().map_or(0, |right| right.borrow().max),
        );
    }

    // クエリ操作
    fn query_max(&self, l: u64, r: u64) -> i64 {
        if l > self.end || r < self.start {
            return 0;
        }
        if l <= self.start && r >= self.end {
            return self.max;
        }

        let left_max = self
            .left
            .as_ref()
            .map_or(0, |left| left.borrow().query_max(l, r));
        let right_max = self
            .right
            .as_ref()
            .map_or(0, |right| right.borrow().query_max(l, r));
        max(left_max, right_max)
    }
}

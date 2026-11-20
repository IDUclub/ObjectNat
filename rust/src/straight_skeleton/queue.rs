//! Indexed binary min-heap: one entry per slot, so the queue never holds more entries than there
//! are live wavefront vertices and edges (memory stays linear in the input size).

pub(crate) struct EventQueue {
    heap: Vec<(f64, u32)>,
    /// slot -> position in `heap` plus one (0 = not queued)
    pos: Vec<u32>,
}

/// Order by time, then even slots before odd ones, then by slot.
#[inline]
fn less(a: (f64, u32), b: (f64, u32)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && (a.1 & 1, a.1) < (b.1 & 1, b.1))
}

impl EventQueue {
    pub(crate) fn new() -> Self {
        EventQueue {
            heap: Vec::new(),
            pos: Vec::new(),
        }
    }

    /// Insert `slot` with `time`, or move it to `time` if it is already queued.
    pub(crate) fn set(&mut self, slot: u32, time: f64) {
        let s = slot as usize;
        if s >= self.pos.len() {
            self.pos.resize(s + 1, 0);
        }
        let p = self.pos[s];
        if p == 0 {
            self.heap.push((time, slot));
            let i = self.heap.len() - 1;
            self.pos[s] = i as u32 + 1;
            self.up(i);
        } else {
            let i = p as usize - 1;
            let old = self.heap[i].0;
            self.heap[i].0 = time;
            if time < old {
                self.up(i);
            } else {
                self.down(i);
            }
        }
    }

    pub(crate) fn remove(&mut self, slot: u32) {
        let s = slot as usize;
        if s >= self.pos.len() || self.pos[s] == 0 {
            return;
        }
        let i = self.pos[s] as usize - 1;
        self.pos[s] = 0;
        let last = self.heap.pop().expect("heap is not empty");
        if i < self.heap.len() {
            self.heap[i] = last;
            self.pos[last.1 as usize] = i as u32 + 1;
            self.up(i);
            let j = self.pos[last.1 as usize] as usize - 1;
            self.down(j);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<(f64, u32)> {
        let top = *self.heap.first()?;
        self.remove(top.1);
        Some(top)
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    fn up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if less(self.heap[i], self.heap[p]) {
                self.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            let mut m = i;
            if l < n && less(self.heap[l], self.heap[m]) {
                m = l;
            }
            if r < n && less(self.heap[r], self.heap[m]) {
                m = r;
            }
            if m == i {
                break;
            }
            self.swap(i, m);
            i = m;
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.pos[self.heap[i].1 as usize] = i as u32 + 1;
        self.pos[self.heap[j].1 as usize] = j as u32 + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::EventQueue;

    #[test]
    fn orders_and_updates() {
        let mut q = EventQueue::new();
        q.set(3, 5.0);
        q.set(1, 2.0);
        q.set(7, 9.0);
        q.set(2, 1.0);
        q.set(7, 0.5);
        q.remove(1);
        q.set(3, 10.0);
        let mut out = Vec::new();
        while let Some((t, s)) = q.pop() {
            out.push((t, s));
        }
        assert_eq!(out, vec![(0.5, 7), (1.0, 2), (10.0, 3)]);
        q.set(5, 1.0);
        q.set(4, 1.0);
        q.set(1, 1.0);
        assert_eq!(q.pop(), Some((1.0, 4)));
        assert_eq!(q.pop(), Some((1.0, 1)));
    }
}
